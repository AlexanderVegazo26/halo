// Hostile-template tests (security review S-3, S-8, S-10; docs/security-hardening.md).
//
// Every S-3 evidence row runs in a forked death-test child, on a std::thread with the
// default stack (8 MiB, like a server worker), under an alarm. The child exits 0 only if the
// row threw halo::Error with the expected code and a message naming the expected bound, so a
// crash (SIGSEGV), hang (SIGALRM), bad_alloc or silent success is a test failure rather than
// the death of the whole test binary.
//
// Demonstrating a row red: HALO_TEMPLATE_LIMITS_DISABLE=<field>[,<field>...] zeroes the named
// TemplateLimits fields in rows() (0 disables a limit), e.g.
//   HALO_TEMPLATE_LIMITS_DISABLE=max_render_depth ./test_template --gtest_filter='*MacroRecursion*'
// makes the unbounded-recursion row crash again.

#include <gtest/gtest.h>
#include <pthread.h>
#include <sys/resource.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "halo/core/error.h"
#include "halo/template/chat_template.h"

namespace fs = std::filesystem;
using halo::ErrorCode;
using halo::chat::ChatTemplate;
using halo::chat::OrderedJson;
using halo::chat::RenderOptions;
using halo::chat::RenderStats;
using halo::chat::TemplateLimits;

namespace {

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define HALO_TEST_ASAN 1
#endif
#endif
#if defined(HALO_TEST_ASAN)
// ASan + Debug costs ~20x per template step, and ctest runs rows in parallel (the step-cap
// row took 84 s alone, 295 s under ctest -j16): ASan checks memory safety, not speed, so only
// a hang guard applies there. The release build enforces the "completes quickly" budget.
constexpr unsigned kRowAlarmSeconds = 1200;  // below ctest's default 1500 s timeout
constexpr double kRowSecondsBudget = 1200.0;
#else
constexpr unsigned kRowAlarmSeconds = 60;
constexpr double kRowSecondsBudget = 10.0;  // "completes quickly"; D-001: dev-host timing
#endif

TemplateLimits rows() {
    TemplateLimits l;
    const char* env = std::getenv("HALO_TEMPLATE_LIMITS_DISABLE");
    if (env == nullptr) return l;
    std::stringstream ss{std::string(env)};
    std::string name;
    while (std::getline(ss, name, ',')) {
        if (name == "max_source_bytes") l.max_source_bytes = 0;
        else if (name == "max_parse_depth") l.max_parse_depth = 0;
        else if (name == "parse_stack_bytes") l.parse_stack_bytes = 0;
        else if (name == "max_render_depth") l.max_render_depth = 0;
        else if (name == "max_steps") l.max_steps = 0;
        else if (name == "max_loop_iterations") l.max_loop_iterations = 0;
        else if (name == "max_output_bytes") l.max_output_bytes = 0;
        else if (name == "max_string_bytes") l.max_string_bytes = 0;
        else if (name == "max_alloc_bytes") l.max_alloc_bytes = 0;
        else std::fprintf(stderr, "unknown TemplateLimits field '%s'\n", name.c_str());
    }
    return l;
}

// Protects the host when a row is run with its bound disabled (red demonstration): an
// unbounded allocation then fails with bad_alloc instead of reaching the OOM killer. ASan
// reserves terabytes of shadow address space, so it is not applied there.
void limit_address_space() {
#if !defined(HALO_TEST_ASAN)
    const rlimit lim{std::uint64_t{8} << 30, std::uint64_t{8} << 30};
    setrlimit(RLIMIT_AS, &lim);
#endif
}

// Child side of a row. Exit codes: 0 expected typed error, 1 no error, 2 wrong ErrorCode,
// 3 non-halo exception, 4 message does not name the bound, 5 too slow.
[[noreturn]] void run_row(const std::function<void()>& body, ErrorCode expect, std::string_view needle) {
    alarm(kRowAlarmSeconds);
    limit_address_space();
    int rc = 1;
    std::thread worker([&] {
        const auto t0 = std::chrono::steady_clock::now();
        try {
            body();
        } catch (const halo::Error& e) {
            const std::string what = e.what();
            rc = e.code() != expect ? 2 : what.find(needle) == std::string::npos ? 4 : 0;
            std::fprintf(stderr, "row threw: %.300s\n", what.c_str());
        } catch (const std::exception& e) {
            rc = 3;
            std::fprintf(stderr, "row threw non-halo: %.300s\n", e.what());
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "row took %.3f s\n", s);
        if (rc == 0 && s > kRowSecondsBudget) rc = 5;
    });
    worker.join();
    std::exit(rc);
}

// Same, but the body must complete without throwing (exit 0).
[[noreturn]] void run_ok(const std::function<void()>& body) {
    alarm(kRowAlarmSeconds);
    limit_address_space();
    int rc = 0;
    std::thread worker([&] {
        try {
            body();
        } catch (const std::exception& e) {
            rc = 3;
            std::fprintf(stderr, "threw: %.300s\n", e.what());
        }
    });
    worker.join();
    std::exit(rc);
}

void render_empty(const ChatTemplate& t) { (void)t.render(OrderedJson::array(), nullptr, RenderOptions{}); }

std::string repeat(std::string_view s, std::size_t n) {
    std::string out;
    out.reserve(s.size() * n);
    for (std::size_t i = 0; i < n; ++i) out.append(s);
    return out;
}

ErrorCode error_code_of(const std::function<void()>& f) {
    try {
        f();
    } catch (const halo::Error& e) {
        return e.code();
    }
    return ErrorCode::Cancelled;  // "did not throw"
}

class TemplateRows : public ::testing::Test {
protected:
    void SetUp() override { GTEST_FLAG_SET(death_test_style, "threadsafe"); }
};

}  // namespace

// ---- S-3 evidence table: one test per row ------------------------------------------------

TEST_F(TemplateRows, MacroRecursionHitsRenderDepth) {
    const auto body = [] {
        const ChatTemplate t("{% macro f(n) %}{{ f(n + 1) }}{% endmacro %}{{ f(0) }}", {}, {}, rows());
        render_empty(t);
    };
    EXPECT_EXIT(run_row(body, ErrorCode::Api, "chat template limit exceeded: template render depth"),
                ::testing::ExitedWithCode(0), "");
}

TEST_F(TemplateRows, HundredKilobyteCommentHitsSourceCap) {
    const auto body = [] { const ChatTemplate t("{# " + std::string(100000, 'a') + " #}hello", {}, {}, rows()); };
    EXPECT_EXIT(run_row(body, ErrorCode::Config, "bytes (limit 65536)"), ::testing::ExitedWithCode(0), "");
}

TEST_F(TemplateRows, FiveThousandParensHitParseDepth) {
    const auto body = [] {
        const ChatTemplate t("{{ " + std::string(5000, '(') + "1" + std::string(5000, ')') + " }}", {}, {}, rows());
    };
    EXPECT_EXIT(run_row(body, ErrorCode::Config, "nesting too deep"), ::testing::ExitedWithCode(0), "");
}

TEST_F(TemplateRows, ThreeThousandParensHitParseDepthQuickly) {
    const auto body = [] {
        const ChatTemplate t("{{ " + std::string(3000, '(') + "1" + std::string(3000, ')') + " }}", {}, {}, rows());
    };
    EXPECT_EXIT(run_row(body, ErrorCode::Config, "nesting too deep"), ::testing::ExitedWithCode(0), "");
}

TEST_F(TemplateRows, NestedRangeLoopsHitIterationCap) {
    const auto body = [] {
        const ChatTemplate t("{% for i in range(99999) %}{% for j in range(99999) %}{% endfor %}{% endfor %}", {}, {},
                             rows());
        render_empty(t);
    };
    EXPECT_EXIT(run_row(body, ErrorCode::Api, "template loop iteration limit"), ::testing::ExitedWithCode(0), "");
}

TEST_F(TemplateRows, HundredGigabyteOutputHitsOutputCap) {
    const auto body = [] {
        const ChatTemplate t("{% set s = 'x' * 1000000 %}{% for i in range(99999) %}{{ s }}{% endfor %}", {}, {}, rows());
        render_empty(t);
    };
    EXPECT_EXIT(run_row(body, ErrorCode::Api, "template output size limit"), ::testing::ExitedWithCode(0), "");
}

TEST_F(TemplateRows, HundredMegabyteStringHitsValueCap) {
    const auto body = [] {
        const ChatTemplate t("{{ 'x' * 100000000 }}", {}, {}, rows());
        render_empty(t);
    };
    EXPECT_EXIT(run_row(body, ErrorCode::Api, "template value size limit"), ::testing::ExitedWithCode(0), "");
}

// ---- render bounds beyond the table ------------------------------------------------------

TEST_F(TemplateRows, ExponentialMacroHitsStepCap) {
    // Depth 40, no loops, tiny output: only the step budget stops 2^40 calls.
    const auto body = [] {
        const ChatTemplate t(
            "{% macro f(n) %}{% if n > 0 %}{{ f(n - 1) }}{{ f(n - 1) }}{% endif %}{% endmacro %}{{ f(40) }}", {}, {},
            rows());
        render_empty(t);
    };
    EXPECT_EXIT(run_row(body, ErrorCode::Api, "template step limit"), ::testing::ExitedWithCode(0), "");
}

TEST_F(TemplateRows, DeepErrorOnLongLineStaysLinear) {
    // An error raised at depth ~300 unwinds through every frame, each appending a source
    // excerpt (three lines; here one 60 KB line). Past 16 KiB the message is passed on as is.
    const auto body = [] {
        TemplateLimits l = rows();
        const ChatTemplate t("{% macro f(n) %}{% if n > 300 %}{{ raise_exception('deep') }}{% endif %}{{ f(n + 1) }}"
                             "{% endmacro %}{{ f(0) }}" +
                                 std::string(60000, 'p'),
                             {}, {}, l);
        render_empty(t);
    };
    EXPECT_EXIT(run_row(body, ErrorCode::Api, "chat template raised: deep"), ::testing::ExitedWithCode(0), "");
}

// ---- render hazards found by reading minja (not in the S-3 table) -------------------------

namespace {

struct HostileCase {
    const char* what;
    std::string source;
    ErrorCode code;
    const char* needle;
};

std::vector<HostileCase> hostile_cases() {
    const char* size = "template value size limit";
    return {
        {"string doubling", "{% set ns = namespace(s='x') %}{% for i in range(40) %}{% set ns.s = ns.s ~ ns.s %}{% endfor %}",
         ErrorCode::Api, size},
        {"array doubling", "{% set ns = namespace(a=[1]) %}{% for i in range(40) %}{% set ns.a = ns.a + ns.a %}{% endfor %}",
         ErrorCode::Api, size},
        {"replace blow-up", "{% set x = ('a' * 1000).replace('a', 'x' * 1000000) %}", ErrorCode::Api, size},
        {"join blow-up", "{% set x = range(100000)|join('x' * 1000000) %}", ErrorCode::Api, size},
        {"indent blow-up", "{% set x = ('a\\n' * 1000)|indent(100000000) %}", ErrorCode::Api, size},
        {"split explosion", "{% set x = ('.' * 30000000).split('.') %}", ErrorCode::Api, size},
        {"tojson indent", "{% set x = range(100000)|tojson(indent=100000) %}", ErrorCode::Api, size},
        {"append copies",
         "{% set s = 'x' * 30000000 %}{% set a = [] %}{% for i in range(1000) %}{% set _ = a.append(s) %}{% endfor %}",
         ErrorCode::Api, "template allocation limit"},
        {"cycle via append", "{% set a = [] %}{% set _ = a.append(a) %}{{ a }}", ErrorCode::Api, "inside itself"},
        {"cycle via namespace", "{% set ns = namespace() %}{% set ns.me = ns %}{{ ns }}", ErrorCode::Api, "inside itself"},
        // A plain {% set %} inside a loop assigns in the loop's scope, so `x` nests once per
        // iteration (100000 levels).
        {"deep value dump",
         "{% for i in range(100000) %}{% set x = [x] %}{% if loop.last %}{{ x }}{% endif %}{% endfor %}", ErrorCode::Api,
         "template render depth limit"},
        {"deep value compare",
         "{% for i in range(100000) %}{% set x = [x] %}{% set y = [y] %}{% if loop.last %}{{ x == y }}{% endif %}"
         "{% endfor %}",
         ErrorCode::Api, "template render depth limit"},
        {"deep value to json",
         "{% for i in range(100000) %}{% set x = [x] %}{% if loop.last %}{{ x|tojson }}{% endif %}{% endfor %}",
         ErrorCode::Api, "template render depth limit"},
        {"wide store scans", "{% set ns = namespace() %}{% set big = range(100000)|list %}"
         "{% for i in range(100000) %}{% set ns.a = big %}{% endfor %}",
         ErrorCode::Api, "template step limit"},
        {"quadratic search", "{% set h = 'a' * 8000000 %}{% set n = 'a' * 4000000 ~ 'b' %}{{ n in h }}", ErrorCode::Api,
         "template step limit"},
        {"empty separator", "{{ 'abc'.split('') }}", ErrorCode::Api, "empty separator"},
    };
}

// Must complete (quickly, without leaking under LSan): these used to crash, hang or leak.
std::vector<std::pair<const char*, std::string>> benign_cases() {
    return {
        {"deep value dropped", "{% for i in range(100000) %}{% set x = [x] %}{% endfor %}"},
        {"shared deep value dropped", "{% for i in range(100000) %}{% set x = [x, x] %}{% endfor %}"},
        {"deep dict dropped", "{% for i in range(100000) %}{% set x = {'k': x} %}{% endfor %}"},
        {"linear replace", "{% set x = ('ab' * 4000000).replace('a', '') %}{{ x|length }}"},
        {"empty repeat", "{{ '' * 9000000000000000000 }}"},
        {"macro defined in a loop", "{% for i in range(3) %}{% macro m() %}x{% endmacro %}{{ m() }}{% endfor %}"},
        {"call block in a loop",
         "{% macro f() %}[{{ caller() }}]{% endmacro %}{% for i in range(2) %}{% call f() %}y{% endcall %}{% endfor %}"},
        {"macro defined in a macro", "{% macro outer() %}{% macro inner() %}i{% endmacro %}{{ inner() }}{% endmacro %}{{ outer() }}"},
    };
}

}  // namespace

TEST_F(TemplateRows, HostileRenderPatternsRaiseTypedErrors) {
    for (const auto& c : hostile_cases()) {
        const std::string src = c.source;
        const auto body = [src] {
            const ChatTemplate t(src, {}, {}, rows());
            render_empty(t);
        };
        EXPECT_EXIT(run_row(body, c.code, c.needle), ::testing::ExitedWithCode(0), "") << c.what;
    }
}

TEST_F(TemplateRows, MillionLevelValueIsTornDownIteratively) {
    // 1M nesting levels (a loop over a 1 MB string; loop/step caps off for this case only).
    // The implicit ~Value recursed once per level: 100K levels fit in 8 MiB, 1M do not.
    for (const char* wrap : {"[x]", "{'k': x}"}) {
        const std::string src = std::string("{% for c in ('a' * 1000000) %}{% set x = ") + wrap + " %}{% endfor %}ok";
        const auto body = [src] {
            TemplateLimits l = rows();
            l.max_loop_iterations = 0;
            l.max_steps = 0;
            l.max_alloc_bytes = 0;
            const ChatTemplate t(src, {}, {}, l);
            if (t.apply(OrderedJson::array(), nullptr, {}) != "ok") throw std::runtime_error("wrong render");
        };
        EXPECT_EXIT(run_ok(body), ::testing::ExitedWithCode(0), "") << wrap;
    }
}

TEST_F(TemplateRows, FormerlyFatalPatternsNowComplete) {
    for (const auto& [what, source] : benign_cases()) {
        const std::string src = source;
        const auto body = [src] {
            const ChatTemplate t(src, {}, {}, rows());
            render_empty(t);
        };
        EXPECT_EXIT(run_ok(body), ::testing::ExitedWithCode(0), "") << what;
    }
}

TEST(TemplateLimitsUnit, ReplaceAndSplitKeepUpstreamSemantics) {
    const OrderedJson none = OrderedJson::array();
    const auto r = [&](const std::string& expr) { return ChatTemplate("{{ " + expr + " }}").apply(none, nullptr, {}); };
    EXPECT_EQ(r("'aXbXc'.replace('X', '--')"), "a--b--c");
    EXPECT_EQ(r("'aaaa'.replace('aa', 'a')"), "aa");
    EXPECT_EQ(r("'aXbXc'.replace('X', '', 1)"), "abXc");
    EXPECT_EQ(r("'abc'.replace('', '-')"), "---abc");  // upstream minja semantics, unchanged
    EXPECT_EQ(r("'abc'.replace('z', '-')"), "abc");
    EXPECT_EQ(r("'a,b,,c'.split(',')|join('|')"), "a|b||c");
    EXPECT_EQ(r("['x', 'y']|join(', ')"), "x, y");
    EXPECT_EQ(r("'a\nb'|indent(2)"), "a\n  b");
    EXPECT_EQ(r("[1, [2, 3]] == [1, [2, 3]]"), "True");
    EXPECT_EQ(r("{'a': [1]}|tojson(indent=2)"), "{\n  \"a\": [\n    1\n  ]\n}");
}

// ---- parse-side hardening beyond the table -----------------------------------------------

TEST_F(TemplateRows, ParseDepthCoversEveryRecursivePath) {
    const auto deep = [](std::string src) {
        return [src] { const ChatTemplate t(src, {}, {}, rows()); };
    };
    const std::size_t n = 2000;
    for (const std::string& src : {
             "{{ " + repeat("not ", n) + "x }}",                        // parseLogicalNot
             "{{ 1" + repeat("|string", n) + " }}",                    // parseMathMulDiv (filter chain)
             "{{ 'a'" + repeat(" ~ 'a'", n) + " }}",                   // parseStringConcat
             "{{ 1" + repeat(" if x else 1", n) + " }}",               // parseExpression (if-else)
             "{{ " + repeat("[", n) + repeat("]", n) + " }}",          // arrays
             "{{ f(" + repeat("(", n) + "1" + repeat(")", n) + ") }}",  // call args
             repeat("{% if x %}", n) + repeat("{% endif %}", n),       // parseTemplate (blocks)
         }) {
        EXPECT_EXIT(run_row(deep(src), ErrorCode::Config, "nesting too deep"), ::testing::ExitedWithCode(0), "")
            << src.substr(0, 40);
    }
}

TEST_F(TemplateRows, LongCommentLexesLinearlyOnSmallStack) {
    // Size cap and parse thread disabled: only minja's (patched) lexer is exercised, on an
    // 8 MiB worker stack. std::regex `[\s\S]*?` crashed here at 100 KB before the patch.
    const auto body = [] {
        TemplateLimits l = rows();
        l.max_source_bytes = 0;
        l.parse_stack_bytes = 0;
        const ChatTemplate t("{# " + std::string(1000000, 'a') + " #}hello", {}, {}, l);
        if (t.apply(OrderedJson::array(), nullptr, {}) != "hello") throw std::runtime_error("wrong render");
    };
    EXPECT_EXIT(run_ok(body), ::testing::ExitedWithCode(0), "");
}

TEST_F(TemplateRows, TokenScanIsNotQuadratic) {
    // 64 KiB of keyword-free tags: before the patch every failed keyword regex scanned to the
    // end of the source (regex_search without match_continuous).
    const auto body = [] {
        std::string src;
        while (src.size() + 16 < (std::size_t{64} << 10)) src += "{{ a.b[c] }}x";
        TemplateLimits l = rows();
        l.parse_stack_bytes = 0;
        const ChatTemplate t(src, {}, {}, l);
    };
    EXPECT_EXIT(run_ok(body), ::testing::ExitedWithCode(0), "");
}

TEST_F(TemplateRows, WhitespaceControlIsLinear) {
    // `{%-` / `-%}` stripping used regex_replace(\s+$ / ^\s+), recursive and quadratic on a
    // long whitespace run.
    const auto body = [] {
        TemplateLimits l = rows();
        l.parse_stack_bytes = 0;
        const std::string spaces(60000, ' ');
        const ChatTemplate a("[" + spaces + "{%- if true %}]{% endif %}", {}, {}, l);
        const std::string half(30000, ' ');
        const ChatTemplate b("{% if true -%}" + half + "[\n\t\v\f\r " + half + "{%- endif %}", {}, {}, l);
        if (a.apply(OrderedJson::array(), nullptr, {}) != "[]" || b.apply(OrderedJson::array(), nullptr, {}) != "[")
            throw std::runtime_error("wrong whitespace control");
    };
    EXPECT_EXIT(run_ok(body), ::testing::ExitedWithCode(0), "");
}

TEST_F(TemplateRows, LongTokensParseOnTheDedicatedStack) {
    // Longest possible single tokens inside the 64 KiB cap: an identifier (\w*) and a
    // `for` variable list, both matched by recursive std::regex. Parsed on the dedicated
    // parse thread, so the 8 MiB worker stack is not used.
    const auto body = [] {
        const std::string ident(60000, 'a');
        const ChatTemplate a("{{ " + ident + " }}", {}, {}, rows());
        std::string names = "a";
        while (names.size() < 60000) names += ", a";
        const ChatTemplate b("{% for " + names + " in x %}{% endfor %}", {}, {}, rows());
        const ChatTemplate c("{{ 'x' }}" + std::string(60000, '\n') + "{{- 'y' }}", {}, {}, rows());
        if (c.apply(OrderedJson::array(), nullptr, {}) != "xy") throw std::runtime_error("wrong render");
    };
    EXPECT_EXIT(run_ok(body), ::testing::ExitedWithCode(0), "");
}

TEST(TemplateLimitsUnit, CommentLexingMatchesUpstreamRegex) {
    const OrderedJson none = OrderedJson::array();
    EXPECT_EQ(ChatTemplate("a  {#- x -#}  b").apply(none, nullptr, {}), "ab");
    // `{#-#}`: the regex took '-' as the opening marker (greedy), leaving no closing marker.
    EXPECT_EQ(ChatTemplate("a   {#-#} b").apply(none, nullptr, {}), "a b");
    EXPECT_EQ(ChatTemplate("a  {#--#}  b").apply(none, nullptr, {}), "ab");
    EXPECT_EQ(ChatTemplate("a {##} b").apply(none, nullptr, {}), "a  b");
    EXPECT_EQ(ChatTemplate("{{ '{# not a comment #}' }}").apply(none, nullptr, {}), "{# not a comment #}");
    EXPECT_EQ(ChatTemplate("x #} y").apply(none, nullptr, {}), "x #} y");
    EXPECT_EQ(ChatTemplate("{# a #} #}").apply(none, nullptr, {}), " #}");
    EXPECT_EQ(error_code_of([] { ChatTemplate t("x {# never closed"); }), ErrorCode::Config);
}

// ---- S-8 ---------------------------------------------------------------------------------

TEST(TemplateLimitsUnit, ExtraContextCannotOverrideReservedVariables) {
    const ChatTemplate t("{% for m in messages %}{{ m.content }}{% endfor %}|{{ bos_token }}|{{ flag }}", "<s>", "</s>");
    const OrderedJson msgs = OrderedJson::parse(R"([{"role":"user","content":"hi"}])");
    for (const char* key : {"messages", "tools", "bos_token", "eos_token", "add_generation_prompt"}) {
        RenderOptions o;
        o.extra_context = {{key, OrderedJson::array({{{"role", "system"}, {"content", "INJECTED"}}})}};
        try {
            (void)t.render(msgs, nullptr, o);
            ADD_FAILURE() << key << " was accepted";
        } catch (const halo::Error& e) {
            EXPECT_EQ(e.code(), ErrorCode::Api) << key;
            EXPECT_NE(std::string(e.what()).find("reserved"), std::string::npos) << e.what();
        }
    }
    RenderOptions ok;
    ok.extra_context = {{"flag", "on"}};
    EXPECT_EQ(t.apply(msgs, nullptr, ok), "hi|<s>|on");
}

// ---- S-10 --------------------------------------------------------------------------------

TEST(TemplateLimitsUnit, RangeCapIsExactlyOneHundredThousand) {
    const OrderedJson none = OrderedJson::array();
    const auto length_of = [&](const std::string& args) {
        return ChatTemplate("{{ range(" + args + ")|length }}").apply(none, nullptr, {});
    };
    EXPECT_EQ(length_of("100000"), "100000");
    EXPECT_EQ(length_of("0, 100000"), "100000");
    EXPECT_EQ(length_of("1, 200001, 2"), "100000");
    EXPECT_EQ(length_of("100000, 0, -1"), "100000");
    EXPECT_EQ(length_of("-5"), "0");
    EXPECT_EQ(length_of("0, 5, 2"), "3");
    for (const char* args : {"100001", "0, 100001", "1, 200002, 2", "100001, 0, -1", "-9223372036854775807, 9223372036854775807"}) {
        EXPECT_EQ(error_code_of([&] { (void)length_of(args); }), ErrorCode::Api) << args;
    }
}

// ---- defaults, sized against the real templates -------------------------------------------

namespace {

std::string read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// A long agentic conversation: system + tools, then `turns` rounds of user -> assistant
// (reasoning + two tool calls) -> two tool results.
OrderedJson big_conversation(std::size_t turns, OrderedJson& tools) {
    tools = OrderedJson::array();
    for (int i = 0; i < 64; ++i) {
        tools.push_back({{"type", "function"},
                         {"function",
                          {{"name", "tool_" + std::to_string(i)},
                           {"description", std::string(200, 'd')},
                           {"parameters",
                            {{"type", "object"},
                             {"properties", {{"a", {{"type", "string"}}}, {"b", {{"type", "integer"}}}}}}}}}});
    }
    OrderedJson m = OrderedJson::array();
    m.push_back({{"role", "system"}, {"content", std::string(2000, 's')}});
    for (std::size_t t = 0; t < turns; ++t) {
        m.push_back({{"role", "user"}, {"content", "question " + std::to_string(t) + std::string(300, 'q')}});
        OrderedJson calls = OrderedJson::array();
        for (int c = 0; c < 2; ++c) {
            calls.push_back({{"type", "function"},
                             {"function", {{"name", "tool_" + std::to_string(c)}, {"arguments", {{"a", "x"}, {"b", c}}}}}});
        }
        m.push_back({{"role", "assistant"},
                     {"content", std::string(200, 'a')},
                     {"reasoning_content", std::string(500, 'r')},
                     {"tool_calls", calls}});
        m.push_back({{"role", "tool"}, {"content", std::string(400, 't')}});
        m.push_back({{"role", "tool"}, {"content", std::string(400, 'u')}});
    }
    m.push_back({{"role", "user"}, {"content", "final"}});
    return m;
}

void track(RenderStats& peak, const RenderStats& s) {
    peak.peak_depth = std::max(peak.peak_depth, s.peak_depth);
    peak.steps = std::max(peak.steps, s.steps);
    peak.loop_iterations = std::max(peak.loop_iterations, s.loop_iterations);
    peak.output_bytes = std::max(peak.output_bytes, s.output_bytes);
    peak.alloc_bytes = std::max(peak.alloc_bytes, s.alloc_bytes);
}

}  // namespace

TEST(TemplateLimitsUnit, DefaultsLeaveHeadroomOverRealTemplates) {
    const fs::path golden = fs::path(HALO_REF_DIR) / "template_golden";
    if (!fs::exists(golden / "ggml.jinja") || !fs::exists(golden / "unsloth.jinja")) {
        GTEST_SKIP() << "template golden missing (" << golden << "): run python/tools/make_template_golden.py";
    }
    const TemplateLimits d;
    for (const char* file : {"ggml.jinja", "unsloth.jinja"}) {
        const std::string source = read_file(golden / file);
        EXPECT_LE(source.size() * 4, d.max_source_bytes) << file;
        // Smallest parser depth limit the template parses under.
        std::size_t need = 1;
        for (;; ++need) {
            TemplateLimits l;
            l.max_parse_depth = need;
            if (error_code_of([&] { ChatTemplate probe(source, {}, {}, l); }) == ErrorCode::Cancelled) break;
            ASSERT_LT(need, d.max_parse_depth) << file;
        }
        std::printf("%s: %zu bytes, parses at max_parse_depth >= %zu\n", file, source.size(), need);
        EXPECT_LE(need * 4, d.max_parse_depth) << file;
        const ChatTemplate t(source);
        OrderedJson tools;
        // ~4000 messages, ~2.6 MB of prompt: beyond any 262K-token context.
        const OrderedJson msgs = big_conversation(1000, tools);
        RenderOptions o;
        o.extra_context = {{"enable_thinking", true}, {"preserve_thinking", true}};
        const auto t0 = std::chrono::steady_clock::now();
        const auto r = t.render(msgs, tools, o);
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        RenderStats peak;
        track(peak, r.stats);
        std::printf("%s: %zu messages, prompt %zu B, %.3f s | depth %zu steps %llu iterations %llu output %llu B alloc %llu B\n",
                    file, msgs.size(), r.prompt.size(), secs, peak.peak_depth,
                    static_cast<unsigned long long>(peak.steps), static_cast<unsigned long long>(peak.loop_iterations),
                    static_cast<unsigned long long>(peak.output_bytes), static_cast<unsigned long long>(peak.alloc_bytes));
        // The output bound is an upper bound of the prompt, and must dominate it.
        EXPECT_GE(r.stats.output_bytes, r.prompt.size());
        // Headroom: every default is at least 4x what this oversized conversation needs.
        EXPECT_LE(peak.peak_depth * 4, d.max_render_depth) << file;
        EXPECT_LE(peak.steps * 4, d.max_steps) << file;
        EXPECT_LE(peak.loop_iterations * 4, d.max_loop_iterations) << file;
        EXPECT_LE(peak.output_bytes * 4, d.max_output_bytes) << file;
        EXPECT_LE(peak.alloc_bytes * 4, d.max_alloc_bytes) << file;
    }
}

// ---- stack use at the render-depth cap ----------------------------------------------------

namespace {

// Runs fn on a thread whose stack we own and pre-paint, and returns the peak number of
// stack bytes it touched ("stack painting").
std::size_t peak_stack_bytes(std::size_t stack_bytes, const std::function<void()>& fn) {
    std::vector<unsigned char> stack(stack_bytes, 0xA5);
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstack(&attr, stack.data(), stack.size());
    struct Job {
        const std::function<void()>* fn;
    } job{&fn};
    pthread_t th{};
    const int rc = pthread_create(
        &th, &attr,
        [](void* p) -> void* {
            try {
                (*static_cast<Job*>(p)->fn)();
            } catch (...) {
            }
            return nullptr;
        },
        &job);
    pthread_attr_destroy(&attr);
    if (rc != 0) return SIZE_MAX;
    pthread_join(th, nullptr);
    std::size_t untouched = 0;  // the stack grows down from the end of the buffer
    while (untouched < stack.size() && stack[untouched] == 0xA5) ++untouched;
    return stack.size() - untouched;
}

}  // namespace

TEST(TemplateLimitsUnit, RenderAtDepthCapFitsTheDocumentedStack) {
    const ChatTemplate t("{% macro f(n) %}{{ f(n + 1) }}{% endmacro %}{{ f(0) }}");
    ErrorCode code = ErrorCode::Cancelled;
    const std::size_t used = peak_stack_bytes(std::size_t{64} << 20, [&] {
        code = error_code_of([&] { render_empty(t); });
    });
    std::printf("render to depth %zu used %zu KiB of stack\n", t.limits().max_render_depth, used >> 10);
    EXPECT_EQ(code, ErrorCode::Api);
    EXPECT_LE(used, halo::chat::kMaxRenderStackBytes);
}
