// halo subcommands: inspect, devices, tokenize, template, run, serve (bench: bench.cpp).

#include "commands.h"

#include <chrono>
#include <format>
#include <limits>
#include <fstream>
#include <iostream>
#include <sstream>

#include <nlohmann/json.hpp>

#include "args.h"
#include "config.h"
#include "halo/core/error.h"
#include "halo/hardware/checklist.h"
#include "halo/memory/planner.h"
#include "halo/model/inspect.h"
#include "halo/model/model.h"
#include "halo/template/output_parser.h"
#include "model_io.h"

#if HALO_CLI_HAVE_API
#include "halo/api/prompt.h"
#include "halo/api/server.h"
#endif

namespace halo::cli {

using nlohmann::json;
using nlohmann::ordered_json;

std::string human_bytes(std::uint64_t b) {
    static constexpr const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double v = static_cast<double>(b);
    std::size_t u = 0;
    while (v >= 1024.0 && u + 1 < std::size(units)) {
        v /= 1024.0;
        ++u;
    }
    return u == 0 ? std::format("{} B", b) : std::format("{:.2f} {}", v, units[u]);
}

std::filesystem::path default_profile_db(const std::map<std::string, std::string>& env) {
    const auto get = [&](const char* k) -> std::string {
        const auto it = env.find(k);
        return it == env.end() ? std::string{} : it->second;
    };
    if (auto p = get("HALO_PROFILE_DB"); !p.empty()) return p;
    if (auto x = get("XDG_CACHE_HOME"); !x.empty()) return std::filesystem::path(x) / "halo/profiles.db";
    if (auto h = get("HOME"); !h.empty()) return std::filesystem::path(h) / ".cache/halo/profiles.db";
    return {};
}

hardware::DiscoveryOptions discovery_options(const std::string& root, const std::map<std::string, std::string>& env) {
    hardware::DiscoveryOptions o;
    o.root = root;
    if (root == "/") {
        o.env = env;
        o.use_process_env = false;  // `env` is the process environment already (Context)
    }
    return o;
}

namespace {

std::string read_input(const std::string& path, Context& ctx, const std::string& what) {
    std::ostringstream ss;
    if (path == "-") {
        HALO_CHECK(ctx.in != nullptr, ErrorCode::Io, "{}: no stdin available", what);
        ss << ctx.in->rdbuf();
        return ss.str();
    }
    std::ifstream f(path, std::ios::binary);
    HALO_CHECK(f.good(), ErrorCode::Io, "cannot read {} '{}'", what, path);
    ss << f.rdbuf();
    return ss.str();
}

std::string model_arg(const ParsedArgs& a, const std::string& usage) {
    if (a.positionals.size() != 1) throw UsageError("expected exactly one model file\n" + usage);
    return a.positionals[0];
}

// ---- inspect -------------------------------------------------------------------------------

memory::ModelShape shape_from(const model::InspectReport& r, std::vector<std::string>& notes) {
    const auto cls = [&](const char* k) -> std::uint64_t {
        const auto it = r.classes.find(k);
        return it == r.classes.end() ? 0 : it->second.bytes;
    };
    memory::ModelShape s;
    s.weights.embeddings = cls("embeddings");
    s.weights.lm_head = cls("lm_head");
    s.weights.attention = cls("attention");
    s.weights.gdn = cls("gdn");
    s.weights.ffn = cls("ffn");
    s.weights.norms = cls("norms");
    s.weights.mtp = cls("mtp") + r.mtp_file_weight_bytes;
    if (const auto other = cls("other"); other > 0) {
        s.weights.ffn += other;
        notes.push_back(std::format("{} of unclassified tensors counted with the FFN weights", human_bytes(other)));
    }
    s.n_attn_layers = r.n_attn_layers;
    s.n_head = r.n_head;
    s.n_head_kv = r.n_head_kv;
    s.key_dim = r.key_length;
    s.value_dim = r.value_length;
    s.n_gdn_layers = r.n_gdn_layers;
    s.n_v_heads = r.gdn_n_v_heads;
    s.d_k = r.gdn_head_k_dim;
    s.d_v = r.gdn_head_v_dim;
    s.conv_kernel = r.gdn_conv_kernel;
    s.conv_channels = r.gdn_conv_channels;
    s.hidden = r.n_embd;
    s.ffn_intermediate = r.n_ff;
    s.vocab = static_cast<std::uint32_t>(r.n_vocab);
    s.max_trained_context = r.context_length;
    s.mtp_attn_layers = r.mtp_present ? 1 : 0;
    return s;
}

const std::vector<OptionSpec>& inspect_opts() {
    static const std::vector<OptionSpec> o = {
        {"mtp", true, 0, false, false, "separate MTP GGUF to attach"},
        {"json", false, 0, false, false, "machine-readable output"},
        {"ctx", true, 0, false, false, "context for the memory estimate (default min(32768, trained))"},
        {"parallel", true, 0, false, false, "sequences for the memory estimate (default 1)"},
        {"mtp-draft", true, 0, false, false, "MTP draft depth for the estimate (default 2; 0 = MTP off)"},
        {"root", true, 0, false, true, "filesystem root for hardware discovery (tests)"},
    };
    return o;
}

}  // namespace

int cmd_inspect(const std::vector<std::string>& args, Context& ctx) {
    const std::string usage = "usage: halo inspect <model.gguf> [options]\n" + options_help(inspect_opts());
    const ParsedArgs a = parse_args(args, inspect_opts(), usage);
    const std::string path = model_arg(a, usage);

    auto m = model::NormalizedModel::load(path, model::GgufMode::HeaderOnly);
    if (auto mtp = a.value("mtp")) m.attach_mtp(*mtp, model::GgufMode::HeaderOnly);
    const model::InspectReport r = m.inspect();

    // Memory estimate via the planner, against the discovered tiers.
    json mem = json::object();
    std::vector<std::string> mem_notes;
    std::optional<memory::MemoryPlan> plan;
    std::uint64_t max_ctx = 0;
    memory::PlanRequest req;
    try {
        const memory::ModelShape shape = shape_from(r, mem_notes);
        const std::uint64_t trained = r.context_length == 0 ? 32768 : r.context_length;
        req.max_context = a.u64("ctx", 1, 1u << 22).value_or(std::min<std::uint64_t>(32768, trained));
        req.max_sequences = static_cast<std::uint32_t>(a.u64("parallel", 1, 256).value_or(1));
        const auto draft = a.u64("mtp-draft", 0, 16).value_or(2);
        req.mtp_enabled = r.mtp_present && draft > 0;
        req.mtp_draft_depth = req.mtp_enabled ? static_cast<std::uint32_t>(draft) : 0;
        const auto hw = hardware::discover(discovery_options(a.value("root").value_or("/"), ctx.env));
        plan = memory::plan_memory(shape, req, hw.tiers);
        max_ctx = memory::max_context_for_budget(shape, req, hw.tiers);
        mem["request"] = {{"max_context", req.max_context},
                          {"max_sequences", req.max_sequences},
                          {"mtp_enabled", req.mtp_enabled},
                          {"mtp_draft_depth", req.mtp_draft_depth}};
        mem["topology"] = std::string(hardware::to_string(hw.tiers.topology));
        mem["plan"] = *plan;
        mem["max_context_for_budget"] = max_ctx;
        mem["notes"] = mem_notes;
    } catch (const UsageError&) {
        throw;
    } catch (const std::exception& e) {
        mem = {{"error", e.what()}};
    }

    std::ostream& out = *ctx.out;
    if (a.flag("json")) {
        json j = {{"report", r.to_json()}, {"memory", mem}};
        out << j.dump(2) << "\n";
        return kExitOk;
    }
    out << std::format("model:        {} ({}, GGUF v{}, {})\n", r.name.empty() ? "(unnamed)" : r.name, r.architecture,
                       r.gguf_version, r.mode);
    out << std::format("file:         {} ({})\n", r.source, human_bytes(r.file_size));
    out << std::format("layers:       {} = {} Gated DeltaNet + {} full attention (every {}th)\n", r.n_layer,
                       r.n_gdn_layers, r.n_attn_layers, r.full_attention_interval);
    out << std::format("hparams:      n_embd {}, n_ff {}, heads {}/{} (q/kv), head dims {}/{}, vocab {}, trained "
                       "context {}\n",
                       r.n_embd, r.n_ff, r.n_head, r.n_head_kv, r.key_length, r.value_length, r.n_vocab,
                       r.context_length);
    out << std::format("GDN:          {} k-heads, {} v-heads, d_k {}, d_v {}, conv kernel {}\n", r.gdn_n_k_heads,
                       r.gdn_n_v_heads, r.gdn_head_k_dim, r.gdn_head_v_dim, r.gdn_conv_kernel);
    out << std::format("weights:      {} in {} tensors (lm_head {}{})\n", human_bytes(r.total_weight_bytes), r.n_tensors,
                       r.lm_head_dtype, r.lm_head_tied ? ", tied to token_embd" : "");
    for (const auto& [k, v] : r.classes) out << std::format("  {:<12}{:>12}  ({} tensors)\n", k, human_bytes(v.bytes), v.count);
    std::string dt;
    for (const auto& [k, v] : r.dtypes) dt += std::format(" {}={}", k, v.count);
    out << "dtypes:      " << dt << "\n";
    out << std::format("state:        KV {} per token; GDN state {} per sequence\n", human_bytes(r.kv_bytes_per_token),
                       human_bytes(r.gdn_state_bytes_per_sequence));
    out << std::format("MTP:          {}{}\n", r.mtp_source,
                       r.mtp_present ? std::format(" ({})", human_bytes(r.mtp_block_bytes + r.mtp_file_weight_bytes)) : "");
    out << std::format("tokenizer:    {}/{}, {} tokens, {} merges, eos {}, chat template {} chars\n", r.tokenizer_model,
                       r.tokenizer_pre, r.n_tokens, r.n_merges, r.eos_id, r.chat_template_chars);
    if (mem.contains("error")) {
        out << "memory:       estimate unavailable: " << mem["error"].get<std::string>() << "\n";
    } else {
        out << std::format("memory:       ctx {} x {} sequence(s){} on {} -> {}\n", req.max_context, req.max_sequences,
                           req.mtp_enabled ? std::format(", MTP draft {}", req.mtp_draft_depth) : "",
                           mem["topology"].get<std::string>(), plan->message);
        out << std::format("              planned {}; largest context that fits: {}\n", human_bytes(plan->planned_total),
                           max_ctx);
        out << plan->table();
        for (const auto& n : mem_notes) out << "  note: " << n << "\n";
    }
    for (const auto& w : r.warnings) out << "warning: " << w << "\n";
    return kExitOk;
}

// ---- devices -------------------------------------------------------------------------------

int cmd_devices(const std::vector<std::string>& args, Context& ctx) {
    static const std::vector<OptionSpec> opts = {
        {"verify", false, 0, false, false, "run the deployment checklist (exit 1 on a failed check)"},
        {"json", false, 0, false, false, "machine-readable output"},
        {"root", true, 0, false, true, "filesystem root to inspect (tests / captured fixtures)"},
    };
    const std::string usage = "usage: halo devices [--verify] [--json]\n" + options_help(opts);
    const ParsedArgs a = parse_args(args, opts, usage);
    if (!a.positionals.empty()) throw UsageError("devices takes no positional arguments\n" + usage);
    const auto info = hardware::discover(discovery_options(a.value("root").value_or("/"), ctx.env));
    std::vector<hardware::CheckItem> checks;
    if (a.flag("verify")) checks = hardware::deployment_checklist(info);
    const auto overall = hardware::overall_status(checks);
    const int rc = a.flag("verify") && overall == hardware::CheckStatus::Fail ? kExitFailure : kExitOk;

    std::ostream& out = *ctx.out;
    if (a.flag("json")) {
        json j = {{"hardware", info}};
        if (a.flag("verify")) {
            j["checklist"] = checks;
            j["overall"] = std::string(hardware::to_string(overall));
        }
        out << j.dump(2) << "\n";
        return rc;
    }
    const auto opt = [](const std::optional<std::uint64_t>& v) { return v ? human_bytes(*v) : std::string("unknown"); };
    out << std::format("cpu:      {} ({} cores / {} threads)\n", info.cpu.model_name.empty() ? "unknown" : info.cpu.model_name,
                       info.cpu.physical_cores, info.cpu.logical_threads);
    out << std::format("memory:   {} total, {} available\n", opt(info.host_memory.total_bytes),
                       opt(info.host_memory.available_bytes));
    if (info.gpus.empty()) out << "gpu:      no AMD GPU found\n";
    for (const auto& g : info.gpus) {
        out << std::format("gpu:      {} {:04x}:{:04x} {} ({} CUs), VRAM {}, GTT {}\n", g.drm_card, g.vendor_id,
                           g.device_id, g.gfx_target.value_or("gfx?"), g.compute_units.value_or(0), opt(g.vram_total),
                           opt(g.gtt_total));
    }
    out << std::format("topology: {} (GPU-accessible {})\n", hardware::to_string(info.tiers.topology),
                       human_bytes(info.tiers.gpu_accessible_bytes));
    out << std::format("os:       {} (kernel {})\n", info.os.distro_pretty_name.value_or("unknown"),
                       info.os.kernel_release.value_or("unknown"));
    out << std::format("rocm:     {}\n", info.rocm.installed ? info.rocm.version.value_or("installed") : "not installed");
    out << std::format("vulkan:   {} driver manifest(s)\n", info.vulkan.icds.size());
    out << std::format("iommu:    {}\n", hardware::to_string(info.iommu.state));
    for (const auto& w : info.warnings) out << "warning:  " << w << "\n";
    if (a.flag("verify")) {
        out << "checklist:\n";
        for (const auto& c : checks) {
            out << std::format("  [{:<7}] {:<16} {}\n", hardware::to_string(c.status), c.check, c.detail);
        }
        out << "overall: " << hardware::to_string(overall) << "\n";
    }
    return rc;
}

// ---- tokenize / template -------------------------------------------------------------------

int cmd_tokenize(const std::vector<std::string>& args, Context& ctx) {
    static const std::vector<OptionSpec> opts = {
        {"text", true, 't', false, false, "text to tokenize (default: read --file or stdin)"},
        {"file", true, 'f', false, false, "read the text from a file ('-' = stdin)"},
        {"parse-special", false, 0, false, false, "turn special-token text into control tokens"},
        {"pieces", false, 0, false, false, "print id and piece per token"},
        {"json", false, 0, false, false, "machine-readable output"},
    };
    const std::string usage = "usage: halo tokenize <model.gguf> [--text T | --file F]\n" + options_help(opts);
    const ParsedArgs a = parse_args(args, opts, usage);
    const std::string path = model_arg(a, usage);
    if (a.has("text") && a.has("file")) throw UsageError("give --text or --file, not both");
    const std::string text = a.value("text") ? *a.value("text") : read_input(a.value("file").value_or("-"), ctx, "text");
    const ModelText mt = load_model_text(path);
    const auto ids = mt.tokenizer->encode(text, a.flag("parse-special"));
    std::ostream& out = *ctx.out;
    if (a.flag("json")) {
        json arr = json::array();
        for (const auto id : ids) {
            if (a.flag("pieces")) arr.push_back({{"id", id}, {"piece", mt.tokenizer->token_to_piece(id)}});
            else arr.push_back(id);
        }
        out << json{{"tokens", arr}, {"count", ids.size()}}.dump() << "\n";
    } else if (a.flag("pieces")) {
        for (const auto id : ids) out << id << "\t" << json(mt.tokenizer->token_to_piece(id)).dump() << "\n";
    } else {
        for (std::size_t i = 0; i < ids.size(); ++i) out << (i ? " " : "") << ids[i];
        out << "\n";
    }
    return kExitOk;
}

namespace {

struct ChatInput {
    ordered_json messages;
    ordered_json tools;
};

ChatInput parse_chat_input(const std::string& text, const std::string& what) {
    ordered_json j = ordered_json::parse(text, nullptr, /*allow_exceptions=*/false, /*ignore_comments=*/false);
    HALO_CHECK(!j.is_discarded(), ErrorCode::Api, "{} is not valid JSON", what);
    ChatInput c{ordered_json::array(), nullptr};
    if (j.is_array()) {
        c.messages = std::move(j);
    } else {
        HALO_CHECK(j.is_object() && j.contains("messages") && j["messages"].is_array(), ErrorCode::Api,
                   "{} must be a messages array or an object with \"messages\"", what);
        c.messages = j["messages"];
        if (j.contains("tools")) c.tools = j["tools"];
    }
    return c;
}

struct EncodedChat {
    std::string text;
    std::vector<std::int32_t> tokens;
    bool starts_in_reasoning = false;
};

/// The same prompt construction as the API's chat routes (special-token protection, see
/// include/halo/api/prompt.h) when halo_api is part of the build.
EncodedChat encode_chat(const tokenizer::Tokenizer& tok, const chat::ChatTemplate& tmpl, const ordered_json& messages,
                        const ordered_json& tools, const chat::RenderOptions& opt) {
#if HALO_CLI_HAVE_API
    const api::SpecialTokens sp(tok);
    auto p = api::build_chat_prompt(tok, sp, tmpl, messages, tools, opt);
    return {std::move(p.text), std::move(p.tokens), p.starts_in_reasoning};
#else
    EncodedChat e;
    e.text = tmpl.apply(messages, tools, opt);
    e.tokens = tok.encode(e.text, true);
    e.starts_in_reasoning = chat::prompt_ends_in_reasoning(e.text);
    return e;
#endif
}

void apply_thinking(chat::RenderOptions& o, bool no_think, const std::optional<std::string>& effort) {
    if (effort && *effort == "none") no_think = true;
    if (no_think) {
        o.extra_context["enable_thinking"] = false;
    } else if (effort) {
        if (*effort != "low" && *effort != "medium" && *effort != "xhigh") {
            throw UsageError("--reasoning-effort must be none, low, medium or xhigh (got '" + *effort + "')");
        }
        o.extra_context["reasoning_effort"] = *effort;
    }
}

}  // namespace

int cmd_template(const std::vector<std::string>& args, Context& ctx) {
    static const std::vector<OptionSpec> opts = {
        {"messages", true, 'm', false, false, "JSON file: messages array or {messages, tools} ('-' = stdin)"},
        {"tools", true, 0, false, false, "JSON file with an OpenAI tools array"},
        {"no-generation-prompt", false, 0, false, false, "do not append the assistant header"},
        {"no-think", false, 0, false, false, "render with enable_thinking=false"},
        {"reasoning-effort", true, 0, false, false, "none | low | medium | xhigh"},
        {"tokenize", false, 0, false, false, "also print the prompt tokens"},
        {"json", false, 0, false, false, "machine-readable output"},
    };
    const std::string usage = "usage: halo template <model.gguf> --messages F\n" + options_help(opts);
    const ParsedArgs a = parse_args(args, opts, usage);
    const std::string path = model_arg(a, usage);
    if (!a.has("messages")) throw UsageError("--messages is required\n" + usage);
    ChatInput in = parse_chat_input(read_input(*a.value("messages"), ctx, "messages"), "--messages");
    if (auto t = a.value("tools")) in.tools = parse_chat_input("{\"messages\":[],\"tools\":" + read_input(*t, ctx, "tools") + "}", "--tools").tools;
    const ModelText mt = load_model_text(path);
    HALO_CHECK(mt.chat_template != nullptr, ErrorCode::Model, "{} has no tokenizer.chat_template", path);
    chat::RenderOptions opt;
    opt.add_generation_prompt = !a.flag("no-generation-prompt");
    opt.parse_string_tool_arguments = true;
    apply_thinking(opt, a.flag("no-think"), a.value("reasoning-effort"));
    const EncodedChat e = encode_chat(*mt.tokenizer, *mt.chat_template, in.messages, in.tools, opt);
    std::ostream& out = *ctx.out;
    if (a.flag("json")) {
        json j = {{"prompt", e.text}};
        if (a.flag("tokenize")) j["tokens"] = e.tokens;
        out << j.dump() << "\n";
        return kExitOk;
    }
    out << e.text;
    if (a.flag("tokenize")) {
        out << "\n--- " << e.tokens.size() << " tokens ---\n";
        for (std::size_t i = 0; i < e.tokens.size(); ++i) out << (i ? " " : "") << e.tokens[i];
        out << "\n";
    }
    return kExitOk;
}

// ---- run -----------------------------------------------------------------------------------

int cmd_run(const std::vector<std::string>& args, Context& ctx) {
    std::vector<OptionSpec> opts = config_options(runtime_keys());
    for (OptionSpec o : std::vector<OptionSpec>{
             {"prompt", true, 'p', false, false, "user message (or the raw prompt with --raw)"},
             {"system", true, 's', false, false, "system message"},
             {"max-tokens", true, 'n', false, false, "tokens to generate (default 512)"},
             {"temperature", true, 0, false, false, "sampling temperature (default 0.7; 0 = greedy)"},
             {"top-p", true, 0, false, false, "nucleus sampling"},
             {"top-k", true, 0, false, false, "top-k sampling"},
             {"seed", true, 0, false, false, "sampling seed"},
             {"no-think", false, 0, false, false, "disable thinking"},
             {"raw", false, 0, false, false, "no chat template: tokenize --prompt as is (special tokens parsed)"},
             {"show-reasoning", false, 0, false, false, "print reasoning to stderr"},
         }) {
        opts.push_back(std::move(o));
    }
    const std::string usage = "usage: halo run <model.gguf> -p PROMPT [options]\n" + options_help(opts);
    ParsedArgs a = parse_args(args, opts, usage);
    if (a.positionals.size() > 1) throw UsageError("expected at most one model file\n" + usage);
    if (a.positionals.size() == 1) {
        if (a.has("model")) throw UsageError("give the model as an argument or --model, not both");
        a.values_["model"] = {a.positionals[0]};
    }
    if (!a.has("prompt")) throw UsageError("--prompt/-p is required\n" + usage);
    const ResolvedConfig cfg = resolve_config(a, ctx.env, runtime_keys());
    const runtime::EngineConfig ec = engine_config(cfg, default_profile_db(ctx.env));
    SamplingParams sp;
    sp.temperature = static_cast<float>(a.number("temperature", 0.0, 2.0).value_or(0.7));
    if (auto v = a.number("top-p", 0.0, 1.0)) sp.top_p = static_cast<float>(*v);
    if (auto v = a.u64("top-k", 0, 1u << 20)) sp.top_k = static_cast<int>(*v);
    if (auto v = a.value("seed")) sp.seed = parse_u64(*v, "--seed", 0, std::numeric_limits<std::uint64_t>::max());
    const std::size_t max_tokens = a.u64("max-tokens", 1, 1u << 20).value_or(512);

    if (!ctx.engine_factory) {
        *ctx.err << "halo: " << kRuntimeNotBuilt << "\n";
        return kExitUsage;
    }
    auto engine = ctx.engine_factory(ec);
    HALO_CHECK(engine != nullptr, ErrorCode::Config, "the engine factory returned no engine");
    const auto& tok = engine->tokenizer();

    runtime::GenerateRequest req;
    req.sampling = sp;
    req.max_tokens = max_tokens;
    bool starts_in_reasoning = false;
    if (a.flag("raw")) {
        req.prompt = tok.encode(*a.value("prompt"), true);
    } else {
        ordered_json msgs = ordered_json::array();
        if (auto s = a.value("system")) msgs.push_back({{"role", "system"}, {"content", *s}});
        msgs.push_back({{"role", "user"}, {"content", *a.value("prompt")}});
        chat::RenderOptions opt;
        apply_thinking(opt, a.flag("no-think"), std::nullopt);
        EncodedChat e = encode_chat(tok, engine->chat_template(), msgs, nullptr, opt);
        req.prompt = std::move(e.tokens);
        starts_in_reasoning = e.starts_in_reasoning;
    }

    chat::ParserOptions po;
    po.starts_in_reasoning = starts_in_reasoning;
    chat::OutputParser parser(po);
    std::ostream& out = *ctx.out;
    std::ostream& err = *ctx.err;
    const bool show_reasoning = a.flag("show-reasoning");
    const bool raw = a.flag("raw");
    const auto deliver = [&](const std::vector<chat::OutputEvent>& evs) {
        for (const auto& e : evs) {
            switch (e.kind) {
                case chat::OutputEvent::Kind::ReasoningDelta:
                    if (show_reasoning) err << e.text << std::flush;
                    break;
                case chat::OutputEvent::Kind::ContentDelta: out << e.text << std::flush; break;
                case chat::OutputEvent::Kind::ToolCall: {
                    const auto& c = parser.message().tool_calls.at(e.tool_index);
                    out << "\n[tool call] " << c.name << " " << c.arguments.dump() << "\n" << std::flush;
                    break;
                }
            }
        }
    };
    const auto t0 = std::chrono::steady_clock::now();
    std::size_t generated = 0;
    const auto result = engine->generate(req, [&](const runtime::TokenEvent& ev) {
        ++generated;
        if (ev.is_eos) return true;
        if (raw) {
            out << ev.piece << std::flush;
        } else {
            deliver(parser.feed(ev.piece));
        }
        return out.good();
    });
    if (!raw) deliver(parser.finish());
    out << "\n" << std::flush;
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const char* finish = result.finish == runtime::FinishReason::Stop     ? "stop"
                         : result.finish == runtime::FinishReason::Length ? "length"
                         : result.finish == runtime::FinishReason::Cancelled ? "cancelled"
                                                                             : "error";
    err << std::format("[halo] prompt {} tokens, generated {} tokens in {:.2f} s ({:.2f} tok/s decode), finish {}\n",
                       req.prompt.size(), generated, secs, result.decode_tps, finish);
    if (result.finish == runtime::FinishReason::Error) {
        err << "halo: generation failed: " << result.error << "\n";
        return kExitFailure;
    }
    return kExitOk;
}

// ---- serve ---------------------------------------------------------------------------------

int cmd_serve(const std::vector<std::string>& args, Context& ctx) {
    std::vector<std::string> keys = runtime_keys();
    keys.insert(keys.end(), server_keys().begin(), server_keys().end());
    std::vector<OptionSpec> opts = config_options(keys);
    opts.push_back({"print-config", false, 0, false, false, "print the resolved configuration (API key redacted) and exit"});
    const std::string usage = "usage: halo serve --model <model.gguf> [options]\n" + options_help(opts);
    ParsedArgs a = parse_args(args, opts, usage);
    if (a.positionals.size() > 1) throw UsageError("expected at most one model file\n" + usage);
    if (a.positionals.size() == 1) {
        if (a.has("model")) throw UsageError("give the model as an argument or --model, not both");
        a.values_["model"] = {a.positionals[0]};
    }
    const ResolvedConfig cfg = resolve_config(a, ctx.env, keys);
    if (a.flag("print-config")) {
        *ctx.out << cfg.to_json().dump(2) << "\n";
        return kExitOk;
    }
#if HALO_CLI_HAVE_API
    const runtime::EngineConfig ec = engine_config(cfg, default_profile_db(ctx.env));
    api::ServerConfig sc = server_config(cfg);
    // More API slots than engine sequences would park the extra requests inside the engine,
    // where queue_timeout, request_timeout, disconnect and stop() cannot reach them until
    // they produce a token (review R-3). Keep the queueing in the API's admission control.
    if (sc.max_concurrent > ec.max_sequences) {
        *ctx.err << std::format("halo serve: --max-concurrent {} exceeds --parallel {}; using {} (extra requests queue "
                                "in the API, where timeouts and cancellation apply)\n",
                                sc.max_concurrent, ec.max_sequences, ec.max_sequences);
        sc.max_concurrent = ec.max_sequences;
    }
    // Fail before the (slow) model load on an unsafe bind (PRD §12, review A-6).
    // (also S-19 wildcard CORS without a key, S-20 unauthenticated remote without allowed_hosts)
    api::validate_server_config(sc);
    if (!ctx.engine_factory) {
        *ctx.err << "halo: " << kRuntimeNotBuilt << "\n";
        return kExitUsage;
    }
    auto engine = ctx.engine_factory(ec);
    HALO_CHECK(engine != nullptr, ErrorCode::Config, "the engine factory returned no engine");
    api::ApiServer server(*engine, sc);
    const int port = server.bind();
    *ctx.out << std::format("halo: serving {} on http://{}:{}{}\n", engine->model().id, sc.host, port,
                            sc.api_key ? " (API key required)" : "")
             << std::flush;
    if (ctx.on_serving) {
        server.start();
        ctx.on_serving(server, [&server] { server.stop(); });
        server.stop();
    } else {
        server.listen();
    }
    *ctx.out << "halo: server stopped\n";
    return kExitOk;
#else
    (void)cfg;
    *ctx.err << "halo: the API server is not part of this build (HALO_BUILD_SERVER=OFF)\n";
    return kExitUsage;
#endif
}

}  // namespace halo::cli
