#include "tokenizer/unicode.h"

#include <algorithm>
#include <array>
#include <span>

namespace halo::tokenizer::unicode {
namespace {

struct CpRange {
    std::uint32_t first;
    std::uint32_t last;
};
struct CccRange {
    std::uint32_t first;
    std::uint32_t last;
    std::uint8_t ccc;
};
struct DecompEntry {
    std::uint32_t cp;
    std::uint16_t offset;
    std::uint8_t len;
};
struct CompEntry {
    std::uint32_t first;
    std::uint32_t second;
    std::uint32_t composite;
};

#include "tokenizer/unicode_data.inc"

constexpr std::uint32_t kMaxCp = 0x110000;
constexpr std::uint8_t kClassMask = 0x03;
constexpr std::uint8_t kNfcCheckBit = 0x04;

// Hangul (Unicode §3.12).
constexpr char32_t kSBase = 0xAC00, kLBase = 0x1100, kVBase = 0x1161, kTBase = 0x11A7;
constexpr char32_t kLCount = 19, kVCount = 21, kTCount = 28;
constexpr char32_t kNCount = kVCount * kTCount, kSCount = kLCount * kNCount;

// One byte per codepoint (1.1 MB), built once on first use; lookups are O(1).
const std::vector<std::uint8_t>& flag_table() {
    static const std::vector<std::uint8_t> table = [] {
        std::vector<std::uint8_t> t(kMaxCp, 0);
        auto fill = [&t](std::span<const CpRange> ranges, std::uint8_t value, std::uint8_t mask) {
            for (const auto& r : ranges) {
                for (std::uint32_t c = r.first; c <= r.last; ++c) {
                    t[c] = static_cast<std::uint8_t>((t[c] & ~mask) | value);
                }
            }
        };
        fill(kLetterMark, static_cast<std::uint8_t>(CpClass::LetterMark), kClassMask);
        fill(kNumber, static_cast<std::uint8_t>(CpClass::Number), kClassMask);
        fill(kWhitespace, static_cast<std::uint8_t>(CpClass::Space), kClassMask);
        fill(kNfcCheck, kNfcCheckBit, kNfcCheckBit);
        return t;
    }();
    return table;
}

bool contains(std::span<const std::uint32_t> set, char32_t cp) noexcept {
    return std::find(set.begin(), set.end(), static_cast<std::uint32_t>(cp)) != set.end();
}

std::uint8_t ccc_of(char32_t cp) noexcept {
    const auto* it = std::upper_bound(std::begin(kCcc), std::end(kCcc), static_cast<std::uint32_t>(cp),
                                      [](std::uint32_t v, const CccRange& r) { return v < r.first; });
    if (it == std::begin(kCcc)) return 0;
    --it;
    return cp <= it->last ? it->ccc : 0;
}

std::span<const std::uint32_t> decomposition(char32_t cp) noexcept {
    const auto* it = std::lower_bound(std::begin(kDecomp), std::end(kDecomp), static_cast<std::uint32_t>(cp),
                                      [](const DecompEntry& e, std::uint32_t v) { return e.cp < v; });
    if (it == std::end(kDecomp) || it->cp != cp) return {};
    return std::span<const std::uint32_t>(kDecompData).subspan(it->offset, it->len);
}

// Returns 0 when (a, b) does not compose.
char32_t compose(char32_t a, char32_t b) noexcept {
    if (a >= kLBase && a < kLBase + kLCount && b >= kVBase && b < kVBase + kVCount) {
        return kSBase + ((a - kLBase) * kVCount + (b - kVBase)) * kTCount;
    }
    if (a >= kSBase && a < kSBase + kSCount && (a - kSBase) % kTCount == 0 && b > kTBase &&
        b < kTBase + kTCount) {
        return a + (b - kTBase);
    }
    const auto* it = std::lower_bound(std::begin(kComp), std::end(kComp), std::pair<char32_t, char32_t>{a, b},
                                      [](const CompEntry& e, const std::pair<char32_t, char32_t>& k) {
                                          return e.first != k.first ? e.first < k.first : e.second < k.second;
                                      });
    if (it != std::end(kComp) && it->first == a && it->second == b) return it->composite;
    return 0;
}

}  // namespace

CpClass classify(char32_t cp) noexcept {
    if (cp >= kMaxCp) return CpClass::Other;
    return static_cast<CpClass>(flag_table()[cp] & kClassMask);
}

bool fold_single(char32_t cp) noexcept { return contains(kFoldSingle, cp); }
bool fold_re_ve_first(char32_t cp) noexcept { return contains(kFoldReVeFirst, cp); }
bool fold_e_second(char32_t cp) noexcept { return contains(kFoldESecond, cp); }
bool fold_l_first(char32_t cp) noexcept { return contains(kFoldLFirst, cp); }
bool fold_l_second(char32_t cp) noexcept { return contains(kFoldLSecond, cp); }

Utf8Step decode_step(std::string_view s, std::size_t i) noexcept {
    using St = Utf8Step::Status;
    const auto byte = [&](std::size_t k) { return static_cast<unsigned char>(s[k]); };
    const unsigned char b0 = byte(i);
    if (b0 < 0x80) return {St::Ok, b0, 1};
    std::size_t need = 0;
    char32_t cp = 0;
    unsigned char lo = 0x80, hi = 0xBF;  // valid range of the second byte
    if (b0 >= 0xC2 && b0 <= 0xDF) {
        need = 1;
        cp = b0 & 0x1Fu;
    } else if (b0 >= 0xE0 && b0 <= 0xEF) {
        need = 2;
        cp = b0 & 0x0Fu;
        if (b0 == 0xE0) lo = 0xA0;
        if (b0 == 0xED) hi = 0x9F;
    } else if (b0 >= 0xF0 && b0 <= 0xF4) {
        need = 3;
        cp = b0 & 0x07u;
        if (b0 == 0xF0) lo = 0x90;
        if (b0 == 0xF4) hi = 0x8F;
    } else {
        return {St::Invalid, 0, 1};
    }
    for (std::size_t k = 1; k <= need; ++k) {
        if (i + k >= s.size()) return {St::Incomplete, 0, k};
        const unsigned char b = byte(i + k);
        const unsigned char l = k == 1 ? lo : 0x80, h = k == 1 ? hi : 0xBF;
        if (b < l || b > h) return {St::Invalid, 0, k};
        cp = (cp << 6) | (b & 0x3Fu);
    }
    return {St::Ok, cp, need + 1};
}

bool is_valid_utf8(std::string_view s) noexcept {
    std::size_t i = 0;
    while (i < s.size()) {
        if (static_cast<unsigned char>(s[i]) < 0x80) {
            ++i;
            continue;
        }
        const Utf8Step st = decode_step(s, i);
        if (st.status != Utf8Step::Status::Ok) return false;
        i += st.len;
    }
    return true;
}

void append_utf8(std::string& out, char32_t cp) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

std::string to_valid_utf8(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    std::size_t i = 0;
    while (i < s.size()) {
        const Utf8Step st = decode_step(s, i);
        if (st.status == Utf8Step::Status::Ok) {
            out.append(s.substr(i, st.len));
        } else {
            append_utf8(out, kReplacement);
        }
        i += st.len;
    }
    return out;
}

bool nfc_quick_check(std::string_view s) noexcept {
    const auto& table = flag_table();
    std::size_t i = 0;
    while (i < s.size()) {
        const auto b = static_cast<unsigned char>(s[i]);
        if (b < 0x80) {  // ASCII never needs NFC work
            ++i;
            continue;
        }
        const Utf8Step st = decode_step(s, i);
        if (st.status != Utf8Step::Status::Ok) return false;
        if (table[st.cp] & kNfcCheckBit) return false;
        i += st.len;
    }
    return true;
}

void nfc(std::vector<CpOff>& cps) {
    // 1. Full canonical decomposition.
    std::vector<CpOff> d;
    d.reserve(cps.size() + cps.size() / 4);
    for (const CpOff& c : cps) {
        if (c.cp >= kSBase && c.cp < kSBase + kSCount) {
            const char32_t si = c.cp - kSBase;
            d.push_back({kLBase + si / kNCount, c.off});
            d.push_back({kVBase + (si % kNCount) / kTCount, c.off});
            if (si % kTCount != 0) d.push_back({kTBase + si % kTCount, c.off});
            continue;
        }
        const auto dec = decomposition(c.cp);
        if (dec.empty()) {
            d.push_back(c);
        } else {
            for (std::uint32_t x : dec) d.push_back({x, c.off});
        }
    }
    // 2. Canonical ordering: stable sort of every run of non-starters by ccc.
    std::vector<std::uint8_t> cls(d.size());
    for (std::size_t i = 0; i < d.size(); ++i) cls[i] = ccc_of(d[i].cp);
    for (std::size_t i = 0; i < d.size();) {
        if (cls[i] == 0) {
            ++i;
            continue;
        }
        std::size_t j = i;
        std::uint32_t min_off = d[i].off;
        while (j < d.size() && cls[j] != 0) {
            min_off = std::min(min_off, d[j].off);
            ++j;
        }
        if (j - i > 1) {
            std::vector<std::pair<std::uint8_t, char32_t>> run;
            run.reserve(j - i);
            for (std::size_t k = i; k < j; ++k) run.emplace_back(cls[k], d[k].cp);
            const auto by_class = [](const auto& a, const auto& b) { return a.first < b.first; };
            if (!std::is_sorted(run.begin(), run.end(), by_class)) {
                std::stable_sort(run.begin(), run.end(), by_class);
                for (std::size_t k = i; k < j; ++k) {
                    cls[k] = run[k - i].first;
                    d[k] = {run[k - i].second, min_off};
                }
            }
        }
        i = j;
    }
    // 3. Canonical composition (UAX #15 reference algorithm).
    cps.clear();
    if (d.empty()) return;
    cps.push_back(d[0]);
    std::size_t starter = 0;
    int last_class = cls[0] == 0 ? 0 : 256;
    for (std::size_t i = 1; i < d.size(); ++i) {
        const int cc = cls[i];
        const char32_t comp = compose(cps[starter].cp, d[i].cp);
        if (comp != 0 && (last_class < cc || last_class == 0)) {
            cps[starter].cp = comp;
            continue;
        }
        if (cc == 0) starter = cps.size();
        last_class = cc;
        cps.push_back(d[i]);
    }
}

std::string nfc_utf8(std::string_view s) {
    std::vector<CpOff> cps;
    std::size_t i = 0;
    while (i < s.size()) {
        const Utf8Step st = decode_step(s, i);
        cps.push_back({st.status == Utf8Step::Status::Ok ? st.cp : kReplacement, static_cast<std::uint32_t>(i)});
        i += st.len;
    }
    nfc(cps);
    std::string out;
    out.reserve(s.size());
    for (const CpOff& c : cps) append_utf8(out, c.cp);
    return out;
}

}  // namespace halo::tokenizer::unicode
