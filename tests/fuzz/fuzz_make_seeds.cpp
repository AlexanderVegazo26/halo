// Writes the binary GGUF seeds of the fuzz corpus (tests/fuzz/corpus/gguf) from the same
// in-memory builders the model tests use. Rerun after changing tiny_qwen35:
//   build/tests/fuzz/fuzz_make_seeds tests/fuzz/corpus/gguf

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "gguf_builder.h"

namespace {

bool write(const std::filesystem::path& p, const std::vector<std::byte>& b) {
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
    std::printf("%s: %zu bytes\n", p.c_str(), b.size());
    return static_cast<bool>(out);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <corpus/gguf dir>\n", argv[0]);
        return 2;
    }
    const std::filesystem::path dir(argv[1]);
    std::filesystem::create_directories(dir);
    using halo::test::tiny_qwen35;
    bool ok = write(dir / "tiny_qwen35_mtp.header.gguf", tiny_qwen35().build(false));
    ok = write(dir / "tiny_qwen35.header.gguf", tiny_qwen35({.mtp = false}).build(false)) && ok;
    ok = write(dir / "tiny_qwen35_mtp_file.header.gguf", halo::test::tiny_qwen35_mtp_file().build(false)) && ok;
    return ok ? 0 : 1;
}
