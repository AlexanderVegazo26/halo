// Replay driver for the fuzz targets in a normal (non-libFuzzer) build: runs
// LLVMFuzzerTestOneInput once per file named on the command line (directories are expanded
// to their regular files, sorted, so the order is stable). A target signals a bug by
// aborting, so a crash here names the input that caused it on the line just before.
//
//   fuzz_gguf tests/fuzz/corpus/gguf            # the whole seed corpus
//   fuzz_gguf crash-0123abcd                    # one reproducer from a libFuzzer run

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size);

namespace {

std::vector<std::filesystem::path> expand(int argc, char** argv) {
    std::vector<std::filesystem::path> out;
    for (int i = 1; i < argc; ++i) {
        const std::filesystem::path p(argv[i]);
        if (std::filesystem::is_directory(p)) {
            std::vector<std::filesystem::path> files;
            for (const auto& e : std::filesystem::directory_iterator(p)) {
                if (e.is_regular_file()) files.push_back(e.path());
            }
            std::ranges::sort(files);
            out.insert(out.end(), files.begin(), files.end());
        } else {
            out.push_back(p);
        }
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <input file or corpus dir>...\n", argv[0]);
        return 2;
    }
    const auto inputs = expand(argc, argv);
    if (inputs.empty()) {
        std::fprintf(stderr, "replay: no input files under the given paths (empty corpus?)\n");
        return 1;
    }
    for (const auto& path : inputs) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            std::fprintf(stderr, "replay: cannot read %s\n", path.c_str());
            return 1;
        }
        const std::vector<char> bytes{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
        std::fprintf(stderr, "replay: %s (%zu bytes)\n", path.c_str(), bytes.size());
        std::fflush(stderr);
        (void)LLVMFuzzerTestOneInput(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size());
    }
    std::printf("replay: %zu inputs ok\n", inputs.size());
    return 0;
}
