// Fuzz target: the GGUF parser (both modes) + model validation + tokenizer construction.
// Invariant: any input either loads or throws halo::Error; another exception type, a crash
// or (in sanitizer builds) a memory error / UB is a bug. Seeds: tests/fuzz/corpus/gguf,
// written by fuzz_make_seeds from the tests' tiny_qwen35 builders.

#include <cstddef>
#include <cstdint>
#include <exception>
#include <vector>

#include "fuzz_check.h"
#include "halo/core/error.h"
#include "halo/model/gguf.h"
#include "halo/model/model.h"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    using halo::model::GgufFile;
    using halo::model::GgufMode;
    const auto* first = reinterpret_cast<const std::byte*>(data);
    for (GgufMode mode : {GgufMode::HeaderOnly, GgufMode::Full}) {
        try {
            GgufFile f = GgufFile::parse(std::vector<std::byte>(first, first + size), mode, "fuzz");
            halo::model::NormalizedModel m = halo::model::NormalizedModel::from_gguf(std::move(f));
            (void)m.tokenizer();
        } catch (const halo::Error&) {
            // typed rejection: the expected outcome for most inputs
        } catch (const std::exception& e) {
            HALO_FUZZ_FAIL("non-halo exception (mode %s): %s", mode == GgufMode::Full ? "Full" : "HeaderOnly",
                           e.what());
        }
    }
    return 0;
}
