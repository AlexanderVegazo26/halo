#pragma once
// HALO error model (TRD §45). HALO uses exceptions for errors that cross module
// boundaries; every thrown error carries a typed ErrorCode. Code that must not throw
// (destructors, noexcept paths, C callbacks) converts to ErrorCode + message.

#include <format>
#include <stdexcept>
#include <string>
#include <string_view>

namespace halo {

enum class ErrorCode {
    Model,        // malformed/unsupported model file or metadata
    Memory,       // allocation failure, budget exceeded
    Backend,      // backend API failure
    Kernel,       // kernel launch/validation failure
    Config,       // invalid configuration
    Api,          // bad API request
    Device,       // device discovery / driver problem
    Unsupported,  // valid input, feature not supported (e.g. vision tower in v1)
    Io,           // file / network I/O
    Cancelled,    // request cancelled
};

[[nodiscard]] constexpr std::string_view to_string(ErrorCode c) noexcept {
    switch (c) {
        case ErrorCode::Model: return "MODEL_ERROR";
        case ErrorCode::Memory: return "MEMORY_ERROR";
        case ErrorCode::Backend: return "BACKEND_ERROR";
        case ErrorCode::Kernel: return "KERNEL_ERROR";
        case ErrorCode::Config: return "CONFIG_ERROR";
        case ErrorCode::Api: return "API_ERROR";
        case ErrorCode::Device: return "DEVICE_ERROR";
        case ErrorCode::Unsupported: return "UNSUPPORTED_ERROR";
        case ErrorCode::Io: return "IO_ERROR";
        case ErrorCode::Cancelled: return "CANCELLED";
    }
    return "UNKNOWN_ERROR";
}

class Error : public std::runtime_error {
public:
    Error(ErrorCode code, const std::string& message)
        : std::runtime_error(std::string(to_string(code)) + ": " + message), code_(code) {}

    [[nodiscard]] ErrorCode code() const noexcept { return code_; }

private:
    ErrorCode code_;
};

// throw_error(ErrorCode::Model, "tensor {} missing", name)
template <typename... Args>
[[noreturn]] void throw_error(ErrorCode code, std::format_string<Args...> fmt, Args&&... args) {
    throw Error(code, std::format(fmt, std::forward<Args>(args)...));
}

// Precondition check that stays on in release builds (validation of external data).
#define HALO_CHECK(cond, code, ...)                     \
    do {                                                \
        if (!(cond)) [[unlikely]] {                     \
            ::halo::throw_error((code), __VA_ARGS__);   \
        }                                               \
    } while (0)

}  // namespace halo
