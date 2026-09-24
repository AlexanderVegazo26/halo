#pragma once
// Read-only memory-mapped file (POSIX mmap), RAII. Move-only.

#include <cstddef>
#include <filesystem>
#include <span>

namespace halo::model {

class MappedFile {
public:
    /// Maps the whole file read-only. Throws Error(Io) if it cannot be opened, stat'ed or
    /// mapped. An empty file maps to an empty span (no mapping is created).
    static MappedFile open(const std::filesystem::path& path);

    MappedFile() noexcept = default;
    ~MappedFile();
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&& other) noexcept;
    MappedFile& operator=(MappedFile&& other) noexcept;

    [[nodiscard]] std::span<const std::byte> bytes() const noexcept { return {data_, size_}; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }

private:
    MappedFile(const std::byte* data, std::size_t size) noexcept : data_(data), size_(size) {}
    void reset() noexcept;

    const std::byte* data_ = nullptr;
    std::size_t size_ = 0;
};

}  // namespace halo::model
