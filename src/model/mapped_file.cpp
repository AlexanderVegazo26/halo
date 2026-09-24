#include "halo/model/mapped_file.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <utility>

#include "halo/core/error.h"

namespace halo::model {
namespace {

struct Fd {
    int fd = -1;
    ~Fd() {
        if (fd >= 0) ::close(fd);
    }
};

}  // namespace

MappedFile MappedFile::open(const std::filesystem::path& path) {
    Fd f{::open(path.c_str(), O_RDONLY | O_CLOEXEC)};
    if (f.fd < 0) {
        throw_error(ErrorCode::Io, "cannot open {}: {}", path.string(), std::strerror(errno));
    }
    struct stat st {};
    if (::fstat(f.fd, &st) != 0) {
        throw_error(ErrorCode::Io, "cannot stat {}: {}", path.string(), std::strerror(errno));
    }
    if (!S_ISREG(st.st_mode)) {
        throw_error(ErrorCode::Io, "{} is not a regular file", path.string());
    }
    const auto size = static_cast<std::size_t>(st.st_size);
    if (size == 0) {
        return MappedFile{};
    }
    void* p = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, f.fd, 0);
    if (p == MAP_FAILED) {
        throw_error(ErrorCode::Io, "cannot mmap {} ({} bytes): {}", path.string(), size, std::strerror(errno));
    }
    return MappedFile{static_cast<const std::byte*>(p), size};
}

void MappedFile::reset() noexcept {
    if (data_ != nullptr) {
        ::munmap(const_cast<std::byte*>(data_), size_);
    }
    data_ = nullptr;
    size_ = 0;
}

MappedFile::~MappedFile() { reset(); }

MappedFile::MappedFile(MappedFile&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this != &other) {
        reset();
        data_ = std::exchange(other.data_, nullptr);
        size_ = std::exchange(other.size_, 0);
    }
    return *this;
}

}  // namespace halo::model
