#include "fsutil.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <random>

#include "halo/core/error.h"

namespace halo::cli {

namespace {

std::string errno_text() { return std::strerror(errno); }

}  // namespace

void refuse_symlink(const std::filesystem::path& path, std::string_view what) {
    struct stat st {};
    if (::lstat(path.c_str(), &st) == 0 && S_ISLNK(st.st_mode)) {
        throw_error(ErrorCode::Io, "refusing to use {} {}: it is a symbolic link", what, path.string());
    }
}

void write_file_no_follow(const std::filesystem::path& path, std::string_view content) {
    refuse_symlink(path, "output file");
    const std::filesystem::path dir = path.has_parent_path() ? path.parent_path() : std::filesystem::path(".");
    std::random_device rd;
    std::string tmp;
    int fd = -1;
    for (int attempt = 0; attempt < 16 && fd < 0; ++attempt) {
        tmp = (dir / ("." + path.filename().string() + ".tmp." + std::to_string(rd()))).string();
        fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0644);
        if (fd < 0 && errno != EEXIST) break;
    }
    HALO_CHECK(fd >= 0, ErrorCode::Io, "cannot create a temporary file next to {}: {}", path.string(), errno_text());
    std::size_t done = 0;
    while (done < content.size()) {
        const ssize_t n = ::write(fd, content.data() + done, content.size() - done);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            const std::string why = errno_text();
            ::close(fd);
            ::unlink(tmp.c_str());
            throw_error(ErrorCode::Io, "writing {} failed: {}", path.string(), why);
        }
        done += static_cast<std::size_t>(n);
    }
    if (::fsync(fd) != 0 || ::close(fd) != 0) {
        const std::string why = errno_text();
        ::unlink(tmp.c_str());
        throw_error(ErrorCode::Io, "writing {} failed: {}", path.string(), why);
    }
    if (::rename(tmp.c_str(), path.c_str()) != 0) {
        const std::string why = errno_text();
        ::unlink(tmp.c_str());
        throw_error(ErrorCode::Io, "cannot replace {}: {}", path.string(), why);
    }
}

FileIdentity file_identity(const std::filesystem::path& path) {
    struct stat st {};
    HALO_CHECK(::stat(path.c_str(), &st) == 0, ErrorCode::Io, "cannot stat {}: {}", path.string(), errno_text());
    return {static_cast<std::uint64_t>(st.st_dev), static_cast<std::uint64_t>(st.st_ino),
            static_cast<std::uint64_t>(st.st_size),
            static_cast<std::int64_t>(st.st_mtim.tv_sec) * 1000000000LL + st.st_mtim.tv_nsec};
}

}  // namespace halo::cli
