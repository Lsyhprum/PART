#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "rabitqlib/utils/path.hpp"

namespace rabitqlib::attribute {
namespace io_detail {

inline size_t checked_mul(size_t lhs, size_t rhs) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        throw std::length_error("Attribute file layout exceeds size_t");
    }
    return lhs * rhs;
}

inline std::vector<char> read_all_bytes(const char* filename) {
    const std::filesystem::path path = io_impl::filesystem_path(filename);
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error(std::string("File does not exist: ") + filename);
    }
    const auto file_size = std::filesystem::file_size(path);
    if (file_size > std::numeric_limits<size_t>::max()) {
        throw std::length_error("Attribute file is too large to address");
    }
    const auto bytes = static_cast<size_t>(file_size);
    if (bytes > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::length_error("Attribute file is too large to read");
    }
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error(std::string("Cannot open attribute file: ") + filename);
    }
    std::vector<char> buffer(bytes);
    if (bytes != 0) {
        input.read(buffer.data(), static_cast<std::streamsize>(bytes));
        if (!input) {
            throw std::runtime_error(std::string("Unexpected end of file: ") + filename);
        }
    }
    return buffer;
}

inline std::ofstream open_output(const char* filename) {
    std::ofstream output(io_impl::filesystem_path(filename), std::ios::binary);
    if (!output.is_open()) {
        throw std::runtime_error(std::string("Cannot open output file: ") + filename);
    }
    return output;
}

template <typename T>
size_t payload_bytes(size_t rows, size_t cols) {
    return checked_mul(checked_mul(rows, cols), sizeof(T));
}

}  // namespace io_detail

// Row-major, owning matrix of trivially copyable values. Its constructor,
// data() and operator() follow the contract of rabitqlib::load_vecs / load_bin,
// so callers can reuse those readers without depending on an index component.
template <typename T>
struct AttributeMatrix {
    static_assert(std::is_trivially_copyable_v<T>);

    std::vector<T> buf;
    size_t rows = 0;
    size_t cols = 0;

    AttributeMatrix() = default;
    AttributeMatrix(size_t rows_in, size_t cols_in)
        : buf(io_detail::checked_mul(rows_in, cols_in)), rows(rows_in), cols(cols_in) {}

    T* data() { return buf.data(); }
    const T* data() const { return buf.data(); }
    T& operator()(size_t i, size_t j) { return buf[i * cols + j]; }
    const T& operator()(size_t i, size_t j) const { return buf[i * cols + j]; }
};

// Attribute files use an 8-byte little-endian [rows, cols] uint32 header
// followed by rows*cols values of type T in row-major order. A headerless file
// is accepted only when both expect_rows and expect_cols are supplied.
template <typename T>
AttributeMatrix<T> read_attr_bin(
    const char* filename, size_t expect_rows = 0, size_t expect_cols = 0
) {
    static_assert(
        std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t>,
        "Attribute payload must be uint32_t or int32_t"
    );
    const std::vector<char> bytes = io_detail::read_all_bytes(filename);
    constexpr size_t kHeaderBytes = 2 * sizeof(uint32_t);

    size_t rows = 0;
    size_t cols = 0;
    size_t offset = 0;
    bool has_header = false;

    if (bytes.size() >= kHeaderBytes) {
        uint32_t header_rows = 0;
        uint32_t header_cols = 0;
        std::memcpy(&header_rows, bytes.data(), sizeof(uint32_t));
        std::memcpy(&header_cols, bytes.data() + sizeof(uint32_t), sizeof(uint32_t));
        if (header_rows != 0 && header_cols != 0) {
            const size_t payload =
                io_detail::payload_bytes<T>(header_rows, header_cols);
            if (bytes.size() - kHeaderBytes == payload) {
                rows = header_rows;
                cols = header_cols;
                offset = kHeaderBytes;
                has_header = true;
            }
        }
    }

    if (!has_header) {
        if (expect_rows == 0 || expect_cols == 0) {
            throw std::runtime_error(
                std::string("Attribute file has no usable header: ") + filename
            );
        }
        const size_t payload = io_detail::payload_bytes<T>(expect_rows, expect_cols);
        if (bytes.size() != payload) {
            throw std::runtime_error(
                std::string("Attribute payload size does not match expected shape: ") +
                filename
            );
        }
        rows = expect_rows;
        cols = expect_cols;
        offset = 0;
    }

    if ((expect_rows != 0 && rows != expect_rows) ||
        (expect_cols != 0 && cols != expect_cols)) {
        throw std::runtime_error(
            std::string("Attribute dimensions do not match expected shape: ") + filename
        );
    }

    AttributeMatrix<T> matrix(rows, cols);
    const size_t payload = io_detail::payload_bytes<T>(rows, cols);
    if (payload != 0) {
        std::memcpy(matrix.data(), bytes.data() + offset, payload);
    }
    return matrix;
}

template <typename T>
void write_attr_bin(const char* filename, const T* data, size_t rows, size_t cols) {
    static_assert(
        std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t>,
        "Attribute payload must be uint32_t or int32_t"
    );
    if (rows > std::numeric_limits<uint32_t>::max() ||
        cols > std::numeric_limits<uint32_t>::max()) {
        throw std::length_error("Attribute dimensions exceed the uint32 file header");
    }
    const size_t payload = io_detail::payload_bytes<T>(rows, cols);
    if (payload != 0 && data == nullptr) {
        throw std::invalid_argument("Attribute data must not be null");
    }
    std::ofstream output = io_detail::open_output(filename);
    const uint32_t header[2] = {
        static_cast<uint32_t>(rows), static_cast<uint32_t>(cols)
    };
    output.write(reinterpret_cast<const char*>(header), sizeof(header));
    if (payload != 0) {
        output.write(
            reinterpret_cast<const char*>(data), static_cast<std::streamsize>(payload)
        );
    }
    if (!output) {
        throw std::runtime_error(
            std::string("Failed to write attribute file: ") + filename
        );
    }
}

// Salt files hold attr_dim * pair_dim little-endian uint64 values (the bit
// pattern of the int64 values produced by the NumPy generator).
inline std::vector<uint64_t> read_salt_file(
    const char* filename, size_t attr_dim, size_t pair_dim
) {
    const size_t count = io_detail::checked_mul(attr_dim, pair_dim);
    const size_t bytes = io_detail::checked_mul(count, sizeof(uint64_t));
    const std::vector<char> raw = io_detail::read_all_bytes(filename);
    if (raw.size() != bytes) {
        throw std::runtime_error(
            std::string("Salt file size does not match A*(D/2) uint64 values: ") + filename
        );
    }
    std::vector<uint64_t> salt(count);
    if (bytes != 0) {
        std::memcpy(salt.data(), raw.data(), bytes);
    }
    return salt;
}

// Writes the standard fvecs layout: per row one uint32 dimension followed by
// dim float32 values.
inline void write_fvecs(
    const char* filename, const float* data, size_t rows, size_t dim
) {
    if (dim == 0 || dim > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("fvecs dimension must fit in uint32");
    }
    const size_t row_payload = io_detail::checked_mul(dim, sizeof(float));
    const size_t total = io_detail::checked_mul(rows, dim);
    if (total != 0 && data == nullptr) {
        throw std::invalid_argument("fvecs data must not be null");
    }
    std::ofstream output = io_detail::open_output(filename);
    const uint32_t header = static_cast<uint32_t>(dim);
    for (size_t i = 0; i < rows; ++i) {
        output.write(reinterpret_cast<const char*>(&header), sizeof(uint32_t));
        output.write(
            reinterpret_cast<const char*>(data + i * dim),
            static_cast<std::streamsize>(row_payload)
        );
    }
    if (!output) {
        throw std::runtime_error(std::string("Failed to write fvecs file: ") + filename);
    }
}

}  // namespace rabitqlib::attribute
