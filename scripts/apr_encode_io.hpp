#pragma once
// ============================================================================
// apr_encode_io.hpp -- shared IO / multithreaded scheduling framework for the
// rotation encoders.
//
// Extracted from the anonymous namespace of RaBitQ-inline-rot (eq package)
// src/encode_rot.cpp, and shared by the range-package encoder
// (encode_range_rot.cpp):
//   - fvecs probe / streaming chunked read-write (row = 4B int32 dim header +
//     d*float32)
//   - header-prefixed self-describing attribute bin read (8B int32 [N, A] +
//     payload, uint32/int32 -> int64)
//   - multithreaded per-row chunked parallel encoding (each chunk splits index
//     ranges; threads write their private ranges, output in chunk order)
//   - per-row norm statistics (matching the Python report convention)
//   - optional per-row normalization (pairwise norm + float32 division,
//     bit-identical to numpy)
//
// Depends on apr_encode_kernels.hpp for block_split / AttrTable / encode_row.
// ============================================================================
#include <sys/stat.h>

#include <cerrno>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "apr_encode_kernels.hpp"

namespace apr_io {

[[noreturn]] inline void die(const char* fmt, ...) {
    std::fflush(stdout);  // ensure printed progress appears before the error message
    va_list ap;
    va_start(ap, fmt);
    std::fprintf(stderr, "[error] ");
    std::vfprintf(stderr, fmt, ap);
    std::fprintf(stderr, "\n");
    va_end(ap);
    std::exit(1);
}

// Read the fvecs dimension header and return the row count (validate file size
// is row-aligned).
inline size_t fvecs_probe(const std::string& path, int32_t expect_d) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (fp == nullptr) die("cannot open fvecs file: %s", path.c_str());
    int32_t d = 0;
    if (std::fread(&d, sizeof(d), 1, fp) != 1) die("fvecs file too short (no dimension header): %s", path.c_str());
    if (d != expect_d) {
        std::fclose(fp);
        die("fvecs dimension mismatch: %s header d=%d, expected %d", path.c_str(), d, expect_d);
    }
    struct stat st{};
    if (stat(path.c_str(), &st) != 0) die("cannot stat: %s", path.c_str());
    const size_t row_bytes = 4 + static_cast<size_t>(d) * 4;
    if (static_cast<size_t>(st.st_size) % row_bytes != 0) {
        std::fclose(fp);
        die("fvecs file size not row-aligned: %s (size=%lld, row_bytes=%zu)", path.c_str(),
            static_cast<long long>(st.st_size), row_bytes);
    }
    std::fclose(fp);
    return static_cast<size_t>(st.st_size) / row_bytes;
}

// Read a header-prefixed self-describing attribute bin: 8-byte int32 [N, A] +
// N*A payload. signed_payload=false -> uint32 zero-extended (base); true ->
// int32 sign-extended (query, may contain -1).
inline std::vector<int64_t> attr_bin_read(const std::string& path, bool signed_payload,
                                          int expect_A, size_t expect_n) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (fp == nullptr) die("cannot open attribute file: %s", path.c_str());
    int32_t hdr[2] = {0, 0};
    if (std::fread(hdr, sizeof(hdr), 1, fp) != 1) die("attribute file header too short: %s", path.c_str());
    const size_t n = static_cast<size_t>(hdr[0]);
    if (hdr[1] != expect_A) {
        std::fclose(fp);
        die("attribute file A mismatch: %s header A=%d, expected %d", path.c_str(), hdr[1], expect_A);
    }
    if (expect_n != 0 && n != expect_n) {
        std::fclose(fp);
        die("attribute file row count mismatch: %s header N=%zu, expected %zu", path.c_str(), n, expect_n);
    }
    std::vector<int64_t> out(n * static_cast<size_t>(expect_A));
    if (signed_payload) {
        std::vector<int32_t> tmp(out.size());
        if (std::fread(tmp.data(), sizeof(int32_t), tmp.size(), fp) != tmp.size()) {
            std::fclose(fp);
            die("attribute file data section too short: %s", path.c_str());
        }
        for (size_t i = 0; i < tmp.size(); ++i) out[i] = tmp[i];  // sign extension
    } else {
        std::vector<uint32_t> tmp(out.size());
        if (std::fread(tmp.data(), sizeof(uint32_t), tmp.size(), fp) != tmp.size()) {
            std::fclose(fp);
            die("attribute file data section too short: %s", path.c_str());
        }
        for (size_t i = 0; i < tmp.size(); ++i) out[i] = tmp[i];  // zero extension
    }
    std::fclose(fp);
    return out;
}

// ------------------------------ encoding schedule ------------------------------

struct NormStat {
    double sum = 0;    // sum(||row||)
    double sum2 = 0;   // sum(||row||^2)
    size_t count = 0;
};

struct EncodeJob {
    std::vector<char>* buf;        // chunk buffer (row = 4B header + d*4B)
    size_t row_bytes;
    size_t d;
    size_t row_off;  // first row's global row index in the file
    const std::vector<apr_encode::Block>* blocks;
    const std::vector<apr_encode::AttrTable>* tables;
    const std::vector<int64_t>* attrs;  // (n, A) row-major
    size_t A;
    bool normalize = false;  // true = L2-normalize each row before encoding (range version needs this)
};

// Encode rows [loc0, loc1) within the chunk (chunk-local indices); write norm
// statistics to st.
inline void encode_rows(const EncodeJob& job, size_t loc0, size_t loc1, NormStat& st) {
    double s1 = 0.0;
    double s2 = 0.0;
    for (size_t i = loc0; i < loc1; ++i) {
        const size_t gi = job.row_off + i;
        float* v = reinterpret_cast<float*>(job.buf->data() + i * job.row_bytes + 4);
        if (job.normalize) {
            const float nrm = apr_encode::pairwise_norm_f32(v, job.d);
            if (nrm == 0.0F) {
                die("row %zu norm is 0, cannot normalize (Python version would produce NaN)", gi);
            }
            for (size_t j = 0; j < job.d; ++j) {
                v[j] = v[j] / nrm;  // float32 division, consistent with numpy broadcast division
            }
        }
        apr_encode::encode_row(v, job.d, *job.blocks, *job.tables,
                               job.attrs->data() + gi * job.A, job.A);
        double n2 = 0.0;
        for (size_t j = 0; j < job.d; ++j) {
            const double x = v[j];
            n2 += x * x;
        }
        const double nrm = std::sqrt(n2);
        s1 += nrm;
        s2 += nrm * nrm;
    }
    st.sum = s1;
    st.sum2 = s2;
    st.count = loc1 - loc0;
}

// Stream-encode a single fvecs file; return norm statistics (matching the Python
// report convention).
inline NormStat encode_file(const std::string& in_path, const std::string& out_path, int d,
                            const std::vector<int64_t>& attrs, size_t A,
                            const std::vector<apr_encode::Block>& blocks,
                            const std::vector<apr_encode::AttrTable>& tables, size_t threads,
                            size_t chunk_rows, bool normalize = false) {
    const size_t n = fvecs_probe(in_path, d);
    if (attrs.size() != n * A) {
        die("attribute matrix row count != vector row count: %s attrs=%zu rows, vectors=%zu rows (A=%zu)",
            in_path.c_str(), attrs.size() / A, n, A);
    }
    const size_t row_bytes = 4 + static_cast<size_t>(d) * 4;

    FILE* fin = std::fopen(in_path.c_str(), "rb");
    if (fin == nullptr) die("cannot open input: %s", in_path.c_str());
    FILE* fout = std::fopen(out_path.c_str(), "wb");
    if (fout == nullptr) die("cannot open output: %s", out_path.c_str());
    std::vector<char> buf(chunk_rows * row_bytes);
    // large stdio buffer to reduce write syscalls
    std::vector<char> obuf(1 << 20);
    std::setvbuf(fout, obuf.data(), _IOFBF, obuf.size());

    NormStat total{};
    EncodeJob job{&buf, row_bytes, static_cast<size_t>(d), 0, &blocks, &tables, &attrs, A,
                  normalize};

    for (size_t r0 = 0; r0 < n; r0 += chunk_rows) {
        const size_t rows = (n - r0 < chunk_rows) ? (n - r0) : chunk_rows;
        if (std::fread(buf.data(), row_bytes, rows, fin) != rows) {
            die("read failed: %s (at row %zu)", in_path.c_str(), r0);
        }
        // validate each row's dimension header (Python's fvecs read implies the same check)
        for (size_t i = 0; i < rows; ++i) {
            int32_t h = 0;
            std::memcpy(&h, buf.data() + i * row_bytes, 4);
            if (h != d) die("fvecs row %zu dim header = %d, expected %d: %s", r0 + i, h, d, in_path.c_str());
        }
        job.row_off = r0;
        std::vector<NormStat> stats(threads);
        std::vector<std::thread> ts;
        ts.reserve(threads);
        const size_t per = (rows + threads - 1) / threads;
        for (size_t t = 0; t < threads; ++t) {
            const size_t loc0 = t * per;
            if (loc0 >= rows) break;
            const size_t loc1 = (loc0 + per < rows) ? (loc0 + per) : rows;
            ts.emplace_back(encode_rows, std::cref(job), loc0, loc1, std::ref(stats[t]));
        }
        for (auto& th : ts) th.join();
        if (std::fwrite(buf.data(), row_bytes, rows, fout) != rows) {
            die("write failed: %s", out_path.c_str());
        }
        for (const auto& s : stats) {
            total.sum += s.sum;
            total.sum2 += s.sum2;
            total.count += s.count;
        }
    }
    std::fclose(fin);
    if (std::fflush(fout) != 0 || std::fclose(fout) != 0) die("write failed: %s", out_path.c_str());
    return total;
}

inline void copy_file(const std::string& src, const std::string& dst) {
    FILE* in = std::fopen(src.c_str(), "rb");
    if (in == nullptr) die("cannot open file to copy: %s", src.c_str());
    FILE* out = std::fopen(dst.c_str(), "wb");
    if (out == nullptr) die("cannot create copy target: %s", dst.c_str());
    std::vector<char> b(1 << 20);
    size_t r = 0;
    while ((r = std::fread(b.data(), 1, b.size(), in)) > 0) {
        if (std::fwrite(b.data(), 1, r, out) != r) die("copy write failed: %s", dst.c_str());
    }
    std::fclose(in);
    std::fclose(out);
}

// Norm-report mean/std (consistent with Python print convention; double accum).
inline double norm_mean(const NormStat& s) {
    return s.count ? s.sum / static_cast<double>(s.count) : 0.0;
}

inline double norm_std(const NormStat& s) {
    if (s.count == 0) return 0.0;
    const double m = norm_mean(s);
    const double var = s.sum2 / static_cast<double>(s.count) - m * m;
    return var > 0 ? std::sqrt(var) : 0.0;
}

}  // namespace apr_io
