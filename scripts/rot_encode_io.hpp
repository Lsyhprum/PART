#pragma once
// ============================================================================
// rot_encode_io.hpp — 旋转编码器的共享 IO / 多线程调度框架
//
// 自 RaBitQ-inline-rot（eq 包）src/encode_rot.cpp 的匿名命名空间提炼，
// range 包的编码器（encode_range_rot.cpp）共用：
//   - fvecs 探测/流式分块读写（行 = 4B int32 维度头 + d×float32）
//   - 带头自描述属性 bin 读取（8B int32 [N, A] + payload，uint32/int32 -> int64）
//   - 多线程按行分块并行编码（每 chunk 内切分区间，线程写私有区间、按 chunk 顺序写出）
//   - 行范数统计（与 Python 版对齐的报告口径）
//   - 可选的行归一化（pairwise 范数 + float32 除法，位级复刻 numpy）
//
// 依赖 rot_encode_kernels.hpp 的 block_split / AttrTable / encode_row。
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

#include "rot_encode_kernels.hpp"

namespace rot_io {

[[noreturn]] inline void die(const char* fmt, ...) {
    std::fflush(stdout);  // 保证已打印进度先于错误信息出现
    va_list ap;
    va_start(ap, fmt);
    std::fprintf(stderr, "[error] ");
    std::vfprintf(stderr, fmt, ap);
    std::fprintf(stderr, "\n");
    va_end(ap);
    std::exit(1);
}

// 读 fvecs 头部维度并返回行数（校验文件大小按行对齐）
inline size_t fvecs_probe(const std::string& path, int32_t expect_d) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (fp == nullptr) die("无法打开 fvecs 文件: %s", path.c_str());
    int32_t d = 0;
    if (std::fread(&d, sizeof(d), 1, fp) != 1) die("fvecs 文件过短（无维度头）: %s", path.c_str());
    if (d != expect_d) {
        std::fclose(fp);
        die("fvecs 维度不一致: %s 头部 d=%d，期望 %d", path.c_str(), d, expect_d);
    }
    struct stat st{};
    if (stat(path.c_str(), &st) != 0) die("无法 stat: %s", path.c_str());
    const size_t row_bytes = 4 + static_cast<size_t>(d) * 4;
    if (static_cast<size_t>(st.st_size) % row_bytes != 0) {
        std::fclose(fp);
        die("fvecs 文件大小不是行对齐: %s（size=%lld, row_bytes=%zu）", path.c_str(),
            static_cast<long long>(st.st_size), row_bytes);
    }
    std::fclose(fp);
    return static_cast<size_t>(st.st_size) / row_bytes;
}

// 读带头自描述属性 bin：8 字节 int32 [N, A] + N*A payload。
// signed_payload=false -> uint32 零扩展（base）；true -> int32 符号扩展（query，可含 -1）。
inline std::vector<int64_t> attr_bin_read(const std::string& path, bool signed_payload,
                                          int expect_A, size_t expect_n) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (fp == nullptr) die("无法打开属性文件: %s", path.c_str());
    int32_t hdr[2] = {0, 0};
    if (std::fread(hdr, sizeof(hdr), 1, fp) != 1) die("属性文件头过短: %s", path.c_str());
    const size_t n = static_cast<size_t>(hdr[0]);
    if (hdr[1] != expect_A) {
        std::fclose(fp);
        die("属性文件 A 不一致: %s 头部 A=%d，期望 %d", path.c_str(), hdr[1], expect_A);
    }
    if (expect_n != 0 && n != expect_n) {
        std::fclose(fp);
        die("属性文件行数不一致: %s 头部 N=%zu，期望 %zu", path.c_str(), n, expect_n);
    }
    std::vector<int64_t> out(n * static_cast<size_t>(expect_A));
    if (signed_payload) {
        std::vector<int32_t> tmp(out.size());
        if (std::fread(tmp.data(), sizeof(int32_t), tmp.size(), fp) != tmp.size()) {
            std::fclose(fp);
            die("属性文件数据区过短: %s", path.c_str());
        }
        for (size_t i = 0; i < tmp.size(); ++i) out[i] = tmp[i];  // 符号扩展
    } else {
        std::vector<uint32_t> tmp(out.size());
        if (std::fread(tmp.data(), sizeof(uint32_t), tmp.size(), fp) != tmp.size()) {
            std::fclose(fp);
            die("属性文件数据区过短: %s", path.c_str());
        }
        for (size_t i = 0; i < tmp.size(); ++i) out[i] = tmp[i];  // 零扩展
    }
    std::fclose(fp);
    return out;
}

// ------------------------------ 编码调度 ------------------------------

struct NormStat {
    double sum = 0;    // sum(||row||)
    double sum2 = 0;   // sum(||row||^2)
    size_t count = 0;
};

struct EncodeJob {
    std::vector<char>* buf;        // 块缓冲（行 = 4B 头 + d*4B）
    size_t row_bytes;
    size_t d;
    size_t row_off;  // 块首行在文件中的全局行号
    const std::vector<rot_encode::Block>* blocks;
    const std::vector<rot_encode::AttrTable>* tables;
    const std::vector<int64_t>* attrs;  // (n, A) 行主序
    size_t A;
    bool normalize = false;  // true = 编码前按行 L2 归一化（range 版需要）
};

// 编码块内行区间 [loc0, loc1)（块内局部索引），范数统计写入 st
inline void encode_rows(const EncodeJob& job, size_t loc0, size_t loc1, NormStat& st) {
    double s1 = 0.0;
    double s2 = 0.0;
    for (size_t i = loc0; i < loc1; ++i) {
        const size_t gi = job.row_off + i;
        float* v = reinterpret_cast<float*>(job.buf->data() + i * job.row_bytes + 4);
        if (job.normalize) {
            const float nrm = rot_encode::pairwise_norm_f32(v, job.d);
            if (nrm == 0.0F) {
                die("第 %zu 行范数为 0，无法归一化（Python 版会产生 NaN）", gi);
            }
            for (size_t j = 0; j < job.d; ++j) {
                v[j] = v[j] / nrm;  // float32 除法，与 numpy 广播除法一致
            }
        }
        rot_encode::encode_row(v, job.d, *job.blocks, *job.tables,
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

// 流式编码单个 fvecs 文件；返回范数统计（与 Python 版报告口径一致）
inline NormStat encode_file(const std::string& in_path, const std::string& out_path, int d,
                            const std::vector<int64_t>& attrs, size_t A,
                            const std::vector<rot_encode::Block>& blocks,
                            const std::vector<rot_encode::AttrTable>& tables, size_t threads,
                            size_t chunk_rows, bool normalize = false) {
    const size_t n = fvecs_probe(in_path, d);
    if (attrs.size() != n * A) {
        die("属性矩阵与向量行数不一致: %s 属性=%zu 行，向量=%zu 行（A=%zu）", in_path.c_str(),
            attrs.size() / A, n, A);
    }
    const size_t row_bytes = 4 + static_cast<size_t>(d) * 4;

    FILE* fin = std::fopen(in_path.c_str(), "rb");
    if (fin == nullptr) die("无法打开输入: %s", in_path.c_str());
    FILE* fout = std::fopen(out_path.c_str(), "wb");
    if (fout == nullptr) die("无法打开输出: %s", out_path.c_str());
    std::vector<char> buf(chunk_rows * row_bytes);
    // 大块 stdio 缓冲，减少 write 系统调用
    std::vector<char> obuf(1 << 20);
    std::setvbuf(fout, obuf.data(), _IOFBF, obuf.size());

    NormStat total{};
    EncodeJob job{&buf, row_bytes, static_cast<size_t>(d), 0, &blocks, &tables, &attrs, A,
                  normalize};

    for (size_t r0 = 0; r0 < n; r0 += chunk_rows) {
        const size_t rows = (n - r0 < chunk_rows) ? (n - r0) : chunk_rows;
        if (std::fread(buf.data(), row_bytes, rows, fin) != rows) {
            die("读取失败: %s（行 %zu 处）", in_path.c_str(), r0);
        }
        // 校验每行维度头（Python 的 fvecs 读入同样隐含此校验）
        for (size_t i = 0; i < rows; ++i) {
            int32_t h = 0;
            std::memcpy(&h, buf.data() + i * row_bytes, 4);
            if (h != d) die("fvecs 行 %zu 维度头 = %d，期望 %d: %s", r0 + i, h, d, in_path.c_str());
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
            die("写出失败: %s", out_path.c_str());
        }
        for (const auto& s : stats) {
            total.sum += s.sum;
            total.sum2 += s.sum2;
            total.count += s.count;
        }
    }
    std::fclose(fin);
    if (std::fflush(fout) != 0 || std::fclose(fout) != 0) die("写出失败: %s", out_path.c_str());
    return total;
}

inline void copy_file(const std::string& src, const std::string& dst) {
    FILE* in = std::fopen(src.c_str(), "rb");
    if (in == nullptr) die("无法打开待拷贝文件: %s", src.c_str());
    FILE* out = std::fopen(dst.c_str(), "wb");
    if (out == nullptr) die("无法创建拷贝目标: %s", dst.c_str());
    std::vector<char> b(1 << 20);
    size_t r = 0;
    while ((r = std::fread(b.data(), 1, b.size(), in)) > 0) {
        if (std::fwrite(b.data(), 1, r, out) != r) die("拷贝写出失败: %s", dst.c_str());
    }
    std::fclose(in);
    std::fclose(out);
}

// 范数报告的均值/标准差（与 Python 打印口径一致；double 累加）
inline double norm_mean(const NormStat& s) {
    return s.count ? s.sum / static_cast<double>(s.count) : 0.0;
}

inline double norm_std(const NormStat& s) {
    if (s.count == 0) return 0.0;
    const double m = norm_mean(s);
    const double var = s.sum2 / static_cast<double>(s.count) - m * m;
    return var > 0 ? std::sqrt(var) : 0.0;
}

}  // namespace rot_io
