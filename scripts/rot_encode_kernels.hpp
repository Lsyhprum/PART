#pragma once
// ============================================================================
// rot_encode_kernels.hpp — 属性旋转编码内核（range 包，C++ 移植）
//
// 覆盖两条编码路线（与 _utils/encode_attrs.py 对应）：
//   (A) 数值范围 RoPE：角度 = (value/L) * rope_freq(j)，freq 随维度对递增
//       —— encode_range_rot.cpp 的 base/query 范围编码（本包主用）
//   (B) 类别等值哈希相位：hash_phase(id, salt) —— 保留备用（旧 encode_range.py
//       / 3.encode_dataset.py 路线，当前 range 流水线不使用）
//
// 公共旋转内核 `encode_row`：逐 2D 对 float64 旋转后 cast float32，与 Python
// `_rotated` 一致；块不相交且每维度对仅旋转一次，故 Python 逐属性顺序施加 ≡
// 单趟每对一次旋转，块外对恒等（cos0=1, sin0=0 精确）bit 保留。
//
// 已知微小差异（不影响数值，用户已确认不需字节级一致）：Python 逐属性恒等通道
// 会把个别 -0.0 翻成 +0.0（如 0.0 + (-0.0) = +0.0）；本实现不复刻该副作用。
//
// 三角函数按「distinct 属性值 × 块内维度对」预计算成表（位级等价于逐行计算），
// 把 N×P 次 sin/cos 降为 distinct×|block| 次。
//
// 数值路径与 Python 一致：float32 读入 -> double 运算 -> cast float32。
// 编译需 -ffp-contract=off，避免 FMA 收缩改变舍入（numpy 逐元素不收缩）。
// ============================================================================
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace rot_encode {

// 维度对索引区间 [begin, end)（对应 Python block_split 的连续切分）
using Block = std::pair<size_t, size_t>;

// 角度函数：属性值 + 全局维度对索引 -> 旋转角（rad）。
// RoPE 路线：value/L * rope_freq(j)；哈希路线：hash_phase(value, salt[j])。
using AngleFn = double (*)(int64_t value, size_t pair_j);

// 把 P 个维度对连续均分给 A 个属性（前 P%A 个属性多分一对）。
// 与 _dataset_utils.py / encode_attrs.py::block_split(A, P, scheme="contiguous") 一致。
inline std::vector<Block> block_split(size_t A, size_t P) {
    std::vector<Block> blocks;
    blocks.reserve(A);
    const size_t base = P / A;
    const size_t rem = P % A;
    size_t start = 0;
    for (size_t a = 0; a < A; ++a) {
        const size_t sz = base + (a < rem ? 1 : 0);
        blocks.emplace_back(start, start + sz);
        start += sz;
    }
    return blocks;
}

// ---------------- (B) 类别等值：哈希相位（备用） ----------------

// 位级复刻 _hash_to_phases 的相位计算：
//   h = (id * 2654435761) ^ (salt * 40503)      # int64 回绕（模 2^64）
//   h = (h ^ (h >> 15)) & 0x7FFFFFFF            # 算术右移（numpy int64 语义）
//   phase = (h / 2147483647.0) * 2.0 * pi
inline double hash_phase(int64_t id, int64_t salt) {
    const int64_t h =
        static_cast<int64_t>(static_cast<uint64_t>(id) * 2654435761ULL ^
                             static_cast<uint64_t>(salt) * 40503ULL);
    const int64_t m = (h ^ (h >> 15)) & 0x7FFFFFFF;  // g++ 有符号右移即算术移位
    return (static_cast<double>(m) / 2147483647.0) * 2.0 * M_PI;
}

// ---------------- (A) 数值范围：RoPE 频率 ----------------

// rope_angles 的频率项：freq[j] = (π * base_freq) * (j + 1.0) / P。
// 运算序与 Python `np.pi * base_freq * (j + 1.0) / P` 严格一致（从左到右）。
inline double rope_freq(size_t j, size_t P, double base_freq) {
    return (M_PI * base_freq) * (static_cast<double>(j) + 1.0) / static_cast<double>(P);
}

// 整数 floor 除法（复刻 Python `//` 对负数的向下取整语义）。
inline int64_t floor_div(int64_t a, int64_t b) {
    const int64_t q = a / b;
    const int64_t r = a % b;
    return (r != 0 && ((r < 0) != (b < 0))) ? q - 1 : q;
}

// ---------------- 通用三角函数表 ----------------

// 单个属性块的三角函数表：按 distinct 属性值组织。
// 布局：每个 slot 占 block_size*2 个 double，cos/sin 分离且按维度对重复：
//   cdup = [c_0, c_0, c_1, c_1, ...]   sdup = [s_0, s_0, s_1, s_1, ...]
// 便于 AVX2 直接 loadu 成 [c_t, c_t, c_{t+1}, c_{t+1}] / [s_t, s_t, s_{t+1}, s_{t+1}]。
class AttrTable {
   public:
    AttrTable() = default;

    // 正式入口：属性 a 一列（ids[i*A + a]，i=0..n-1）；angle: 值+全局维度对 -> 角度(rad)。
    template <typename Fn>
    void build_column(size_t pair_begin, size_t pair_end, const int64_t* ids, size_t n,
                      size_t A, size_t a, Fn&& angle) {
        block_size_ = pair_end - pair_begin;
        slot_.clear();
        if (n == 0 || block_size_ == 0) {
            cdup_.clear();
            sdup_.clear();
            return;
        }
        for (size_t i = 0; i < n; ++i) {
            const int64_t v = ids[i * A + a];
            if (slot_.find(v) == slot_.end()) {
                slot_.emplace(v, static_cast<uint32_t>(slot_.size()));
            }
        }
        cdup_.assign(slot_.size() * block_size_ * 2, 0.0);
        sdup_.assign(slot_.size() * block_size_ * 2, 0.0);
        for (const auto& kv : slot_) {
            const size_t off = static_cast<size_t>(kv.second) * block_size_ * 2;
            for (size_t j = pair_begin; j < pair_end; ++j) {
                const double ph = angle(kv.first, j);
                const double c = std::cos(ph);
                const double s = std::sin(ph);
                const size_t q = off + (j - pair_begin) * 2;
                cdup_[q] = c;
                cdup_[q + 1] = c;
                sdup_[q] = s;
                sdup_[q + 1] = s;
            }
        }
    }

    // 取该属性值的 cos/sin 表指针（各长 block_size*2，按对重复）；
    // 未收录的值返回 false（正常流程中 attrs 全值已建表，仅作防御）。
    [[nodiscard]] bool entry(int64_t value, const double*& cdup, const double*& sdup) const {
        const auto it = slot_.find(value);
        if (it == slot_.end()) {
            cdup = nullptr;
            sdup = nullptr;
            return false;
        }
        const size_t off = static_cast<size_t>(it->second) * block_size_ * 2;
        cdup = cdup_.data() + off;
        sdup = sdup_.data() + off;
        return true;
    }

   private:
    std::vector<double> cdup_;  // 每对两个 cos
    std::vector<double> sdup_;  // 每对两个 sin
    std::unordered_map<int64_t, uint32_t> slot_;
    size_t block_size_ = 0;
};

// 对一行向量施加全部属性块的旋转（块外维度对恒等，原地修改）。
// tables[a].entry(attrs_row[a]) 给出该属性值的块内 cos/sin 表。
inline void encode_row(float* v, size_t d, const std::vector<Block>& blocks,
                       const std::vector<AttrTable>& tables, const int64_t* attrs_row,
                       size_t A) {
    const size_t P = d / 2;
    for (size_t a = 0; a < A; ++a) {
        const size_t begin = blocks[a].first;
        const size_t end = blocks[a].second;
        if (begin >= end || end > P) {
            continue;
        }
        const double* cp = nullptr;
        const double* sp = nullptr;
        if (!tables[a].entry(attrs_row[a], cp, sp)) {
            continue;
        }
        const size_t n = end - begin;
        size_t k = 0;
#if defined(__AVX2__)
        // AVX2：一次处理 2 个维度对（4 个 float -> 4 个 double），保持 double 精度
        for (; k + 2 <= n; k += 2) {
            float* xp = v + 2 * (begin + k);
            const __m256d vd = _mm256_cvtps_pd(_mm_loadu_ps(xp));  // [x0 x1 x2 x3]
            const __m256d cd = _mm256_loadu_pd(cp + 2 * k);        // [c_k c_k c_k1 c_k1]
            const __m256d sd = _mm256_loadu_pd(sp + 2 * k);        // [s_k s_k s_k1 s_k1]
            const __m256d e = _mm256_unpacklo_pd(vd, vd);         // [x0 x0 x2 x2]
            const __m256d o = _mm256_unpackhi_pd(vd, vd);         // [x1 x1 x3 x3]
            const __m256d ne = _mm256_sub_pd(_mm256_mul_pd(e, cd), _mm256_mul_pd(o, sd));
            const __m256d no = _mm256_add_pd(_mm256_mul_pd(e, sd), _mm256_mul_pd(o, cd));
            const __m256d r = _mm256_unpacklo_pd(ne, no);  // [nx0 nx1 nx2 nx3]
            _mm_storeu_ps(xp, _mm256_cvtpd_ps(r));
        }
#endif
        // 标量回退（含 AVX2 路径的剩余奇数对）
        for (; k < n; ++k) {
            const double c = cp[2 * k];
            const double s = sp[2 * k];
            const size_t j = begin + k;
            const double x0 = v[2 * j];
            const double x1 = v[2 * j + 1];
            v[2 * j] = static_cast<float>(x0 * c - x1 * s);
            v[2 * j + 1] = static_cast<float>(x0 * s + x1 * c);
        }
    }
}

// ---------------- 归一化（位级复刻 np.linalg.norm + float32 除法） ----------------

// 复刻 numpy add.reduce(x*x, axis=-1) 的 float32 pairwise 求和顺序：
//   n < 8      顺序累加
//   n <= 128   8 个累加器，末轮 ((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7))，尾部顺序
//   n > 128    n2 = n/2 去掉 8 倍数余量后二分：sum(a,n2) + sum(a+n2, n-n2)
// 实测与 np.linalg.norm(float32) 在 d=128/1024/1536 上逐位一致。
inline float pairwise_sum_sq_f32(const float* a, size_t n) {
    if (n < 8) {
        float r = 0.0F;
        for (size_t i = 0; i < n; ++i) {
            r += a[i] * a[i];
        }
        return r;
    }
    if (n <= 128) {
        float r[8];
        for (size_t k = 0; k < 8; ++k) {
            r[k] = a[k] * a[k];
        }
        const size_t lim = n - (n % 8);
        size_t i = 8;
        for (; i < lim; i += 8) {
            for (size_t k = 0; k < 8; ++k) {
                r[k] += a[i + k] * a[i + k];
            }
        }
        float res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) {
            res += a[i] * a[i];
        }
        return res;
    }
    size_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum_sq_f32(a, n2) + pairwise_sum_sq_f32(a + n2, n - n2);
}

// float32 L2 范数（等价 np.linalg.norm(x) 对 float32 行的结果）；零向量返回 0。
inline float pairwise_norm_f32(const float* x, size_t d) {
    const float s = pairwise_sum_sq_f32(x, d);
    return s > 0.0F ? std::sqrt(s) : 0.0F;
}

}  // namespace rot_encode
