#pragma once
// ============================================================================
// apr_encode_kernels.hpp -- attribute rotation encoding kernels (range package,
// C++ port).
//
// Covers two encoding routes (corresponding to _utils/encode_attrs.py):
//   (A) numeric-range RoPE: angle = (value/L) * rope_freq(j), frequency grows
//       with the dimension-pair index
//       -- base/query range encoding of encode_range_rot.cpp (primary use here)
//   (B) categorical equality hash phase: hash_phase(id, salt) -- kept as a
//       fallback (old encode_range.py / 3.encode_dataset.py route, not used by
//       the current range pipeline)
//
// Shared rotation kernel `encode_row`: per 2D pair float64 rotation then cast
// float32, matching Python `_rotated`; blocks are disjoint and each dimension
// pair is rotated only once, so Python's per-attribute sequential application is
// equivalent to a single once-per-pair rotation, and out-of-block pairs are
// identity (cos0=1, sin0=0 exact) preserving bits.
//
// Known tiny difference (does not affect numerics; user confirmed byte-exact
// consistency is not required): Python's per-attribute identity channel can flip
// a few -0.0 into +0.0 (e.g. 0.0 + (-0.0) = +0.0); this implementation does not
// replicate that side effect.
//
// Trigonometric tables are precomputed per (distinct attribute value x in-block
// dimension pair) (bit-equivalent to per-row computation), reducing N*P sin/cos
// to distinct*|block|.
//
// Numeric path matches Python: float32 in -> double compute -> cast float32.
// Build requires -ffp-contract=off to avoid FMA contraction changing rounding
// (numpy operates element-wise without contraction).
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

namespace apr_encode {

// Dimension-pair index range [begin, end) (corresponding to Python block_split's
// contiguous split).
using Block = std::pair<size_t, size_t>;

// Angle function: attribute value + global dimension-pair index -> rotation angle
// (rad). RoPE route: value/L * rope_freq(j); hash route: hash_phase(value,
// salt[j]).
using AngleFn = double (*)(int64_t value, size_t pair_j);

// Evenly split the P dimension pairs across A attributes (the first P%A
// attributes get one extra pair). Consistent with
// _dataset_utils.py / encode_attrs.py::block_split(A, P, scheme="contiguous").
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

// ---------------- (B) categorical equality: hash phase (fallback) ----------------

// Bit-identical to _hash_to_phases' phase computation:
//   h = (id * 2654435761) ^ (salt * 40503)      # int64 wraparound (mod 2^64)
//   h = (h ^ (h >> 15)) & 0x7FFFFFFF            # arithmetic right shift (numpy int64 semantics)
//   phase = (h / 2147483647.0) * 2.0 * pi
inline double hash_phase(int64_t id, int64_t salt) {
    const int64_t h =
        static_cast<int64_t>(static_cast<uint64_t>(id) * 2654435761ULL ^
                             static_cast<uint64_t>(salt) * 40503ULL);
    const int64_t m = (h ^ (h >> 15)) & 0x7FFFFFFF;  // g++ signed right shift is arithmetic
    return (static_cast<double>(m) / 2147483647.0) * 2.0 * M_PI;
}

// ---------------- (A) numeric range: RoPE frequency ----------------

// Frequency term of rope_angles: freq[j] = (pi * base_freq) * (j + 1.0) / P.
// Operation order strictly matches Python `np.pi * base_freq * (j + 1.0) / P`
// (left to right).
inline double rope_freq(size_t j, size_t P, double base_freq) {
    return (M_PI * base_freq) * (static_cast<double>(j) + 1.0) / static_cast<double>(P);
}

// Integer floor division (replicates Python `//` floor semantics on negatives).
inline int64_t floor_div(int64_t a, int64_t b) {
    const int64_t q = a / b;
    const int64_t r = a % b;
    return (r != 0 && ((r < 0) != (b < 0))) ? q - 1 : q;
}

// ---------------- generic trigonometric table ----------------

// Trig table for one attribute block: organized by distinct attribute values.
// Layout: each slot holds block_size*2 doubles, cos/sin separated and repeated
// per dimension pair:
//   cdup = [c_0, c_0, c_1, c_1, ...]   sdup = [s_0, s_0, s_1, s_1, ...]
// so AVX2 can directly loadu into [c_t, c_t, c_{t+1}, c_{t+1}] /
// [s_t, s_t, s_{t+1}, s_{t+1}].
class AttrTable {
   public:
    AttrTable() = default;

    // Canonical entry: attribute a column (ids[i*A + a], i=0..n-1); angle: value +
    // global dimension-pair index -> angle (rad).
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

    // Fetch the cos/sin table pointer for an attribute value (each length
    // block_size*2, repeated per pair); unrecorded values return false (defensive
    // only -- in normal flow all attr values are already tabled).
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
    std::vector<double> cdup_;  // two cos per pair
    std::vector<double> sdup_;  // two sin per pair
    std::unordered_map<int64_t, uint32_t> slot_;
    size_t block_size_ = 0;
};

// Apply the rotation of every attribute block to one vector row (out-of-block
// dimension pairs are identity, modified in place). tables[a].entry(attrs_row[a])
// gives that attribute value's in-block cos/sin table.
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
        // AVX2: process 2 dimension pairs at a time (4 floats -> 4 doubles),
        // keeping double precision
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
        // scalar fallback (including leftover odd pairs from the AVX2 path)
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

// ---------------- normalization (bit-identical to np.linalg.norm + float32 div) ----------------

// Replicate numpy add.reduce(x*x, axis=-1)'s float32 pairwise summation order:
//   n < 8      sequential accumulation
//   n <= 128   8 accumulators, final round ((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7)),
//              tail sequential
//   n > 128    n2 = n/2 with 8-multiple remainder removed, then bisect:
//              sum(a,n2) + sum(a+n2, n-n2)
// Empirically bit-identical to np.linalg.norm(float32) at d=128/1024/1536.
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

// float32 L2 norm (equivalent to np.linalg.norm(x) on a float32 row); zero
// vector returns 0.
inline float pairwise_norm_f32(const float* x, size_t d) {
    const float s = pairwise_sum_sq_f32(x, d);
    return s > 0.0F ? std::sqrt(s) : 0.0F;
}

}  // namespace apr_encode
