#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "rabitqlib/attribute/types.hpp"

namespace rabitqlib::attribute {

// Attribute Phase Rotation (APR) plan. It is an index-agnostic, immutable value
// object: it stores the dimension, the number of attributes and the per-pair
// salt, and contains no reference to IVF, SymphonyQG or any quantizer. Given
// the same salt, base vectors rotated by their own attributes and query vectors
// rotated by the target attributes preserve exact distances on full matches.
class AprRotationPlan {
   public:
    // salt holds attr_dim * (dim / 2) uint64 values, laid out as A blocks of P.
    // Use the NumPy snippet in docs/attribute-rotation.md to reproduce the
    // salt of published experiments.
    static AprRotationPlan make(
        const AprConfig& config, const uint64_t* salt, size_t salt_len
    );

    // Pure-C++ splitmix salt for environments without Python. The resulting
    // rotation is NOT comparable with published experiments.
    static AprRotationPlan make_standalone(const AprConfig& config, uint64_t seed);

    // HAPR layouts add a deterministic scaled phase for every range column on
    // top of the salted hash phases of the equality columns. Both factories
    // keep the same salt contract (attr_dim blocks of dim/2 uint64) so an
    // all-equality HAPR plan is byte-identical to make()/make_standalone().
    // range_l is the numeric value range; it must be positive whenever the
    // layout contains a range column.
    static AprRotationPlan make_hapr(
        const AprConfig& config,
        const uint64_t* salt,
        size_t salt_len,
        const AttributeKind* layout,
        size_t layout_len,
        double range_l
    );
    static AprRotationPlan make_hapr_standalone(
        const AprConfig& config,
        uint64_t seed,
        const AttributeKind* layout,
        size_t layout_len,
        double range_l
    );

    // CPR (categorical disjoint-block) factories. The salt contract matches
    // make()/make_standalone() (attr_dim blocks of dim/2 uint64), so the same
    // salt file layout is shared across the encoders.
    static AprRotationPlan make_cpr(
        const AprConfig& config, const uint64_t* salt, size_t salt_len
    );
    static AprRotationPlan make_cpr_standalone(const AprConfig& config, uint64_t seed);

    // SPR (scaled-phase rotation) factories — the pure-range specialisation of
    // HAPR from the PART paper (Algorithm III). Every column is a numeric range
    // column, so the plan layout is entirely AttributeKind::kRange and the
    // shared salt contract (attr_dim blocks of dim/2 uint64) is unchanged.
    static AprRotationPlan make_spr(
        const AprConfig& config,
        const uint64_t* salt,
        size_t salt_len,
        double range_l
    );
    static AprRotationPlan make_spr_standalone(
        const AprConfig& config, uint64_t seed, double range_l
    );

    [[nodiscard]] size_t dim() const { return config_.dim; }
    [[nodiscard]] size_t attr_dim() const { return config_.attr_dim; }
    [[nodiscard]] size_t pair_dim() const { return pair_dim_; }
    [[nodiscard]] const uint64_t* salt() const { return salt_.data(); }
    [[nodiscard]] const std::vector<AttributeKind>& layout() const { return layout_; }
    [[nodiscard]] bool has_range() const { return has_range_; }
    [[nodiscard]] double range_l() const { return range_l_; }

   private:
    AprRotationPlan(AprConfig config, std::vector<uint64_t> salt);
    AprRotationPlan(
        AprConfig config,
        std::vector<uint64_t> salt,
        std::vector<AttributeKind> layout,
        bool has_range,
        double range_l
    );

    AprConfig config_;
    size_t pair_dim_ = 0;
    std::vector<uint64_t> salt_;
    std::vector<AttributeKind> layout_;
    bool has_range_ = false;
    double range_l_ = 0.0;
};

// Rotate one row. Base attributes are uint32; query targets are int32 and must
// be non-negative (in particular -1 wildcards are rejected by the rotation
// layer, while attr_match keeps the wildcard semantics for filtering gates).
void rotate_vector(
    const AprRotationPlan& plan, const float* src, const uint32_t* attrs, float* dst
);
void rotate_vector(
    const AprRotationPlan& plan, const float* src, const int32_t* attrs, float* dst
);

// Rotate a row-major batch. num_threads = 0 runs serially; larger values enable
// row-wise parallelism when the library is built with OpenMP.
void rotate_batch(
    const AprRotationPlan& plan,
    const float* src,
    size_t rows,
    const uint32_t* attrs,
    float* dst,
    size_t num_threads = 0
);
void rotate_batch(
    const AprRotationPlan& plan,
    const float* src,
    size_t rows,
    const int32_t* attrs,
    float* dst,
    size_t num_threads = 0
);

// HAPR entry point. values is a row-major rows x attr_dim matrix of doubles:
// equality columns carry non-negative integer values, range columns carry the
// numeric value (base) or the interval centre (query), which may be a half
// integer such as 20.5. The plan's layout decides per column whether a salted
// hash phase or the scaled phase pi*(j/P)*(value/L) is accumulated; the shared
// phase vector is then applied by the same Givens rotation used above.
void rotate_batch_hapr(
    const AprRotationPlan& plan,
    const float* src,
    size_t rows,
    const double* values,
    float* dst,
    size_t num_threads = 0
);
void rotate_vector_hapr(
    const AprRotationPlan& plan, const float* src, const double* values, float* dst
);

// CPR (categorical disjoint-block) entry point. Unlike APR/HAPR, the A attributes
// do NOT share every dimension pair: block_split partitions the P pairs into A
// disjoint, contiguous blocks, and attribute a rotates only its own block. The
// phase for pair j inside attribute a's block is the categorical hash phase
// hash(value_a, salt[a*P + j]) == (value*2654435761) ^ (salt*40503), matching
// the former Python CPR encoder so existing rotated datasets stay reproducible
// from the same --salt-file. The salt contract (attr_dim blocks of dim/2 uint64)
// matches make()/make_standalone() so the two encoders share the same salt file
// layout.
void rotate_batch_cpr(
    const AprRotationPlan& plan,
    const float* src,
    size_t rows,
    const uint32_t* attrs,
    float* dst,
    size_t num_threads = 0
);
void rotate_batch_cpr(
    const AprRotationPlan& plan,
    const float* src,
    size_t rows,
    const int32_t* attrs,
    float* dst,
    size_t num_threads = 0
);
void rotate_vector_cpr(
    const AprRotationPlan& plan, const float* src, const uint32_t* attrs, float* dst
);
void rotate_vector_cpr(
    const AprRotationPlan& plan, const float* src, const int32_t* attrs, float* dst
);

// SPR (scaled-phase rotation) entry point. Identical signature to the HAPR
// entry point: range columns carry the numeric value (base) or the interval
// centre (query). The rotation reuses compute_entry_hapr's scaled phase, so an
// SPR plan (all-range layout) is byte-identical to a HAPR plan with the same
// all-range layout and the same salt.
void rotate_vector_spr(
    const AprRotationPlan& plan, const float* src, const double* values, float* dst
);
void rotate_batch_spr(
    const AprRotationPlan& plan,
    const float* src,
    size_t rows,
    const double* values,
    float* dst,
    size_t num_threads = 0
);

}  // namespace rabitqlib::attribute
