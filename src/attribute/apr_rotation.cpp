#include "rabitqlib/attribute/apr_rotation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rabitqlib::attribute {
namespace {

constexpr uint64_t kMixC1 = 0xBF58476D1CE4E5B9ULL;
constexpr uint64_t kMixC2 = 0x94D049BB133111EBULL;
constexpr uint64_t kGoldenRatio = 0x9E3779B97F4A7C15ULL;
constexpr uint64_t kPhaseMask = 0x7FFFFFFFULL;
constexpr double kPhaseDivisor = 2147483647.0;
constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * 3.14159265358979323846;
constexpr size_t kMaxTableBytes = 64ULL * 1024U * 1024U;

inline uint64_t mix64(uint64_t x) {
    x = (x ^ (x >> 30)) * kMixC1;
    x = (x ^ (x >> 27)) * kMixC2;
    return x ^ (x >> 31);
}

inline uint64_t splitmix64_next(uint64_t& state) {
    uint64_t z = (state += kGoldenRatio);
    z = (z ^ (z >> 30)) * kMixC1;
    z = (z ^ (z >> 27)) * kMixC2;
    return z ^ (z >> 31);
}

size_t checked_mul(size_t lhs, size_t rhs) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        throw std::length_error("APR layout exceeds size_t");
    }
    return lhs * rhs;
}

void validate_config(const AprConfig& config) {
    if (config.dim == 0 || config.dim % 2 != 0) {
        throw std::invalid_argument("APR dimension must be a positive even number");
    }
    if (config.attr_dim == 0) {
        throw std::invalid_argument("APR attribute dimension must be positive");
    }
    if (config.predicate != PredicateKind::kEquality &&
        config.predicate != PredicateKind::kRange) {
        throw std::invalid_argument("APR supports equality and range predicates only");
    }
    if (config.wildcard != WildcardPolicy::kReject) {
        throw std::invalid_argument("APR supports the reject-wildcard policy only");
    }
}

// A HAPR plan keeps one column per attribute. Range columns need a positive
// value range; all-equality layouts must leave it at zero so they cannot pick
// up a scaled phase by accident.
bool validate_hapr_layout(
    const AprConfig& config,
    const AttributeKind* layout,
    size_t layout_len,
    double range_l
) {
    if (layout == nullptr || layout_len != config.attr_dim) {
        throw std::invalid_argument("HAPR layout length must equal attr_dim");
    }
    bool has_range = false;
    for (size_t a = 0; a < layout_len; ++a) {
        if (layout[a] == AttributeKind::kRange) {
            has_range = true;
        } else if (layout[a] != AttributeKind::kEquality) {
            throw std::invalid_argument("HAPR layout has an unknown attribute kind");
        }
    }
    if (has_range && !(range_l > 0.0)) {
        throw std::invalid_argument(
            "HAPR range_l must be positive when the layout has range columns"
        );
    }
    if (!has_range && range_l != 0.0) {
        throw std::invalid_argument(
            "HAPR range_l must be zero for an all-equality layout"
        );
    }
    return has_range;
}

// One cos/sin entry per distinct attribute tuple. total is a scratch buffer of
// P doubles and out receives 2P doubles (cos, sin per dimension pair).
template <typename AttrT>
void compute_entry(
    const uint64_t* salt,
    size_t attr_dim,
    size_t pair_dim,
    const AttrT* values,
    double* total,
    double* out
) {
    std::fill(total, total + pair_dim, 0.0);
    for (size_t a = 0; a < attr_dim; ++a) {
        const uint64_t v = static_cast<uint64_t>(static_cast<uint32_t>(values[a]));
        const uint64_t base = mix64(v * kGoldenRatio);
        const uint64_t* salt_row = salt + a * pair_dim;
        for (size_t j = 0; j < pair_dim; ++j) {
            const uint64_t h = mix64(base ^ salt_row[j]) & kPhaseMask;
            total[j] += (static_cast<double>(h) / kPhaseDivisor) * kTwoPi;
        }
    }
    for (size_t j = 0; j < pair_dim; ++j) {
        out[2 * j] = std::cos(total[j]);
        out[2 * j + 1] = std::sin(total[j]);
    }
}

// HAPR parallel path. Equality columns reuse exactly the salted hash chain
// above; range columns add the deterministic scaled phase
// pi * (j / P) * (value / L), with j the global pair index 0..P-1 (no +1
// offset, no per-block renumbering, no base_freq). Phases from all columns are
// summed in the same double buffer before one cos/sin per pair.
void compute_entry_hapr(
    const uint64_t* salt,
    size_t attr_dim,
    size_t pair_dim,
    const AttributeKind* layout,
    double range_l,
    const double* values,
    double* total,
    double* out
) {
    std::fill(total, total + pair_dim, 0.0);
    for (size_t a = 0; a < attr_dim; ++a) {
        if (layout[a] == AttributeKind::kEquality) {
            const uint64_t v = static_cast<uint64_t>(static_cast<uint32_t>(values[a]));
            const uint64_t base = mix64(v * kGoldenRatio);
            const uint64_t* salt_row = salt + a * pair_dim;
            for (size_t j = 0; j < pair_dim; ++j) {
                const uint64_t h = mix64(base ^ salt_row[j]) & kPhaseMask;
                total[j] += (static_cast<double>(h) / kPhaseDivisor) * kTwoPi;
            }
        } else {
            for (size_t j = 0; j < pair_dim; ++j) {
                const double scale =
                    (kPi * static_cast<double>(j) / static_cast<double>(pair_dim)) /
                    range_l;
                total[j] += values[a] * scale;
            }
        }
    }
    for (size_t j = 0; j < pair_dim; ++j) {
        out[2 * j] = std::cos(total[j]);
        out[2 * j + 1] = std::sin(total[j]);
    }
}

// Transparent comparator enables allocation-free lookup by string_view over the
// raw attribute row bytes. std::map is sufficient here: the number of distinct
// tuples is small (at most the attribute cardinality product).
struct BytesLess {
    using is_transparent = void;
    bool operator()(std::string_view lhs, std::string_view rhs) const noexcept {
        return lhs < rhs;
    }
};
using SlotMap = std::map<std::string, size_t, BytesLess>;

// Splits each multiply and add into independent double statements so that no
// FMA contraction can change the float32 result relative to the NumPy reference.
inline void rotate_row(
    const float* src, float* dst, size_t pair_dim, const double* cos_sin
) {
    for (size_t j = 0; j < pair_dim; ++j) {
        const double c = cos_sin[2 * j];
        const double s = cos_sin[2 * j + 1];
        const double x0 = static_cast<double>(src[2 * j]);
        const double x1 = static_cast<double>(src[2 * j + 1]);
        const double first = x0 * c;
        const double second = x1 * s;
        const double third = x0 * s;
        const double fourth = x1 * c;
        dst[2 * j] = static_cast<float>(first - second);
        dst[2 * j + 1] = static_cast<float>(third + fourth);
    }
}

template <typename F>
void run_rows(size_t rows, size_t num_threads, F&& body) {
#if RABITQ_USE_OPENMP
    if (num_threads > 1) {
        const int workers = static_cast<int>(std::min<size_t>(num_threads, rows));
        #pragma omp parallel for schedule(static) num_threads(workers)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(rows); ++i) {
            body(static_cast<size_t>(i));
        }
        return;
    }
#else
    (void)num_threads;
#endif
    for (size_t i = 0; i < rows; ++i) {
        body(i);
    }
}

// ---------------------------------------------------------------------------
// CPR (categorical disjoint-block) helpers.
//
// block_split_cpr partitions the P dimension pairs into A contiguous, disjoint
// blocks (the first P%A blocks get one extra pair). Every pair belongs to
// exactly one attribute's block, so a CPR rotation touches each pair once.
// ---------------------------------------------------------------------------
std::vector<std::pair<size_t, size_t>> block_split_cpr(size_t attr_dim, size_t pair_dim) {
    std::vector<std::pair<size_t, size_t>> blocks;
    blocks.reserve(attr_dim);
    const size_t base = pair_dim / attr_dim;
    const size_t rem = pair_dim % attr_dim;
    size_t start = 0;
    for (size_t a = 0; a < attr_dim; ++a) {
        const size_t sz = base + (a < rem ? 1 : 0);
        blocks.emplace_back(start, start + sz);
        start += sz;
    }
    return blocks;
}

// Categorical hash phase for CPR (bit-identical layout to the attribute salt
// protocol):
//     h = (id * 2654435761) ^ (salt * 40503)
//     h = (h ^ (h >> 15)) & 0x7FFFFFFF
//     phase = (h / 0x7FFFFFFF) * 2.0 * pi
// id is a non-negative attribute value; salt is the per-(attribute, pair) salt
// value read from the --salt-file (layout salt[a*P + j]). Computed in uint64 so
// the modular arithmetic matches NumPy's int64 wraparound exactly; the final
// & 0x7FFFFFFF makes arithmetic vs logical shift irrelevant for the kept bits.
inline double hash_phase_cpr(uint32_t value, uint64_t salt) {
    uint64_t h = static_cast<uint64_t>(value) * 2654435761ULL;
    h ^= (salt & 0x7FFFFFFFULL) * 40503ULL;
    h ^= (h >> 15);
    h &= 0x7FFFFFFFULL;
    return (static_cast<double>(h) / 2147483647.0) * 2.0 * kPi;
}

// Disjoint-block categorical rotation. Each attribute a rotates only its own
// block of dimension pairs; out-of-block pairs are left untouched (and every
// pair is covered by exactly one block). A per-attribute value->cos/sin table
// keeps the hot loop to A table lookups plus P rotations per row.
template <typename AttrT>
void rotate_batch_cpr_impl(
    const AprRotationPlan& plan,
    const float* src,
    size_t rows,
    const AttrT* attrs,
    float* dst,
    size_t num_threads
) {
    if (rows == 0) {
        return;
    }
    if (src == nullptr || dst == nullptr || attrs == nullptr) {
        throw std::invalid_argument("CPR rotate_batch buffers must not be null");
    }
    const size_t dim = plan.dim();
    const size_t attr_dim = plan.attr_dim();
    const size_t pair_dim = plan.pair_dim();
    const uint64_t* salt = plan.salt();
    if (salt == nullptr) {
        throw std::invalid_argument("CPR plan has no salt");
    }
    if constexpr (std::is_same_v<AttrT, int32_t>) {
        for (size_t i = 0; i < checked_mul(rows, attr_dim); ++i) {
            if (attrs[i] < 0) {
                throw std::invalid_argument(
                    "CPR query targets must be non-negative; -1 wildcards require a "
                    "multi-probe strategy"
                );
            }
        }
    }

    const std::vector<std::pair<size_t, size_t>> blocks = block_split_cpr(attr_dim, pair_dim);
    std::vector<std::unordered_map<uint32_t, std::vector<double>>> tables(attr_dim);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t a = 0; a < attr_dim; ++a) {
            const uint32_t v = static_cast<uint32_t>(attrs[i * attr_dim + a]);
            auto& tbl = tables[a];
            if (tbl.find(v) != tbl.end()) {
                continue;
            }
            const auto& blk = blocks[a];
            const size_t sz = blk.second - blk.first;
            std::vector<double> cs(sz * 2);
            for (size_t k = 0; k < sz; ++k) {
                const size_t j = blk.first + k;
                const double ph = hash_phase_cpr(v, salt[a * pair_dim + j]);
                cs[2 * k] = std::cos(ph);
                cs[2 * k + 1] = std::sin(ph);
            }
            tbl.emplace(v, std::move(cs));
        }
    }

    run_rows(rows, num_threads, [&](size_t i) {
        for (size_t a = 0; a < attr_dim; ++a) {
            const uint32_t v = static_cast<uint32_t>(attrs[i * attr_dim + a]);
            const auto& cs = tables[a].at(v);
            const auto& blk = blocks[a];
            for (size_t k = 0; k < blk.second - blk.first; ++k) {
                const size_t j = blk.first + k;
                const double c = cs[2 * k];
                const double s = cs[2 * k + 1];
                const double x0 = static_cast<double>(src[i * dim + 2 * j]);
                const double x1 = static_cast<double>(src[i * dim + 2 * j + 1]);
                dst[i * dim + 2 * j] = static_cast<float>(x0 * c - x1 * s);
                dst[i * dim + 2 * j + 1] = static_cast<float>(x0 * s + x1 * c);
            }
        }
    });
}

template <typename AttrT>
void rotate_batch_impl(
    const AprRotationPlan& plan,
    const float* src,
    size_t rows,
    const AttrT* attrs,
    float* dst,
    size_t num_threads
) {
    if (rows == 0) {
        return;
    }
    if (src == nullptr || dst == nullptr || attrs == nullptr) {
        throw std::invalid_argument("APR rotate_batch buffers must not be null");
    }
    const size_t dim = plan.dim();
    const size_t attr_dim = plan.attr_dim();
    const size_t pair_dim = plan.pair_dim();
    const size_t stride = checked_mul(attr_dim, sizeof(AttrT));

    if constexpr (std::is_same_v<AttrT, int32_t>) {
        for (size_t i = 0; i < checked_mul(rows, attr_dim); ++i) {
            if (attrs[i] < 0) {
                throw std::invalid_argument(
                    "APR query targets must be non-negative; -1 wildcards require a "
                    "multi-probe strategy"
                );
            }
        }
    }

    // Build the tuple table once, single-threaded, before entering the parallel
    // region. The hot loop below is then a read-only lookup plus P rotations.
    const char* attr_bytes = reinterpret_cast<const char*>(attrs);
    const size_t entry_doubles = checked_mul(2U, pair_dim);
    const size_t max_doubles = kMaxTableBytes / sizeof(double);
    SlotMap slots;
    std::vector<double> table;
    std::vector<double> total_scratch(pair_dim);
    std::vector<double> entry_scratch(entry_doubles);
    bool use_table = true;

    for (size_t i = 0; i < rows; ++i) {
        const std::string_view key(attr_bytes + i * stride, stride);
        if (slots.find(key) != slots.end()) {
            continue;
        }
        compute_entry(
            plan.salt(), attr_dim, pair_dim, attrs + i * attr_dim,
            total_scratch.data(), entry_scratch.data()
        );
        if (entry_doubles > max_doubles || table.size() > max_doubles - entry_doubles) {
            use_table = false;
            break;
        }
        const size_t slot = table.size() / entry_doubles;
        table.insert(table.end(), entry_scratch.begin(), entry_scratch.end());
        slots.emplace(std::string(key), slot);
    }

    if (!use_table) {
        slots.clear();
        table.clear();
        table.shrink_to_fit();
    }

    const SlotMap& slot_index = slots;
    const std::vector<double>& cos_sin = table;
    run_rows(rows, num_threads, [&](size_t i) {
        if (use_table) {
            const std::string_view key(attr_bytes + i * stride, stride);
            const auto it = slot_index.find(key);
            const double* cs = cos_sin.data() + it->second * entry_doubles;
            rotate_row(src + i * dim, dst + i * dim, pair_dim, cs);
        } else {
            thread_local std::vector<double> local;
            local.assign(checked_mul(3U, pair_dim), 0.0);
            compute_entry(
                plan.salt(), attr_dim, pair_dim, attrs + i * attr_dim,
                local.data(), local.data() + pair_dim
            );
            rotate_row(src + i * dim, dst + i * dim, pair_dim, local.data() + pair_dim);
        }
    });
}

void rotate_batch_hapr_impl(
    const AprRotationPlan& plan,
    const float* src,
    size_t rows,
    const double* values,
    float* dst,
    size_t num_threads
) {
    if (rows == 0) {
        return;
    }
    if (src == nullptr || dst == nullptr || values == nullptr) {
        throw std::invalid_argument("HAPR rotate_batch buffers must not be null");
    }
    const size_t dim = plan.dim();
    const size_t attr_dim = plan.attr_dim();
    const size_t pair_dim = plan.pair_dim();
    const double range_l = plan.range_l();
    const std::vector<AttributeKind>& layout = plan.layout();
    const size_t stride = checked_mul(attr_dim, sizeof(double));

    for (size_t i = 0; i < checked_mul(rows, attr_dim); ++i) {
        if (!std::isfinite(values[i]) || values[i] < 0.0) {
            throw std::invalid_argument(
                "HAPR attribute values must be finite and non-negative"
            );
        }
    }
    for (size_t i = 0; i < rows; ++i) {
        for (size_t a = 0; a < attr_dim; ++a) {
            if (layout[a] != AttributeKind::kEquality) {
                continue;
            }
            const double value = values[i * attr_dim + a];
            if (value != std::floor(value) ||
                value > static_cast<double>(std::numeric_limits<uint32_t>::max())) {
                throw std::invalid_argument(
                    "HAPR equality columns must hold non-negative integer values"
                );
            }
        }
    }

    // Key the tuple table on the raw double row. Equality values are integral
    // and range centres are exact halves, so the byte pattern is canonical.
    const char* value_bytes = reinterpret_cast<const char*>(values);
    const size_t entry_doubles = checked_mul(2U, pair_dim);
    const size_t max_doubles = kMaxTableBytes / sizeof(double);
    SlotMap slots;
    std::vector<double> table;
    std::vector<double> total_scratch(pair_dim);
    std::vector<double> entry_scratch(entry_doubles);
    bool use_table = true;

    for (size_t i = 0; i < rows; ++i) {
        const std::string_view key(value_bytes + i * stride, stride);
        if (slots.find(key) != slots.end()) {
            continue;
        }
        compute_entry_hapr(
            plan.salt(), attr_dim, pair_dim, layout.data(), range_l,
            values + i * attr_dim, total_scratch.data(), entry_scratch.data()
        );
        if (entry_doubles > max_doubles || table.size() > max_doubles - entry_doubles) {
            use_table = false;
            break;
        }
        const size_t slot = table.size() / entry_doubles;
        table.insert(table.end(), entry_scratch.begin(), entry_scratch.end());
        slots.emplace(std::string(key), slot);
    }

    if (!use_table) {
        slots.clear();
        table.clear();
        table.shrink_to_fit();
    }

    const SlotMap& slot_index = slots;
    const std::vector<double>& cos_sin = table;
    run_rows(rows, num_threads, [&](size_t i) {
        if (use_table) {
            const std::string_view key(value_bytes + i * stride, stride);
            const auto it = slot_index.find(key);
            const double* cs = cos_sin.data() + it->second * entry_doubles;
            rotate_row(src + i * dim, dst + i * dim, pair_dim, cs);
        } else {
            thread_local std::vector<double> local;
            local.assign(checked_mul(3U, pair_dim), 0.0);
            compute_entry_hapr(
                plan.salt(), attr_dim, pair_dim, layout.data(), range_l,
                values + i * attr_dim, local.data(), local.data() + pair_dim
            );
            rotate_row(src + i * dim, dst + i * dim, pair_dim, local.data() + pair_dim);
        }
    });
}

}  // namespace

AprRotationPlan::AprRotationPlan(AprConfig config, std::vector<uint64_t> salt)
    : AprRotationPlan(
          config,
          std::move(salt),
          std::vector<AttributeKind>(config.attr_dim, AttributeKind::kEquality),
          false,
          0.0
      ) {}

AprRotationPlan::AprRotationPlan(
    AprConfig config,
    std::vector<uint64_t> salt,
    std::vector<AttributeKind> layout,
    bool has_range,
    double range_l
)
    : config_(config),
      pair_dim_(config.dim / 2),
      salt_(std::move(salt)),
      layout_(std::move(layout)),
      has_range_(has_range),
      range_l_(range_l) {}

AprRotationPlan AprRotationPlan::make(
    const AprConfig& config, const uint64_t* salt, size_t salt_len
) {
    validate_config(config);
    if (config.predicate != PredicateKind::kEquality) {
        throw std::invalid_argument("APR make() requires an all-equality layout");
    }
    if (salt == nullptr) {
        throw std::invalid_argument("APR salt must not be null");
    }
    const size_t expected = checked_mul(config.attr_dim, config.dim / 2);
    if (salt_len != expected) {
        throw std::invalid_argument("APR salt length must equal attr_dim * (dim / 2)");
    }
    return AprRotationPlan(config, std::vector<uint64_t>(salt, salt + salt_len));
}

AprRotationPlan AprRotationPlan::make_standalone(
    const AprConfig& config, uint64_t seed
) {
    validate_config(config);
    if (config.predicate != PredicateKind::kEquality) {
        throw std::invalid_argument(
            "APR make_standalone() requires an all-equality layout"
        );
    }
    const size_t count = checked_mul(config.attr_dim, config.dim / 2);
    std::vector<uint64_t> salt(count);
    uint64_t state = seed;
    for (size_t i = 0; i < count; ++i) {
        salt[i] = splitmix64_next(state) >> 1;
    }
    return AprRotationPlan(config, std::move(salt));
}

AprRotationPlan AprRotationPlan::make_hapr(
    const AprConfig& config,
    const uint64_t* salt,
    size_t salt_len,
    const AttributeKind* layout,
    size_t layout_len,
    double range_l
) {
    validate_config(config);
    if (salt == nullptr) {
        throw std::invalid_argument("APR salt must not be null");
    }
    const size_t expected = checked_mul(config.attr_dim, config.dim / 2);
    if (salt_len != expected) {
        throw std::invalid_argument("APR salt length must equal attr_dim * (dim / 2)");
    }
    const bool has_range = validate_hapr_layout(config, layout, layout_len, range_l);
    return AprRotationPlan(
        config,
        std::vector<uint64_t>(salt, salt + salt_len),
        std::vector<AttributeKind>(layout, layout + layout_len),
        has_range,
        range_l
    );
}

AprRotationPlan AprRotationPlan::make_hapr_standalone(
    const AprConfig& config,
    uint64_t seed,
    const AttributeKind* layout,
    size_t layout_len,
    double range_l
) {
    validate_config(config);
    const bool has_range = validate_hapr_layout(config, layout, layout_len, range_l);
    const size_t count = checked_mul(config.attr_dim, config.dim / 2);
    std::vector<uint64_t> salt(count);
    uint64_t state = seed;
    for (size_t i = 0; i < count; ++i) {
        salt[i] = splitmix64_next(state) >> 1;
    }
    return AprRotationPlan(
        config,
        std::move(salt),
        std::vector<AttributeKind>(layout, layout + layout_len),
        has_range,
        range_l
    );
}

void rotate_vector(
    const AprRotationPlan& plan, const float* src, const uint32_t* attrs, float* dst
) {
    rotate_batch(plan, src, 1, attrs, dst, 0);
}

void rotate_vector(
    const AprRotationPlan& plan, const float* src, const int32_t* attrs, float* dst
) {
    rotate_batch(plan, src, 1, attrs, dst, 0);
}

void rotate_batch(
    const AprRotationPlan& plan,
    const float* src,
    size_t rows,
    const uint32_t* attrs,
    float* dst,
    size_t num_threads
) {
    rotate_batch_impl(plan, src, rows, attrs, dst, num_threads);
}

void rotate_batch(
    const AprRotationPlan& plan,
    const float* src,
    size_t rows,
    const int32_t* attrs,
    float* dst,
    size_t num_threads
) {
    rotate_batch_impl(plan, src, rows, attrs, dst, num_threads);
}

void rotate_batch_hapr(
    const AprRotationPlan& plan,
    const float* src,
    size_t rows,
    const double* values,
    float* dst,
    size_t num_threads
) {
    rotate_batch_hapr_impl(plan, src, rows, values, dst, num_threads);
}

void rotate_vector_hapr(
    const AprRotationPlan& plan, const float* src, const double* values, float* dst
) {
    rotate_batch_hapr(plan, src, 1, values, dst, 0);
}

AprRotationPlan AprRotationPlan::make_cpr(
    const AprConfig& config, const uint64_t* salt, size_t salt_len
) {
    validate_config(config);
    if (config.predicate != PredicateKind::kEquality) {
        throw std::invalid_argument("APR make_cpr() requires an all-equality layout");
    }
    if (salt == nullptr) {
        throw std::invalid_argument("CPR salt must not be null");
    }
    const size_t expected = checked_mul(config.attr_dim, config.dim / 2);
    if (salt_len != expected) {
        throw std::invalid_argument("CPR salt length must equal attr_dim * (dim / 2)");
    }
    return AprRotationPlan(config, std::vector<uint64_t>(salt, salt + salt_len));
}

AprRotationPlan AprRotationPlan::make_cpr_standalone(
    const AprConfig& config, uint64_t seed
) {
    validate_config(config);
    if (config.predicate != PredicateKind::kEquality) {
        throw std::invalid_argument(
            "APR make_cpr_standalone() requires an all-equality layout"
        );
    }
    const size_t count = checked_mul(config.attr_dim, config.dim / 2);
    std::vector<uint64_t> salt(count);
    uint64_t state = seed;
    for (size_t i = 0; i < count; ++i) {
        salt[i] = splitmix64_next(state) & 0x7FFFFFFFULL;
    }
    return AprRotationPlan(config, std::move(salt));
}

void rotate_batch_cpr(
    const AprRotationPlan& plan,
    const float* src,
    size_t rows,
    const uint32_t* attrs,
    float* dst,
    size_t num_threads
) {
    rotate_batch_cpr_impl(plan, src, rows, attrs, dst, num_threads);
}

void rotate_batch_cpr(
    const AprRotationPlan& plan,
    const float* src,
    size_t rows,
    const int32_t* attrs,
    float* dst,
    size_t num_threads
) {
    rotate_batch_cpr_impl(plan, src, rows, attrs, dst, num_threads);
}

void rotate_vector_cpr(
    const AprRotationPlan& plan, const float* src, const uint32_t* attrs, float* dst
) {
    rotate_batch_cpr(plan, src, 1, attrs, dst, 0);
}

void rotate_vector_cpr(
    const AprRotationPlan& plan, const float* src, const int32_t* attrs, float* dst
) {
    rotate_batch_cpr(plan, src, 1, attrs, dst, 0);
}

}  // namespace rabitqlib::attribute
