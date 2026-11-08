#include "ivf_search.hpp"

#include <cstddef>
#include <cstdint>

#include "rabitqlib/attribute/attr_filter.hpp"
#include "rabitqlib/index/detail/buffer.hpp"
#include "rabitqlib/index/types.hpp"

namespace rabitqlib::ivf::detail {
// Keep the scalar top-k loop out of the large inlined IVF search body. This avoids
// register spills in wheel builds without changing candidate order or tie handling.
void insert_candidates(
    buffer::SearchBuffer<float>& result_pool,
    const PID* ids,
    const float* distances,
    size_t count
) {
    for (size_t i = 0; i < count; ++i) {
        result_pool.insert(ids[i], distances[i]);
    }
}

void insert_candidates_filtered(
    buffer::SearchBuffer<float>& result_pool,
    const PID* ids,
    const float* distances,
    size_t count,
    const uint32_t* base_attr,
    const int32_t* query_attr,
    size_t attr_dim,
    const rabitqlib::attribute::AttributeKind* layout
) {
    for (size_t i = 0; i < count; ++i) {
        const PID id = ids[i];
        if (!rabitqlib::attribute::predicate_matches(
                base_attr + static_cast<size_t>(id) * attr_dim,
                query_attr,
                attr_dim,
                layout
            )) {
            continue;
        }
        result_pool.insert(id, distances[i]);
    }
}
}  // namespace rabitqlib::ivf::detail
