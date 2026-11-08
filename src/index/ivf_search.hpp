#pragma once

#include <cstddef>
#include <cstdint>

#include "rabitqlib/attribute/types.hpp"
#include "rabitqlib/index/detail/buffer.hpp"
#include "rabitqlib/index/types.hpp"

namespace rabitqlib::ivf::detail {
void insert_candidates(
    buffer::SearchBuffer<float>& result_pool,
    const PID* ids,
    const float* distances,
    size_t count
);

// Predicate-gated variant. base_attr is a row-major [num_points, attr_dim]
// table and query_attr points at one expanded row of query_stride(layout,
// attr_dim) int32 values. A null layout selects the all-equality predicate.
// Only matching points enter the result pool; mismatches still take part in
// distance estimation upstream, exactly as in the unfiltered path.
void insert_candidates_filtered(
    buffer::SearchBuffer<float>& result_pool,
    const PID* ids,
    const float* distances,
    size_t count,
    const uint32_t* base_attr,
    const int32_t* query_attr,
    size_t attr_dim,
    const rabitqlib::attribute::AttributeKind* layout
);
}  // namespace rabitqlib::ivf::detail
