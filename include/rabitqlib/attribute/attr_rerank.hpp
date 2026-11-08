#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "rabitqlib/utils/space.hpp"

namespace rabitqlib::attribute {

// Default sentinel for graph result buffers (UINT32_MAX). Kept local so this
// header stays independent of index headers; callers may pass their own.
inline constexpr uint32_t kRerankDefaultInvalidId = 0xFFFFFFFFu;

// Rerank one query's candidates by exact L2 squared in the original
// (unrotated) space. candidate_ids holds candidate_count slots, possibly
// padded with invalid_id; ids outside orig_base_rows are ignored. Writes at
// most k results sorted by (distance, id) and pads the remaining slots with
// invalid_id and +inf. Returns the number of valid results written.
inline size_t rerank_by_original_l2(
    const uint32_t* candidate_ids,
    size_t candidate_count,
    const float* orig_base,
    size_t orig_base_rows,
    size_t dim,
    const float* orig_query,
    uint32_t* result_ids,
    float* result_distances,
    size_t k,
    uint32_t invalid_id = kRerankDefaultInvalidId
) {
    std::vector<std::pair<float, uint32_t>> ranked;
    ranked.reserve(candidate_count);
    for (size_t i = 0; i < candidate_count; ++i) {
        const uint32_t id = candidate_ids[i];
        if (id == invalid_id || static_cast<size_t>(id) >= orig_base_rows) {
            continue;
        }
        const float distance = rabitqlib::euclidean_sqr(
            orig_base + static_cast<size_t>(id) * dim, orig_query, dim
        );
        ranked.emplace_back(distance, id);
    }
    std::sort(ranked.begin(), ranked.end());

    const size_t written = std::min(ranked.size(), k);
    for (size_t i = 0; i < written; ++i) {
        result_ids[i] = ranked[i].second;
        result_distances[i] = ranked[i].first;
    }
    std::fill(result_ids + written, result_ids + k, invalid_id);
    std::fill(
        result_distances + written,
        result_distances + k,
        std::numeric_limits<float>::infinity()
    );
    return written;
}

}  // namespace rabitqlib::attribute
