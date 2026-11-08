#pragma once

#include <cstddef>
#include <cstdint>

#include "rabitqlib/attribute/types.hpp"

namespace rabitqlib::attribute {

// Equality predicate over attr_dim attributes. A query value of -1 is a
// wildcard for that dimension. This helper is independent of any index type so
// that future IVF / SymphonyQG filtering gates can share the same semantics.
inline bool attr_match(
    const uint32_t* attr, const int32_t* query_attrs, size_t attr_dim
) {
    for (size_t i = 0; i < attr_dim; ++i) {
        if (query_attrs[i] != -1 &&
            static_cast<int64_t>(attr[i]) != static_cast<int64_t>(query_attrs[i])) {
            return false;
        }
    }
    return true;
}

// Range predicate over attr_dim attributes. query_range holds attr_dim
// [lo, hi] pairs (2 * attr_dim int32 values). A dimension is unconstrained
// when lo == -1; otherwise the base value must lie inside [lo, hi]. All
// dimensions are combined with AND.
inline bool attr_match_range(
    const uint32_t* attr, const int32_t* query_range, size_t attr_dim
) {
    for (size_t a = 0; a < attr_dim; ++a) {
        const int32_t lo = query_range[2 * a];
        const int32_t hi = query_range[2 * a + 1];
        if (lo != -1 &&
            (static_cast<int64_t>(attr[a]) < static_cast<int64_t>(lo) ||
             static_cast<int64_t>(attr[a]) > static_cast<int64_t>(hi))) {
            return false;
        }
    }
    return true;
}

// Mixed predicate over attr_dim attributes. layout declares the role of each
// column; a null layout means all columns are equality. query is one expanded
// row of query_stride(layout, attr_dim) values: an equality column occupies one
// target, a range column occupies a [lo, hi] pair. All dimensions are combined
// with AND.
inline bool attr_match_mix(
    const uint32_t* attr,
    const int32_t* query,
    size_t attr_dim,
    const AttributeKind* layout
) {
    size_t col = 0;
    for (size_t a = 0; a < attr_dim; ++a) {
        if (layout != nullptr && layout[a] == AttributeKind::kRange) {
            const int32_t lo = query[col];
            const int32_t hi = query[col + 1];
            col += 2;
            if (lo != -1 &&
                (static_cast<int64_t>(attr[a]) < static_cast<int64_t>(lo) ||
                 static_cast<int64_t>(attr[a]) > static_cast<int64_t>(hi))) {
                return false;
            }
        } else {
            const int32_t target = query[col];
            ++col;
            if (target != -1 &&
                static_cast<int64_t>(attr[a]) != static_cast<int64_t>(target)) {
                return false;
            }
        }
    }
    return true;
}

// Dispatch one predicate decision for a gate or an external verifier. A null
// layout selects the original all-equality helper, so callers never need their
// own copy of the branching rule.
inline bool predicate_matches(
    const uint32_t* attr,
    const int32_t* query,
    size_t attr_dim,
    const AttributeKind* layout
) {
    return layout == nullptr ? attr_match(attr, query, attr_dim)
                             : attr_match_mix(attr, query, attr_dim, layout);
}

// Validate one expanded query-attribute row at a gate entry point. Gate
// callers reject every negative value (including -1 wildcards) and require
// 0 <= lo <= hi for range columns: the offline HAPR encoder can only rotate a
// single concrete probe, so exposing a wildcard would disagree with the soft
// filter. The match helpers above keep wildcard semantics for standalone use.
inline bool predicate_row_valid(
    const int32_t* query, size_t attr_dim, const AttributeKind* layout
) {
    size_t col = 0;
    for (size_t a = 0; a < attr_dim; ++a) {
        if (layout != nullptr && layout[a] == AttributeKind::kRange) {
            const int32_t lo = query[col];
            const int32_t hi = query[col + 1];
            col += 2;
            if (lo < 0 || hi < 0 || lo > hi) {
                return false;
            }
        } else {
            if (query[col] < 0) {
                return false;
            }
            ++col;
        }
    }
    return true;
}

}  // namespace rabitqlib::attribute
