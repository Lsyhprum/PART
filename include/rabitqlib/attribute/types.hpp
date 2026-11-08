#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace rabitqlib::attribute {

// kEquality is the baseline all-hash-phase layout. kRange marks a layout that
// contains at least one numeric range dimension; a layout mixing equality and
// range dimensions also uses kRange. The per-attribute roles live in
// AttributeKind / the plan's layout, not in this coarse enum.
enum class PredicateKind : uint8_t { kEquality = 0, kRange = 1 };

// A single rotated probe cannot represent "match any attribute value". Queries
// containing -1 are rejected until a multi-probe strategy is implemented.
enum class WildcardPolicy : uint8_t { kReject = 0 };

// Role of one attribute column inside a rotation layout. Equality columns use
// the salted hash phase; range columns use a deterministic scaled phase.
enum class AttributeKind : uint8_t { kEquality = 0, kRange = 1 };

// Configuration for the index-agnostic attribute phase rotation (APR).
// dim is the original vector dimension, not the padded dimension used by
// quantizers; it must be even because rotation acts on pairs of coordinates.
// range_l is the numeric attribute value range used by the scaled phase;
// 0 means "no range dimension" or "to be inferred by the caller". New fields
// have defaults so existing AprConfig{dim, attr_dim} initializations still work.
struct AprConfig {
    size_t dim = 0;
    size_t attr_dim = 0;
    PredicateKind predicate = PredicateKind::kEquality;
    WildcardPolicy wildcard = WildcardPolicy::kReject;
    double range_l = 0.0;
};

// Width of one expanded query-attribute row. Equality columns occupy a single
// int32 target, range columns occupy a [lo, hi] pair. A null layout denotes an
// all-equality layout, whose stride is exactly attr_dim. This is the single
// source of truth for the QG / IVF gates, the query CLIs and the fixtures.
inline size_t query_stride(const AttributeKind* layout, size_t attr_dim) {
    if (layout == nullptr) {
        return attr_dim;
    }
    size_t stride = 0;
    for (size_t a = 0; a < attr_dim; ++a) {
        size_t width = 1;
        if (layout[a] == AttributeKind::kRange) {
            width = 2;
        } else if (layout[a] != AttributeKind::kEquality) {
            throw std::invalid_argument("Unknown attribute kind in layout");
        }
        if (stride > std::numeric_limits<size_t>::max() - width) {
            throw std::length_error("Attribute query stride is too large");
        }
        stride += width;
    }
    return stride;
}

}  // namespace rabitqlib::attribute
