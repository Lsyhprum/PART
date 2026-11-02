#pragma once

#include <cstdint>

namespace rabitqlib::utils {

// 等值谓词：每个查询属性必须 == 该维度的数据库属性值。
// 某一维查询属性为 -1 表示该维不做约束。
// （语义同 RaBitQ-inline / RaBitQ-post 的 attr_match）
inline bool attr_match(const uint32_t* attr, const int32_t* query_attrs, uint32_t A) {
    for (uint32_t i = 0; i < A; ++i) {
        if (query_attrs[i] != -1 && (int32_t)attr[i] != query_attrs[i])
            return false;
    }
    return true;
}

// 范围谓词：base 属性 attr[a] 必须落在 [lo_a, hi_a] 内（多属性 AND）。
// query_range 为 2*A 个 int32 的 [lo_a, hi_a] 对；某维 lo == -1 表示该维不约束。
// （语义同 RaBitQ-inline / RaBitQ-post 的 attr_match_range）
inline bool attr_match_range(const uint32_t* attr, const int32_t* query_range, uint32_t A) {
    for (uint32_t a = 0; a < A; ++a) {
        int32_t lo = query_range[2 * a];
        int32_t hi = query_range[2 * a + 1];
        if (lo != -1 && !(attr[a] >= (uint32_t)lo && attr[a] <= (uint32_t)hi))
            return false;
    }
    return true;
}

}  // namespace rabitqlib::utils
