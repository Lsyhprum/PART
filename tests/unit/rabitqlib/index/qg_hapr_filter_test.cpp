#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "rabitqlib/attribute/attr_filter.hpp"
#include "rabitqlib/attribute/attr_io.hpp"
#include "rabitqlib/index/symqg/qg.hpp"
#include "rabitqlib/index/symqg/qg_builder.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/io.hpp"

#ifndef RABITQ_TEST_ATTRIBUTE_FIXTURE_ROOT
#define RABITQ_TEST_ATTRIBUTE_FIXTURE_ROOT "tests/fixtures/attribute"
#endif

namespace rabitqlib::symqg {
namespace {

using rabitqlib::attribute::AttributeKind;
using rabitqlib::attribute::AttributeMatrix;
using rabitqlib::attribute::attr_match_mix;
using rabitqlib::attribute::attr_match_range;
using rabitqlib::attribute::predicate_matches;
using rabitqlib::attribute::predicate_row_valid;
using rabitqlib::attribute::query_stride;
using rabitqlib::attribute::read_attr_bin;

constexpr size_t kDim = 64;
constexpr size_t kNumBase = 256;
constexpr size_t kDegree = 32;
constexpr uint32_t kTopK = 10;
constexpr size_t kEfConstruction = 64;
constexpr size_t kEfSearch = 128;

constexpr size_t kMixedAttrDim = 2;
constexpr size_t kMixedQStride = 3;
constexpr size_t kRangeAttrDim = 1;
constexpr size_t kRangeQStride = 2;

const AttributeKind kMixedLayout[2] = {
    AttributeKind::kEquality, AttributeKind::kRange
};
const AttributeKind kRangeLayout[1] = {AttributeKind::kRange};
const AttributeKind kAllEqLayout[2] = {
    AttributeKind::kEquality, AttributeKind::kEquality
};

std::string fixture(const char* name) {
    return std::string(RABITQ_TEST_ATTRIBUTE_FIXTURE_ROOT) + "/" + name;
}

// GT is top-K by rotated-domain L2 inside the predicate-matching subset. HAPR
// rotation is an isometry per attribute value, so the rotated domain is the
// metric the index actually optimises; padding uses kPidMax.
struct Case {
    AttributeMatrix<float> base;
    AttributeMatrix<float> query;
    AttributeMatrix<uint32_t> gt;
    AttributeMatrix<uint32_t> base_attr;
    AttributeMatrix<int32_t> query_attr;
    std::unique_ptr<QuantizedGraph<float>> graph;
};

Case load_case(
    const char* base_name,
    const char* query_name,
    const char* gt_name,
    const char* base_attr_name,
    const char* query_attr_name,
    size_t attr_dim,
    size_t qstride
) {
    Case c;
    load_vecs<float, AttributeMatrix<float>>(fixture(base_name).c_str(), c.base);
    load_vecs<float, AttributeMatrix<float>>(fixture(query_name).c_str(), c.query);
    load_vecs<uint32_t, AttributeMatrix<uint32_t>>(fixture(gt_name).c_str(), c.gt);
    c.base_attr = read_attr_bin<uint32_t>(fixture(base_attr_name).c_str());
    c.query_attr = read_attr_bin<int32_t>(fixture(query_attr_name).c_str());
    if (c.base.rows != kNumBase || c.base.cols != kDim ||
        c.query.cols != kDim || c.gt.rows != c.query.rows ||
        c.gt.cols != kTopK || c.base_attr.rows != kNumBase ||
        c.base_attr.cols != attr_dim || c.query_attr.rows != c.query.rows ||
        c.query_attr.cols != qstride) {
        throw std::runtime_error("QG HAPR fixture has an unexpected shape");
    }
    c.graph = std::make_unique<QuantizedGraph<float>>(
        kNumBase, kDim, kDegree, METRIC_L2, RotatorType::FhtKacRotator, 0, 42
    );
    QGBuilder builder(
        *c.graph, kEfConstruction, c.base.data(), 1, QGInitialization::PiPNN, 42
    );
    builder.build();
    c.graph->set_ef(kEfSearch);
    return c;
}

Case& mixed_case() {
    static Case instance = load_case(
        "hapr_base.fvecs", "hapr_query.fvecs", "hapr_mixed_gt.ivecs",
        "hapr_base_attr.bin", "hapr_query_attr.bin",
        kMixedAttrDim, kMixedQStride
    );
    return instance;
}

Case& range_case() {
    static Case instance = load_case(
        "hapr_range_base.fvecs", "hapr_range_query.fvecs",
        "hapr_range_gt.ivecs", "hapr_range_base_attr.bin",
        "hapr_range_query_attr.bin", kRangeAttrDim, kRangeQStride
    );
    return instance;
}

size_t count_violations(
    const Case& c,
    size_t z,
    const PID* ids,
    size_t count,
    const AttributeKind* layout,
    size_t attr_dim,
    size_t qstride
) {
    size_t violations = 0;
    for (size_t i = 0; i < count; ++i) {
        if (ids[i] == kPidMax) {
            continue;
        }
        if (!attr_match_mix(
                c.base_attr.data() + static_cast<size_t>(ids[i]) * attr_dim,
                c.query_attr.data() + z * qstride,
                attr_dim,
                layout)) {
            ++violations;
        }
    }
    return violations;
}

size_t recall_hits(const Case& c, size_t z, const PID* ids, size_t count) {
    size_t hits = 0;
    for (size_t i = 0; i < count; ++i) {
        if (ids[i] == kPidMax) {
            continue;
        }
        for (size_t k = 0; k < c.gt.cols; ++k) {
            if (c.gt(z, k) == ids[i]) {
                ++hits;
                break;
            }
        }
    }
    return hits;
}

size_t gt_count(const Case& c) {
    size_t total = 0;
    for (size_t z = 0; z < c.gt.rows; ++z) {
        for (size_t k = 0; k < c.gt.cols; ++k) {
            if (c.gt(z, k) != kPidMax) {
                ++total;
            }
        }
    }
    return total;
}

void search_one(
    Case& c,
    size_t z,
    const AttributeKind* layout,
    size_t attr_dim,
    size_t qstride,
    PID* ids,
    float* dists
) {
    c.graph->search(
        c.query.data() + z * kDim,
        kTopK,
        ids,
        dists,
        c.base_attr.data(),
        c.query_attr.data() + z * qstride,
        attr_dim,
        layout
    );
}

TEST(QgHaprFilter, SharedPredicateHelpers) {
    EXPECT_EQ(query_stride(nullptr, 2), 2U);
    EXPECT_EQ(query_stride(kMixedLayout, 2), 3U);
    EXPECT_EQ(query_stride(kRangeLayout, 1), 2U);
    EXPECT_EQ(query_stride(kAllEqLayout, 2), 2U);

    const uint32_t attr[2] = {3, 20};
    const int32_t eq_range[3] = {3, 15, 25};
    EXPECT_TRUE(attr_match_mix(attr, eq_range, 2, kMixedLayout));
    EXPECT_TRUE(predicate_matches(attr, eq_range, 2, kMixedLayout));
    const int32_t wrong_eq[3] = {4, 15, 25};
    EXPECT_FALSE(attr_match_mix(attr, wrong_eq, 2, kMixedLayout));
    const int32_t outside[3] = {3, 21, 30};
    EXPECT_FALSE(attr_match_mix(attr, outside, 2, kMixedLayout));

    const int32_t range[2] = {15, 25};
    EXPECT_TRUE(attr_match_range(attr + 1, range, 1));
    const int32_t wildcard[2] = {-1, 0};
    EXPECT_TRUE(attr_match_range(attr + 1, wildcard, 1));
    const int32_t range_outside[2] = {21, 30};
    EXPECT_FALSE(attr_match_range(attr + 1, range_outside, 1));

    EXPECT_TRUE(predicate_row_valid(eq_range, 2, kMixedLayout));
    const int32_t inverted[3] = {3, 25, 15};
    EXPECT_FALSE(predicate_row_valid(inverted, 2, kMixedLayout));
    const int32_t negative[3] = {3, -1, 15};
    EXPECT_FALSE(predicate_row_valid(negative, 2, kMixedLayout));

    const AttributeKind bad[1] = {static_cast<AttributeKind>(7)};
    EXPECT_THROW(query_stride(bad, 1), std::invalid_argument);
}

void expect_sound(
    Case& c, const AttributeKind* layout, size_t attr_dim, size_t qstride
) {
    size_t violations = 0;
    size_t returned = 0;
    for (size_t z = 0; z < c.query.rows; ++z) {
        std::vector<PID> ids(kTopK);
        std::vector<float> dists(kTopK);
        search_one(c, z, layout, attr_dim, qstride, ids.data(), dists.data());
        for (size_t j = 0; j < kTopK; ++j) {
            if (ids[j] == kPidMax) {
                EXPECT_TRUE(std::isinf(dists[j]));
                continue;
            }
            ++returned;
            ASSERT_LT(ids[j], c.base.rows);
        }
        violations +=
            count_violations(c, z, ids.data(), kTopK, layout, attr_dim, qstride);
    }
    EXPECT_EQ(violations, 0U);
    EXPECT_GT(returned, 0U);
}

TEST(QgHaprFilter, MixedFilteredSearchIsSound) {
    expect_sound(mixed_case(), kMixedLayout, kMixedAttrDim, kMixedQStride);
}

TEST(QgHaprFilter, RangeFilteredSearchIsSound) {
    expect_sound(range_case(), kRangeLayout, kRangeAttrDim, kRangeQStride);
}

TEST(QgHaprFilter, UnfilteredSearchReturnsMismatches) {
    auto& c = mixed_case();
    size_t mismatches = 0;
    for (size_t z = 0; z < c.query.rows; ++z) {
        std::vector<PID> ids(kTopK);
        std::vector<float> dists(kTopK);
        c.graph->search(
            c.query.data() + z * kDim, kTopK, ids.data(), dists.data()
        );
        mismatches += count_violations(
            c, z, ids.data(), kTopK, kMixedLayout, kMixedAttrDim, kMixedQStride
        );
    }
    EXPECT_GT(mismatches, 0U);
}

void expect_batch_matches_single(
    Case& c, const AttributeKind* layout, size_t attr_dim, size_t qstride
) {
    const size_t nq = c.query.rows;
    std::vector<PID> batch_ids(nq * kTopK);
    std::vector<PID> single_ids(nq * kTopK);
    std::vector<float> batch_dists(nq * kTopK);
    std::vector<float> single_dists(nq * kTopK);
    c.graph->search_batch(
        c.query.data(), nq, kTopK, batch_ids.data(), batch_dists.data(),
        c.base_attr.data(), c.query_attr.data(), attr_dim, layout, 1
    );
    for (size_t z = 0; z < nq; ++z) {
        search_one(
            c, z, layout, attr_dim, qstride,
            single_ids.data() + z * kTopK, single_dists.data() + z * kTopK
        );
    }
    EXPECT_EQ(batch_ids, single_ids);
    for (size_t i = 0; i < batch_dists.size(); ++i) {
        EXPECT_FLOAT_EQ(batch_dists[i], single_dists[i]);
    }
}

TEST(QgHaprFilter, BatchMatchesSingleQuery) {
    expect_batch_matches_single(
        mixed_case(), kMixedLayout, kMixedAttrDim, kMixedQStride
    );
    expect_batch_matches_single(
        range_case(), kRangeLayout, kRangeAttrDim, kRangeQStride
    );
}

void expect_parallel_matches_serial(
    Case& c, const AttributeKind* layout, size_t attr_dim
) {
    const size_t nq = c.query.rows;
    std::vector<PID> serial_ids(nq * kTopK);
    std::vector<PID> parallel_ids(nq * kTopK);
    std::vector<float> serial_dists(nq * kTopK);
    std::vector<float> parallel_dists(nq * kTopK);
    c.graph->search_batch(
        c.query.data(), nq, kTopK, serial_ids.data(), serial_dists.data(),
        c.base_attr.data(), c.query_attr.data(), attr_dim, layout, 1
    );
    c.graph->search_batch(
        c.query.data(), nq, kTopK, parallel_ids.data(), parallel_dists.data(),
        c.base_attr.data(), c.query_attr.data(), attr_dim, layout, 2
    );
    EXPECT_EQ(parallel_ids, serial_ids);
    for (size_t i = 0; i < parallel_dists.size(); ++i) {
        EXPECT_FLOAT_EQ(parallel_dists[i], serial_dists[i]);
    }
}

TEST(QgHaprFilter, ParallelBatchMatchesSerial) {
    expect_parallel_matches_serial(mixed_case(), kMixedLayout, kMixedAttrDim);
    expect_parallel_matches_serial(range_case(), kRangeLayout, kRangeAttrDim);
}

void expect_recall(
    Case& c, const AttributeKind* layout, size_t attr_dim, size_t qstride
) {
    size_t hits = 0;
    for (size_t z = 0; z < c.query.rows; ++z) {
        std::vector<PID> ids(kTopK);
        std::vector<float> dists(kTopK);
        search_one(c, z, layout, attr_dim, qstride, ids.data(), dists.data());
        hits += recall_hits(c, z, ids.data(), kTopK);
    }
    EXPECT_GT(hits, 0U);
    EXPECT_LE(hits, gt_count(c));
}

TEST(QgHaprFilter, FilteredRecallAgainstGroundTruth) {
    expect_recall(
        mixed_case(), kMixedLayout, kMixedAttrDim, kMixedQStride
    );
    expect_recall(
        range_case(), kRangeLayout, kRangeAttrDim, kRangeQStride
    );
}

TEST(QgHaprFilter, AllEqualityLayoutMatchesLegacyGate) {
    auto& c = mixed_case();
    const size_t nq = c.query.rows;
    std::vector<int32_t> eq_targets(nq * 2);
    for (size_t z = 0; z < nq; ++z) {
        eq_targets[z * 2] = c.query_attr(z, 0);
        eq_targets[z * 2 + 1] = c.query_attr(z, 1);
    }
    for (size_t z = 0; z < nq; ++z) {
        std::vector<PID> legacy_ids(kTopK);
        std::vector<PID> layout_ids(kTopK);
        std::vector<float> legacy_dists(kTopK);
        std::vector<float> layout_dists(kTopK);
        c.graph->search(
            c.query.data() + z * kDim, kTopK, legacy_ids.data(),
            legacy_dists.data(), c.base_attr.data(), eq_targets.data() + z * 2,
            kMixedAttrDim
        );
        c.graph->search(
            c.query.data() + z * kDim, kTopK, layout_ids.data(),
            layout_dists.data(), c.base_attr.data(), eq_targets.data() + z * 2,
            kMixedAttrDim, kAllEqLayout
        );
        EXPECT_EQ(layout_ids, legacy_ids);
        for (size_t j = 0; j < kTopK; ++j) {
            EXPECT_FLOAT_EQ(layout_dists[j], legacy_dists[j]);
        }
    }
}

TEST(QgHaprFilter, RejectsInvalidPredicates) {
    auto& c = mixed_case();
    std::vector<PID> ids(kTopK);
    std::vector<float> dists(kTopK);
    const float* q = c.query.data();
    const uint32_t* ba = c.base_attr.data();

    EXPECT_THROW(
        c.graph->search(
            q, kTopK, ids.data(), dists.data(), ba, c.query_attr.data(),
            kMixedAttrDim, nullptr
        ),
        std::invalid_argument
    );

    const int32_t negative[3] = {0, -1, 10};
    EXPECT_THROW(
        c.graph->search(
            q, kTopK, ids.data(), dists.data(), ba, negative,
            kMixedAttrDim, kMixedLayout
        ),
        std::invalid_argument
    );

    const int32_t inverted[3] = {0, 20, 10};
    EXPECT_THROW(
        c.graph->search(
            q, kTopK, ids.data(), dists.data(), ba, inverted,
            kMixedAttrDim, kMixedLayout
        ),
        std::invalid_argument
    );

    const AttributeKind bad[2] = {
        AttributeKind::kEquality, static_cast<AttributeKind>(9)
    };
    EXPECT_THROW(
        c.graph->search(
            q, kTopK, ids.data(), dists.data(), ba, c.query_attr.data(),
            kMixedAttrDim, bad
        ),
        std::invalid_argument
    );

    const size_t nq = c.query.rows;
    std::vector<PID> batch_ids(nq * kTopK);
    std::vector<float> batch_dists(nq * kTopK);
    EXPECT_THROW(
        c.graph->search_batch(
            c.query.data(), nq, kTopK, batch_ids.data(), batch_dists.data(),
            ba, c.query_attr.data(), kMixedAttrDim, nullptr, 1
        ),
        std::invalid_argument
    );

    std::vector<int32_t> batch_bad(nq * kMixedQStride, 0);
    batch_bad[kMixedQStride + 1] = -1;
    EXPECT_THROW(
        c.graph->search_batch(
            c.query.data(), nq, kTopK, batch_ids.data(), batch_dists.data(),
            ba, batch_bad.data(), kMixedAttrDim, kMixedLayout, 1
        ),
        std::invalid_argument
    );
}

}  // namespace
}  // namespace rabitqlib::symqg
