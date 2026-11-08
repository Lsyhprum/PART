#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "rabitqlib/attribute/attr_filter.hpp"
#include "rabitqlib/attribute/attr_io.hpp"
#include "rabitqlib/attribute/attr_rerank.hpp"
#include "rabitqlib/index/symqg/qg.hpp"
#include "rabitqlib/index/symqg/qg_builder.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/io.hpp"

namespace rabitqlib::symqg {
namespace {

using rabitqlib::attribute::AttributeMatrix;
using rabitqlib::attribute::attr_match;
using rabitqlib::attribute::read_attr_bin;

constexpr size_t kDim = 64;
constexpr size_t kNumBase = 256;
constexpr size_t kAttrDim = 1;
constexpr size_t kDegree = 32;
constexpr uint32_t kTopK = 10;
constexpr uint32_t kEfConstruction = 64;
constexpr size_t kEfSearch = 128;

std::string fixture(const char* name) {
    return std::string(RABITQ_TEST_ATTRIBUTE_FIXTURE_ROOT) + "/" + name;
}

// One shared, in-memory graph built from the fixed rotated fixture. No index
// file is written and no temporary data is generated at test runtime.
struct Fixture {
    AttributeMatrix<float> base;
    AttributeMatrix<float> query;
    AttributeMatrix<uint32_t> gt;
    AttributeMatrix<uint32_t> base_attr;
    AttributeMatrix<int32_t> query_attr;
    QuantizedGraph<float> graph;

    Fixture()
        : graph(
              kNumBase, kDim, kDegree, METRIC_L2, RotatorType::FhtKacRotator, 0, 42
          ) {
        load_vecs<float, AttributeMatrix<float>>(
            fixture("qg_base.fvecs").c_str(), base
        );
        load_vecs<float, AttributeMatrix<float>>(
            fixture("qg_query.fvecs").c_str(), query
        );
        load_vecs<uint32_t, AttributeMatrix<uint32_t>>(
            fixture("qg_filtered_gt.ivecs").c_str(), gt
        );
        base_attr = read_attr_bin<uint32_t>(fixture("qg_base_attr.bin").c_str());
        query_attr = read_attr_bin<int32_t>(fixture("qg_query_attr.bin").c_str());
        if (base.rows != kNumBase || base.cols != kDim || query.cols != kDim ||
            base_attr.rows != kNumBase || base_attr.cols != kAttrDim ||
            query_attr.rows != query.rows || query_attr.cols != kAttrDim ||
            gt.rows != query.rows || gt.cols != kTopK) {
            throw std::runtime_error("QG attribute fixture has an unexpected shape");
        }
        QGBuilder builder(
            graph, kEfConstruction, base.data(), 1, QGInitialization::PiPNN, 42
        );
        builder.build();
        graph.set_ef(kEfSearch);
    }
};

Fixture& shared_fixture() {
    static Fixture instance;
    instance.graph.set_ef(kEfSearch);
    return instance;
}

void search_filtered(Fixture& data, size_t z, PID* ids, float* dists) {
    data.graph.search(
        data.query.data() + z * kDim,
        kTopK,
        ids,
        dists,
        data.base_attr.data(),
        data.query_attr.data() + z * kAttrDim,
        kAttrDim
    );
}

TEST(QgAttributeFilter, FilteredSearchIsSound) {
    auto& data = shared_fixture();
    size_t violations = 0;
    size_t returned = 0;
    for (size_t z = 0; z < data.query.rows; ++z) {
        std::vector<PID> ids(kTopK);
        std::vector<float> dists(kTopK);
        search_filtered(data, z, ids.data(), dists.data());
        for (size_t j = 0; j < kTopK; ++j) {
            if (ids[j] == kPidMax) {
                EXPECT_TRUE(std::isinf(dists[j]));
                continue;
            }
            ++returned;
            ASSERT_LT(ids[j], data.base.rows);
            if (!attr_match(
                    data.base_attr.data() + static_cast<size_t>(ids[j]) * kAttrDim,
                    data.query_attr.data() + z * kAttrDim,
                    kAttrDim)) {
                ++violations;
            }
        }
    }
    EXPECT_EQ(violations, 0U);
    EXPECT_GT(returned, 0U);
}

TEST(QgAttributeFilter, UnfilteredSearchReturnsMismatches) {
    auto& data = shared_fixture();
    size_t mismatches = 0;
    for (size_t z = 0; z < data.query.rows; ++z) {
        std::vector<PID> ids(kTopK);
        std::vector<float> dists(kTopK);
        data.graph.search(
            data.query.data() + z * kDim, kTopK, ids.data(), dists.data()
        );
        for (const PID id : ids) {
            if (id == kPidMax) {
                continue;
            }
            if (!attr_match(
                    data.base_attr.data() + static_cast<size_t>(id) * kAttrDim,
                    data.query_attr.data() + z * kAttrDim,
                    kAttrDim)) {
                ++mismatches;
            }
        }
    }
    EXPECT_GT(mismatches, 0U);
}

TEST(QgAttributeFilter, BatchMatchesSingleQuery) {
    auto& data = shared_fixture();
    const size_t nq = data.query.rows;
    std::vector<PID> batch_ids(nq * kTopK);
    std::vector<PID> single_ids(nq * kTopK);
    std::vector<float> batch_dists(nq * kTopK);
    std::vector<float> single_dists(nq * kTopK);
    data.graph.search_batch(
        data.query.data(),
        nq,
        kTopK,
        batch_ids.data(),
        batch_dists.data(),
        data.base_attr.data(),
        data.query_attr.data(),
        kAttrDim,
        1
    );
    for (size_t z = 0; z < nq; ++z) {
        search_filtered(
            data, z, single_ids.data() + z * kTopK,
            single_dists.data() + z * kTopK
        );
    }
    EXPECT_EQ(batch_ids, single_ids);
    for (size_t i = 0; i < batch_dists.size(); ++i) {
        EXPECT_FLOAT_EQ(batch_dists[i], single_dists[i]);
    }
}

TEST(QgAttributeFilter, FilteredRecallAgainstGroundTruth) {
    auto& data = shared_fixture();
    size_t correct = 0;
    for (size_t z = 0; z < data.query.rows; ++z) {
        std::vector<PID> ids(kTopK);
        std::vector<float> dists(kTopK);
        search_filtered(data, z, ids.data(), dists.data());
        for (const PID id : ids) {
            if (id == kPidMax) {
                continue;
            }
            for (size_t k = 0; k < data.gt.cols; ++k) {
                if (data.gt(z, k) == id) {
                    ++correct;
                    break;
                }
            }
        }
    }
    const double recall = static_cast<double>(correct) /
                          static_cast<double>(data.query.rows * kTopK);
    EXPECT_GT(recall, 0.0);
}

TEST(QgAttributeFilter, RejectsInvalidGateInputs) {
    auto& data = shared_fixture();
    std::vector<PID> ids(kTopK);
    std::vector<float> dists(kTopK);
    const float* q = data.query.data();
    const uint32_t* ba = data.base_attr.data();
    const int32_t* qa = data.query_attr.data();

    EXPECT_THROW(
        data.graph.search(q, kTopK, ids.data(), dists.data(), nullptr, qa, kAttrDim),
        std::invalid_argument
    );
    EXPECT_THROW(
        data.graph.search(q, kTopK, ids.data(), dists.data(), ba, nullptr, kAttrDim),
        std::invalid_argument
    );
    EXPECT_THROW(
        data.graph.search(q, kTopK, ids.data(), dists.data(), ba, qa, 0),
        std::invalid_argument
    );

    std::vector<int32_t> negative(kAttrDim, -1);
    EXPECT_THROW(
        data.graph.search(
            q, kTopK, ids.data(), dists.data(), ba, negative.data(), kAttrDim
        ),
        std::invalid_argument
    );

    const size_t nq = data.query.rows;
    std::vector<PID> batch_ids(nq * kTopK);
    std::vector<float> batch_dists(nq * kTopK);
    std::vector<int32_t> batch_bad(nq * kAttrDim, 0);
    batch_bad[3] = -1;
    EXPECT_THROW(
        data.graph.search_batch(
            data.query.data(), nq, kTopK, batch_ids.data(), batch_dists.data(),
            ba, batch_bad.data(), kAttrDim, 1
        ),
        std::invalid_argument
    );
}

TEST(QgAttributeFilter, GoldenRotationPreservesRowNorms) {
    AttributeMatrix<float> raw;
    AttributeMatrix<float> rotated;
    load_vecs<float, AttributeMatrix<float>>(fixture("base.fvecs").c_str(), raw);
    load_vecs<float, AttributeMatrix<float>>(
        fixture("apr_base.fvecs").c_str(), rotated
    );
    ASSERT_EQ(raw.rows, rotated.rows);
    ASSERT_EQ(raw.cols, rotated.cols);
    for (size_t i = 0; i < raw.rows; ++i) {
        double before = 0.0;
        double after = 0.0;
        for (size_t j = 0; j < raw.cols; ++j) {
            before += static_cast<double>(raw(i, j)) * raw(i, j);
            after += static_cast<double>(rotated(i, j)) * rotated(i, j);
        }
        EXPECT_NEAR(after, before, 1e-4 * std::max(1.0, before));
    }
}

TEST(QgAttributeFilter, RerankHelperOrdersPadsAndBreaksTies) {
    constexpr size_t kSmallDim = 3;
    const std::vector<float> base = {
        0.0F, 0.0F, 0.0F,
        3.0F, 0.0F, 0.0F,
        0.0F, 4.0F, 0.0F,
        1.0F, 1.0F, 1.0F,
        2.0F, 2.0F, 2.0F,
    };
    const std::vector<float> query = {0.0F, 0.0F, 0.0F};
    const size_t rows = base.size() / kSmallDim;
    std::vector<uint32_t> candidates = {kPidMax, 4, 1, kPidMax, 2, 0, 3};
    constexpr size_t kOut = 4;
    std::vector<uint32_t> ids(kOut, 0);
    std::vector<float> dists(kOut, 0.0F);
    const size_t written = rabitqlib::attribute::rerank_by_original_l2(
        candidates.data(), candidates.size(), base.data(), rows, kSmallDim,
        query.data(), ids.data(), dists.data(), kOut, kPidMax
    );
    EXPECT_EQ(written, kOut);
    EXPECT_EQ(ids[0], 0U);
    EXPECT_EQ(ids[1], 3U);
    EXPECT_EQ(ids[2], 1U);
    EXPECT_EQ(ids[3], 4U);
    for (size_t i = 0; i < kOut; ++i) {
        double expected = 0.0;
        for (size_t j = 0; j < kSmallDim; ++j) {
            const double diff =
                static_cast<double>(base[ids[i] * kSmallDim + j]) - query[j];
            expected += diff * diff;
        }
        EXPECT_NEAR(dists[i], expected, 1e-5);
        if (i > 0) {
            EXPECT_LE(dists[i - 1], dists[i]);
        }
    }

    // Short candidate lists are padded; out-of-range ids are ignored.
    std::vector<uint32_t> short_candidates = {
        0, kPidMax, static_cast<uint32_t>(rows + 100)
    };
    std::vector<uint32_t> short_ids(kOut, 0);
    std::vector<float> short_dists(kOut, 0.0F);
    const size_t short_written = rabitqlib::attribute::rerank_by_original_l2(
        short_candidates.data(), short_candidates.size(), base.data(), rows,
        kSmallDim, query.data(), short_ids.data(), short_dists.data(), kOut,
        kPidMax
    );
    EXPECT_EQ(short_written, 1U);
    EXPECT_EQ(short_ids[0], 0U);
    for (size_t i = 1; i < kOut; ++i) {
        EXPECT_EQ(short_ids[i], kPidMax);
        EXPECT_TRUE(std::isinf(short_dists[i]));
    }

    // Equal distances are ordered by id, so repeated runs are deterministic.
    const std::vector<float> tied_base = {
        1.0F, 0.0F, 0.0F, -1.0F, 0.0F, 0.0F
    };
    std::vector<uint32_t> tied_candidates = {1, 0};
    std::vector<uint32_t> tied_ids(2, kPidMax);
    std::vector<float> tied_dists(2, 0.0F);
    rabitqlib::attribute::rerank_by_original_l2(
        tied_candidates.data(), tied_candidates.size(), tied_base.data(), 2,
        kSmallDim, query.data(), tied_ids.data(), tied_dists.data(), 2, kPidMax
    );
    EXPECT_EQ(tied_ids[0], 0U);
    EXPECT_EQ(tied_ids[1], 1U);
}

TEST(QgAttributeFilter, OversampledRerankKeepsSoundness) {
    auto& data = shared_fixture();
    AttributeMatrix<float> raw_base;
    AttributeMatrix<float> raw_query;
    load_vecs<float, AttributeMatrix<float>>(
        fixture("qg_base_raw.fvecs").c_str(), raw_base
    );
    load_vecs<float, AttributeMatrix<float>>(
        fixture("qg_query_raw.fvecs").c_str(), raw_query
    );
    ASSERT_EQ(raw_base.rows, data.base.rows);
    ASSERT_EQ(raw_base.cols, kDim);
    ASSERT_EQ(raw_query.rows, data.query.rows);
    ASSERT_EQ(raw_query.cols, kDim);

    const size_t nq = data.query.rows;
    for (const size_t factor : {size_t{10}, size_t{40}}) {
        SCOPED_TRACE(factor);
        const size_t scaled = static_cast<size_t>(kTopK) * factor;
        const size_t search_k = std::min(scaled, data.graph.num_points());
        if (factor == 10) {
            EXPECT_EQ(search_k, size_t{100});
        } else {
            EXPECT_EQ(search_k, data.graph.num_points());
        }
        data.graph.set_ef(std::max(kEfSearch, search_k));

        std::vector<PID> candidates(nq * search_k);
        std::vector<float> candidate_dists(nq * search_k);
        data.graph.search_batch(
            data.query.data(), nq, static_cast<uint32_t>(search_k),
            candidates.data(), candidate_dists.data(),
            data.base_attr.data(), data.query_attr.data(), kAttrDim, 1
        );

        size_t violations = 0;
        for (size_t z = 0; z < nq; ++z) {
            std::vector<PID> ids(kTopK);
            std::vector<float> dists(kTopK);
            rabitqlib::attribute::rerank_by_original_l2(
                candidates.data() + z * search_k,
                search_k,
                raw_base.data(),
                raw_base.rows,
                kDim,
                raw_query.data() + z * kDim,
                ids.data(),
                dists.data(),
                kTopK,
                kPidMax
            );
            for (size_t y = 0; y < kTopK; ++y) {
                if (ids[y] == kPidMax) {
                    EXPECT_TRUE(std::isinf(dists[y]));
                    continue;
                }
                ASSERT_LT(ids[y], data.base.rows);
                if (!attr_match(
                        data.base_attr.data() +
                            static_cast<size_t>(ids[y]) * kAttrDim,
                        data.query_attr.data() + z * kAttrDim,
                        kAttrDim)) {
                    ++violations;
                }
                if (y > 0) {
                    EXPECT_LE(dists[y - 1], dists[y]);
                }
            }
        }
        EXPECT_EQ(violations, 0U);
    }
}

}  // namespace
}  // namespace rabitqlib::symqg
