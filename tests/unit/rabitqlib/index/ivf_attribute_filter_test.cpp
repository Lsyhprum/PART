#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "rabitqlib/attribute/attr_filter.hpp"
#include "rabitqlib/attribute/attr_io.hpp"
#include "rabitqlib/attribute/attr_rerank.hpp"
#include "rabitqlib/index/ivf/ivf.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/io.hpp"
#include "rabitqlib/utils/space.hpp"

namespace rabitqlib::ivf {
namespace {

using rabitqlib::attribute::AttributeMatrix;
using rabitqlib::attribute::attr_match;
using rabitqlib::attribute::read_attr_bin;

constexpr size_t kDim = 64;
constexpr size_t kNumBase = 256;
constexpr size_t kNumClusters = 8;
constexpr size_t kAttrDim = 1;
constexpr size_t kTopK = 10;

std::string fixture(const char* name) {
    return std::string(RABITQ_TEST_ATTRIBUTE_FIXTURE_ROOT) + "/" + name;
}

// One shared, in-memory set of indexes built from the fixed rotated fixture.
// No index file is written and no data is generated at test runtime. Three
// bit widths cover both result-pool insertion points: one-bit estimates and
// the stage-2 refined path (extra-bit and raw-vector storage).
struct Fixture {
    AttributeMatrix<float> base;
    AttributeMatrix<float> query;
    AttributeMatrix<uint32_t> gt;
    AttributeMatrix<uint32_t> base_attr;
    AttributeMatrix<int32_t> query_attr;
    AttributeMatrix<float> raw_base;
    AttributeMatrix<float> raw_query;
    AttributeMatrix<float> centroids;
    AttributeMatrix<uint32_t> cluster_ids;
    IVF one_bit;
    IVF quantized;
    IVF raw;

    Fixture()
        : one_bit(kNumBase, kDim, kNumClusters, 1, METRIC_L2),
          quantized(kNumBase, kDim, kNumClusters, 5, METRIC_L2),
          raw(kNumBase, kDim, kNumClusters, 32, METRIC_L2) {
        load_vecs<float, AttributeMatrix<float>>(
            fixture("qg_base.fvecs").c_str(), base
        );
        load_vecs<float, AttributeMatrix<float>>(
            fixture("qg_query.fvecs").c_str(), query
        );
        load_vecs<uint32_t, AttributeMatrix<uint32_t>>(
            fixture("qg_filtered_gt.ivecs").c_str(), gt
        );
        load_vecs<float, AttributeMatrix<float>>(
            fixture("qg_base_raw.fvecs").c_str(), raw_base
        );
        load_vecs<float, AttributeMatrix<float>>(
            fixture("qg_query_raw.fvecs").c_str(), raw_query
        );
        load_vecs<float, AttributeMatrix<float>>(
            fixture("ivf_centroids.fvecs").c_str(), centroids
        );
        load_vecs<uint32_t, AttributeMatrix<uint32_t>>(
            fixture("ivf_cluster_ids.ivecs").c_str(), cluster_ids
        );
        base_attr = read_attr_bin<uint32_t>(fixture("qg_base_attr.bin").c_str());
        query_attr = read_attr_bin<int32_t>(fixture("qg_query_attr.bin").c_str());

        if (base.rows != kNumBase || base.cols != kDim ||
            query.cols != kDim || gt.rows != query.rows || gt.cols != kTopK ||
            raw_base.rows != kNumBase || raw_base.cols != kDim ||
            raw_query.rows != query.rows || raw_query.cols != kDim ||
            centroids.rows != kNumClusters || centroids.cols != kDim ||
            cluster_ids.rows != kNumBase || cluster_ids.cols != 1 ||
            base_attr.rows != kNumBase || base_attr.cols != kAttrDim ||
            query_attr.rows != query.rows || query_attr.cols != kAttrDim) {
            throw std::runtime_error("IVF attribute fixture has an unexpected shape");
        }
        one_bit.construct(base.data(), centroids.data(), cluster_ids.data(), false, 1);
        quantized.construct(base.data(), centroids.data(), cluster_ids.data(), false, 1);
        raw.construct(base.data(), centroids.data(), cluster_ids.data(), false, 1);
    }

    std::array<IVF*, 3> indexes() { return {&one_bit, &quantized, &raw}; }
};

Fixture& shared_fixture() {
    static Fixture instance;
    return instance;
}

void search_filtered(IVF& ivf, Fixture& data, size_t z, PID* ids, float* dists) {
    ivf.search(
        data.query.data() + z * kDim,
        kTopK,
        kNumClusters,
        ids,
        dists,
        data.base_attr.data(),
        data.query_attr.data() + z * kAttrDim,
        kAttrDim
    );
}

size_t count_violations(
    Fixture& data, size_t z, const PID* ids, size_t count
) {
    size_t violations = 0;
    for (size_t i = 0; i < count; ++i) {
        if (ids[i] == kPidMax) {
            continue;
        }
        if (!attr_match(
                data.base_attr.data() + static_cast<size_t>(ids[i]) * kAttrDim,
                data.query_attr.data() + z * kAttrDim,
                kAttrDim)) {
            ++violations;
        }
    }
    return violations;
}

TEST(IvfAttributeFilter, FilteredSearchIsSound) {
    auto& data = shared_fixture();
    for (IVF* ivf : data.indexes()) {
        SCOPED_TRACE(ivf->nbits());
        size_t returned = 0;
        size_t violations = 0;
        for (size_t z = 0; z < data.query.rows; ++z) {
            std::vector<PID> ids(kTopK);
            std::vector<float> dists(kTopK);
            search_filtered(*ivf, data, z, ids.data(), dists.data());
            for (size_t j = 0; j < kTopK; ++j) {
                if (ids[j] == kPidMax) {
                    EXPECT_TRUE(std::isinf(dists[j]));
                    continue;
                }
                ++returned;
                ASSERT_LT(ids[j], data.base.rows);
            }
            violations += count_violations(data, z, ids.data(), kTopK);
        }
        EXPECT_EQ(violations, 0U);
        EXPECT_GT(returned, 0U);
    }
}

TEST(IvfAttributeFilter, UnfilteredSearchReturnsMismatches) {
    auto& data = shared_fixture();
    for (IVF* ivf : data.indexes()) {
        SCOPED_TRACE(ivf->nbits());
        size_t mismatches = 0;
        for (size_t z = 0; z < data.query.rows; ++z) {
            std::vector<PID> ids(kNumBase);
            std::vector<float> dists(kNumBase);
            ivf->search(
                data.query.data() + z * kDim, kNumBase, kNumClusters,
                ids.data(), dists.data()
            );
            mismatches += count_violations(data, z, ids.data(), kNumBase);
        }
        EXPECT_GT(mismatches, 0U);
    }
}

TEST(IvfAttributeFilter, BatchMatchesSingleQuery) {
    auto& data = shared_fixture();
    const size_t nq = data.query.rows;
    for (IVF* ivf : data.indexes()) {
        SCOPED_TRACE(ivf->nbits());
        std::vector<PID> batch_ids(nq * kTopK);
        std::vector<PID> single_ids(nq * kTopK);
        std::vector<float> batch_dists(nq * kTopK);
        std::vector<float> single_dists(nq * kTopK);
        ivf->search_batch(
            data.query.data(), nq, kTopK, kNumClusters,
            batch_ids.data(), batch_dists.data(),
            data.base_attr.data(), data.query_attr.data(), kAttrDim,
            std::nullopt, 1
        );
        for (size_t z = 0; z < nq; ++z) {
            search_filtered(
                *ivf, data, z, single_ids.data() + z * kTopK,
                single_dists.data() + z * kTopK
            );
        }
        EXPECT_EQ(batch_ids, single_ids);
        for (size_t i = 0; i < batch_dists.size(); ++i) {
            EXPECT_FLOAT_EQ(batch_dists[i], single_dists[i]);
        }
    }
}

TEST(IvfAttributeFilter, FilteredRecallAgainstGroundTruth) {
    auto& data = shared_fixture();
    for (IVF* ivf : data.indexes()) {
        SCOPED_TRACE(ivf->nbits());
        size_t correct = 0;
        for (size_t z = 0; z < data.query.rows; ++z) {
            std::vector<PID> ids(kTopK);
            std::vector<float> dists(kTopK);
            search_filtered(*ivf, data, z, ids.data(), dists.data());
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
}

TEST(IvfAttributeFilter, RejectsInvalidGateInputs) {
    auto& data = shared_fixture();
    IVF& ivf = data.quantized;
    std::vector<PID> ids(kTopK);
    std::vector<float> dists(kTopK);
    const float* q = data.query.data();
    const uint32_t* ba = data.base_attr.data();
    const int32_t* qa = data.query_attr.data();

    EXPECT_THROW(
        ivf.search(q, kTopK, kNumClusters, ids.data(), dists.data(),
                   nullptr, qa, kAttrDim),
        std::invalid_argument
    );
    EXPECT_THROW(
        ivf.search(q, kTopK, kNumClusters, ids.data(), dists.data(),
                   ba, nullptr, kAttrDim),
        std::invalid_argument
    );
    EXPECT_THROW(
        ivf.search(q, kTopK, kNumClusters, ids.data(), dists.data(),
                   ba, qa, 0),
        std::invalid_argument
    );
    std::vector<int32_t> negative(kAttrDim, -1);
    EXPECT_THROW(
        ivf.search(q, kTopK, kNumClusters, ids.data(), dists.data(),
                   ba, negative.data(), kAttrDim),
        std::invalid_argument
    );

    const size_t nq = data.query.rows;
    std::vector<PID> batch_ids(nq * kTopK);
    std::vector<float> batch_dists(nq * kTopK);
    std::vector<int32_t> batch_bad(nq * kAttrDim, 0);
    batch_bad[3] = -1;
    EXPECT_THROW(
        ivf.search_batch(
            data.query.data(), nq, kTopK, kNumClusters,
            batch_ids.data(), batch_dists.data(),
            ba, batch_bad.data(), kAttrDim
        ),
        std::invalid_argument
    );

    // The explicit-use_hacc overload shares the same gate.
    EXPECT_NO_THROW(ivf.search(
        q, kTopK, kNumClusters, ids.data(), dists.data(), false,
        ba, qa, kAttrDim
    ));
}

TEST(IvfAttributeFilter, RotatedSpacePreservesMatchingDistances) {
    auto& data = shared_fixture();
    size_t checked = 0;
    for (size_t z = 0; z < data.query.rows; ++z) {
        size_t per_query = 0;
        for (size_t id = 0; id < data.base.rows && per_query < 3; ++id) {
            if (!attr_match(
                    data.base_attr.data() + id * kAttrDim,
                    data.query_attr.data() + z * kAttrDim, kAttrDim)) {
                continue;
            }
            double rotated = 0.0;
            double original = 0.0;
            for (size_t j = 0; j < kDim; ++j) {
                const double dr = static_cast<double>(data.base(id, j)) -
                                  static_cast<double>(data.query(z, j));
                const double du = static_cast<double>(data.raw_base(id, j)) -
                                  static_cast<double>(data.raw_query(z, j));
                rotated += dr * dr;
                original += du * du;
            }
            EXPECT_NEAR(rotated, original, 1e-3 * std::max(1.0, original));
            ++per_query;
            ++checked;
        }
    }
    EXPECT_GT(checked, 0U);
}

TEST(IvfAttributeFilter, OversampledRerankKeepsSoundness) {
    auto& data = shared_fixture();
    IVF& ivf = data.quantized;
    const size_t nq = data.query.rows;
    for (const size_t factor : {size_t{10}, size_t{40}}) {
        SCOPED_TRACE(factor);
        const size_t search_k =
            std::min(kTopK * factor, static_cast<size_t>(ivf.num_points()));
        if (factor == 10) {
            EXPECT_EQ(search_k, size_t{100});
        } else {
            EXPECT_EQ(search_k, static_cast<size_t>(ivf.num_points()));
        }

        std::vector<PID> candidates(nq * search_k);
        std::vector<float> candidate_dists(nq * search_k);
        ivf.search_batch(
            data.query.data(), nq, search_k, kNumClusters,
            candidates.data(), candidate_dists.data(),
            data.base_attr.data(), data.query_attr.data(), kAttrDim,
            std::nullopt, 1
        );

        size_t violations = 0;
        for (size_t z = 0; z < nq; ++z) {
            std::vector<PID> ids(kTopK);
            std::vector<float> dists(kTopK);
            rabitqlib::attribute::rerank_by_original_l2(
                candidates.data() + z * search_k,
                search_k,
                data.raw_base.data(),
                data.raw_base.rows,
                kDim,
                data.raw_query.data() + z * kDim,
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
                if (y > 0) {
                    EXPECT_LE(dists[y - 1], dists[y]);
                }
            }
            violations += count_violations(data, z, ids.data(), kTopK);
        }
        EXPECT_EQ(violations, 0U);
    }
}

}  // namespace
}  // namespace rabitqlib::ivf
