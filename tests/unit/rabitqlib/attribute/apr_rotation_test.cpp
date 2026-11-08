#include "rabitqlib/attribute/apr_rotation.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "rabitqlib/attribute/attr_filter.hpp"
#include "rabitqlib/attribute/attr_io.hpp"
#include "rabitqlib/utils/io.hpp"

#ifndef RABITQ_TEST_ATTRIBUTE_FIXTURE_ROOT
#define RABITQ_TEST_ATTRIBUTE_FIXTURE_ROOT "tests/fixtures/attribute"
#endif

namespace rabitqlib::attribute {
namespace {

constexpr size_t kDim = 8;
constexpr size_t kAttrDim = 1;
constexpr size_t kPairDim = kDim / 2;

// Salt produced by default_rng(1234).integers(0, 2**63, size=D/2, dtype=int64).
// Embedding it keeps the numerical test independent of any external generator.
constexpr std::array<uint64_t, 4> kSalt = {
    9008465316566228445ULL,
    3506686710911391296ULL,
    8515443495629954650ULL,
    2413686584519761735ULL
};

// First row of apr_base.fvecs, cross-checked against the reference NumPy
// protocol when the fixture was generated.
constexpr std::array<float, kDim> kGoldenFirstRow = {
    -0.5918747186660767F,
    -0.6730067133903503F,
    -1.3855758905410767F,
    2.4279472827911377F,
    -2.813793897628784F,
    -1.6713343858718872F,
    3.8328464031219482F,
    1.227776050567627F
};

std::string Fixture(const char* name) {
    return std::string(RABITQ_TEST_ATTRIBUTE_FIXTURE_ROOT) + "/" + name;
}

struct Dataset {
    AttributeMatrix<float> base;
    AttributeMatrix<float> query;
    AttributeMatrix<float> golden_base;
    AttributeMatrix<float> golden_query;
    AttributeMatrix<uint32_t> base_attrs;
    AttributeMatrix<int32_t> query_attrs;
    std::vector<uint64_t> salt;
};

Dataset LoadDataset() {
    Dataset data;
    rabitqlib::load_vecs<float>(Fixture("base.fvecs").c_str(), data.base);
    rabitqlib::load_vecs<float>(Fixture("query.fvecs").c_str(), data.query);
    rabitqlib::load_vecs<float>(Fixture("apr_base.fvecs").c_str(), data.golden_base);
    rabitqlib::load_vecs<float>(Fixture("apr_query.fvecs").c_str(), data.golden_query);
    data.base_attrs = read_attr_bin<uint32_t>(
        Fixture("base_attr.bin").c_str(), data.base.rows, kAttrDim
    );
    data.query_attrs = read_attr_bin<int32_t>(
        Fixture("query_attr.bin").c_str(), data.query.rows, kAttrDim
    );
    data.salt = read_salt_file(Fixture("apr_salt.bin").c_str(), kAttrDim, kPairDim);
    return data;
}

AprRotationPlan ComparablePlan() {
    return AprRotationPlan::make(AprConfig{kDim, kAttrDim}, kSalt.data(), kSalt.size());
}

double SquaredDistance(const float* lhs, const float* rhs, size_t dim) {
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double delta = static_cast<double>(lhs[i]) - static_cast<double>(rhs[i]);
        sum += delta * delta;
    }
    return sum;
}

double SquaredNorm(const float* values, size_t dim) {
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        sum += static_cast<double>(values[i]) * static_cast<double>(values[i]);
    }
    return sum;
}

TEST(AprRotationTest, MatchesGoldenReference) {
    const Dataset data = LoadDataset();
    ASSERT_EQ(data.salt.size(), kSalt.size());
    for (size_t i = 0; i < kSalt.size(); ++i) {
        EXPECT_EQ(data.salt[i], kSalt[i]);
    }
    ASSERT_EQ(data.base.cols, kDim);
    ASSERT_EQ(data.query.cols, kDim);
    ASSERT_EQ(data.golden_base.buf.size(), data.base.buf.size());
    ASSERT_EQ(data.golden_query.buf.size(), data.query.buf.size());

    const AprRotationPlan plan = ComparablePlan();
    std::vector<float> rotated_base(data.base.buf.size());
    std::vector<float> rotated_query(data.query.buf.size());
    rotate_batch(
        plan, data.base.data(), data.base.rows, data.base_attrs.data(),
        rotated_base.data(), 0
    );
    rotate_batch(
        plan, data.query.data(), data.query.rows, data.query_attrs.data(),
        rotated_query.data(), 0
    );

    for (size_t i = 0; i < rotated_base.size(); ++i) {
        EXPECT_NEAR(rotated_base[i], data.golden_base.buf[i], 1e-5F) << "base index " << i;
    }
    for (size_t i = 0; i < rotated_query.size(); ++i) {
        EXPECT_NEAR(rotated_query[i], data.golden_query.buf[i], 1e-5F)
            << "query index " << i;
    }
    for (size_t j = 0; j < kDim; ++j) {
        EXPECT_FLOAT_EQ(rotated_base[j], kGoldenFirstRow[j]) << "first row index " << j;
    }
}

TEST(AprRotationTest, PreservesNormsAndMatchedDistances) {
    const Dataset data = LoadDataset();
    const AprRotationPlan plan = ComparablePlan();
    std::vector<float> rotated_base(data.base.buf.size());
    std::vector<float> rotated_query(data.query.buf.size());
    rotate_batch(
        plan, data.base.data(), data.base.rows, data.base_attrs.data(),
        rotated_base.data(), 0
    );
    rotate_batch(
        plan, data.query.data(), data.query.rows, data.query_attrs.data(),
        rotated_query.data(), 0
    );

    for (size_t i = 0; i < data.base.rows; ++i) {
        const double before = SquaredNorm(data.base.data() + i * kDim, kDim);
        const double after = SquaredNorm(rotated_base.data() + i * kDim, kDim);
        EXPECT_NEAR(after, before, 1e-4 * std::max(1.0, before)) << "base row " << i;
    }

    // Attaching the same target attribute to a base vector and a query vector
    // cancels the relative rotation, so the original L2 distance is preserved.
    size_t checked = 0;
    for (size_t i = 0; i < data.base.rows; ++i) {
        for (size_t j = 0; j < data.query.rows; ++j) {
            if (data.base_attrs(i, 0) != static_cast<uint32_t>(data.query_attrs(j, 0))) {
                continue;
            }
            const double before = SquaredDistance(
                data.base.data() + i * kDim, data.query.data() + j * kDim, kDim
            );
            const double after = SquaredDistance(
                rotated_base.data() + i * kDim, rotated_query.data() + j * kDim, kDim
            );
            EXPECT_NEAR(after, before, 1e-4 * std::max(1.0, before))
                << "pair (" << i << ", " << j << ")";
            ++checked;
        }
    }
    EXPECT_GT(checked, 0U);
}

TEST(AprRotationTest, DifferentAttributesProduceDifferentRotations) {
    const AprRotationPlan plan = ComparablePlan();
    constexpr std::array<float, kDim> kVector = {1, 2, 3, 4, 5, 6, 7, 8};
    const uint32_t first[1] = {0};
    const uint32_t second[1] = {1};
    std::array<float, kDim> rotated_first{};
    std::array<float, kDim> rotated_second{};
    rotate_vector(plan, kVector.data(), first, rotated_first.data());
    rotate_vector(plan, kVector.data(), second, rotated_second.data());

    double max_difference = 0.0;
    for (size_t j = 0; j < kDim; ++j) {
        max_difference = std::max(
            max_difference,
            std::abs(static_cast<double>(rotated_first[j] - rotated_second[j]))
        );
    }
    EXPECT_GT(max_difference, 1e-3);
}

TEST(AprRotationTest, ParallelMatchesSerial) {
    const Dataset data = LoadDataset();
    const AprRotationPlan plan = ComparablePlan();
    std::vector<float> serial(data.base.buf.size());
    std::vector<float> parallel(data.base.buf.size());
    rotate_batch(
        plan, data.base.data(), data.base.rows, data.base_attrs.data(), serial.data(), 0
    );
    rotate_batch(
        plan, data.base.data(), data.base.rows, data.base_attrs.data(), parallel.data(), 2
    );
    EXPECT_EQ(serial, parallel);
}

TEST(AprRotationTest, StandalonePlanIsSelfContained) {
    const AprRotationPlan standalone =
        AprRotationPlan::make_standalone(AprConfig{kDim, kAttrDim}, 42);
    const AprRotationPlan comparable = ComparablePlan();
    constexpr std::array<float, kDim> kVector = {1, -2, 3, -4, 5, -6, 7, -8};
    const uint32_t attr[1] = {0};
    std::array<float, kDim> standalone_out{};
    std::array<float, kDim> comparable_out{};
    rotate_vector(standalone, kVector.data(), attr, standalone_out.data());
    rotate_vector(comparable, kVector.data(), attr, comparable_out.data());

    EXPECT_NE(standalone_out, comparable_out);
    EXPECT_NEAR(
        SquaredNorm(standalone_out.data(), kDim), SquaredNorm(kVector.data(), kDim),
        1e-2
    );
}

TEST(AprRotationTest, AttrMatchTruthTable) {
    const uint32_t row[2] = {3, 7};
    const int32_t exact[2] = {3, 7};
    const int32_t wildcard[2] = {-1, 7};
    const int32_t mismatch[2] = {4, 7};
    EXPECT_TRUE(attr_match(row, exact, 2));
    EXPECT_TRUE(attr_match(row, wildcard, 2));
    EXPECT_FALSE(attr_match(row, mismatch, 2));
    EXPECT_TRUE(attr_match(row, exact, 0));
}

TEST(AprRotationTest, RejectsInvalidConfigAndInputs) {
    EXPECT_THROW(
        (AprRotationPlan::make(AprConfig{7, kAttrDim}, kSalt.data(), kSalt.size())),
        std::invalid_argument
    );
    EXPECT_THROW(
        (AprRotationPlan::make(AprConfig{kDim, 0}, kSalt.data(), kSalt.size())),
        std::invalid_argument
    );
    EXPECT_THROW(
        (AprRotationPlan::make(AprConfig{kDim, kAttrDim}, kSalt.data(), 3)),
        std::invalid_argument
    );
    EXPECT_THROW(
        (AprRotationPlan::make(AprConfig{kDim, kAttrDim}, nullptr, kSalt.size())),
        std::invalid_argument
    );

    const AprRotationPlan plan = ComparablePlan();
    const float vector[kDim] = {1, 2, 3, 4, 5, 6, 7, 8};
    float output[kDim] = {};
    const int32_t wildcard[1] = {-1};
    const uint32_t attr[1] = {0};
    EXPECT_THROW(rotate_vector(plan, vector, wildcard, output), std::invalid_argument);
    EXPECT_THROW(
        (rotate_batch(plan, nullptr, 1, attr, output, 0)), std::invalid_argument
    );
    EXPECT_NO_THROW((rotate_batch(plan, vector, 0, attr, output, 0)));
}

}  // namespace
}  // namespace rabitqlib::attribute
