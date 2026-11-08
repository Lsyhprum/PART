#include "rabitqlib/attribute/apr_rotation.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "rabitqlib/attribute/attr_io.hpp"
#include "rabitqlib/utils/io.hpp"

#ifndef RABITQ_TEST_ATTRIBUTE_FIXTURE_ROOT
#define RABITQ_TEST_ATTRIBUTE_FIXTURE_ROOT "tests/fixtures/attribute"
#endif

namespace rabitqlib::attribute {
namespace {

constexpr size_t kDim = 64;
constexpr size_t kPairDim = kDim / 2;
constexpr uint64_t kSeed = 1234;
constexpr double kRangeL = 100.0;
constexpr double kPi = 3.14159265358979323846;

std::string Fixture(const char* name) {
    return std::string(RABITQ_TEST_ATTRIBUTE_FIXTURE_ROOT) + "/" + name;
}

struct Dataset {
    AttributeMatrix<float> raw_base;
    AttributeMatrix<float> raw_query;
    AttributeMatrix<float> mixed_base;
    AttributeMatrix<float> mixed_query;
    AttributeMatrix<float> range_base;
    AttributeMatrix<float> range_query;
    AttributeMatrix<uint32_t> base_attr;
    AttributeMatrix<int32_t> query_attr;
    AttributeMatrix<uint32_t> range_base_attr;
    AttributeMatrix<int32_t> range_query_attr;
};

Dataset LoadDataset() {
    Dataset data;
    rabitqlib::load_vecs<float>(Fixture("qg_base_raw.fvecs").c_str(), data.raw_base);
    rabitqlib::load_vecs<float>(Fixture("qg_query_raw.fvecs").c_str(), data.raw_query);
    rabitqlib::load_vecs<float>(Fixture("hapr_base.fvecs").c_str(), data.mixed_base);
    rabitqlib::load_vecs<float>(Fixture("hapr_query.fvecs").c_str(), data.mixed_query);
    rabitqlib::load_vecs<float>(
        Fixture("hapr_range_base.fvecs").c_str(), data.range_base
    );
    rabitqlib::load_vecs<float>(
        Fixture("hapr_range_query.fvecs").c_str(), data.range_query
    );
    data.base_attr = read_attr_bin<uint32_t>(Fixture("hapr_base_attr.bin").c_str());
    data.query_attr = read_attr_bin<int32_t>(Fixture("hapr_query_attr.bin").c_str());
    data.range_base_attr =
        read_attr_bin<uint32_t>(Fixture("hapr_range_base_attr.bin").c_str());
    data.range_query_attr =
        read_attr_bin<int32_t>(Fixture("hapr_range_query_attr.bin").c_str());
    return data;
}

const Dataset& SharedDataset() {
    static const Dataset instance = LoadDataset();
    return instance;
}

AprRotationPlan MakeMixedPlan() {
    const AttributeKind layout[2] = {
        AttributeKind::kEquality, AttributeKind::kRange
    };
    const AprConfig config{
        kDim, 2, PredicateKind::kRange, WildcardPolicy::kReject, kRangeL
    };
    return AprRotationPlan::make_hapr_standalone(config, kSeed, layout, 2, kRangeL);
}

AprRotationPlan MakeRangePlan() {
    const AttributeKind layout[1] = {AttributeKind::kRange};
    const AprConfig config{
        kDim, 1, PredicateKind::kRange, WildcardPolicy::kReject, kRangeL
    };
    return AprRotationPlan::make_hapr_standalone(config, kSeed, layout, 1, kRangeL);
}

std::vector<double> MixedBaseValues(const Dataset& data) {
    std::vector<double> values(data.raw_base.rows * 2);
    for (size_t i = 0; i < data.raw_base.rows; ++i) {
        values[2 * i] = static_cast<double>(data.base_attr(i, 0));
        values[2 * i + 1] = static_cast<double>(data.base_attr(i, 1));
    }
    return values;
}

std::vector<double> MixedQueryValues(const Dataset& data) {
    std::vector<double> values(data.raw_query.rows * 2);
    for (size_t i = 0; i < data.raw_query.rows; ++i) {
        values[2 * i] = static_cast<double>(data.query_attr(i, 0));
        values[2 * i + 1] =
            (static_cast<double>(data.query_attr(i, 1)) +
             static_cast<double>(data.query_attr(i, 2))) /
            2.0;
    }
    return values;
}

std::vector<double> RangeBaseValues(const Dataset& data) {
    std::vector<double> values(data.raw_base.rows);
    for (size_t i = 0; i < data.raw_base.rows; ++i) {
        values[i] = static_cast<double>(data.range_base_attr(i, 0));
    }
    return values;
}

std::vector<double> RangeQueryValues(const Dataset& data) {
    std::vector<double> values(data.raw_query.rows);
    for (size_t i = 0; i < data.raw_query.rows; ++i) {
        values[i] =
            (static_cast<double>(data.range_query_attr(i, 0)) +
             static_cast<double>(data.range_query_attr(i, 1))) /
            2.0;
    }
    return values;
}

double SquaredNorm(const float* values, size_t dim) {
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        sum += static_cast<double>(values[i]) * static_cast<double>(values[i]);
    }
    return sum;
}

double SquaredDistance(const float* lhs, const float* rhs, size_t dim) {
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double delta = static_cast<double>(lhs[i]) - static_cast<double>(rhs[i]);
        sum += delta * delta;
    }
    return sum;
}

std::vector<float> PairBasis(size_t dim) {
    std::vector<float> values(dim, 0.0F);
    for (size_t j = 0; j < dim / 2; ++j) {
        values[2 * j] = 1.0F;
    }
    return values;
}

AprRotationPlan MakeSmallRangePlan() {
    constexpr size_t kSmallDim = 8;
    const AttributeKind layout[1] = {AttributeKind::kRange};
    const AprConfig config{
        kSmallDim, 1, PredicateKind::kRange, WildcardPolicy::kReject, kRangeL
    };
    const std::vector<uint64_t> salt(kSmallDim / 2, 0);
    return AprRotationPlan::make_hapr(
        config, salt.data(), salt.size(), layout, 1, kRangeL
    );
}

TEST(HaprRotationTest, MatchesGoldenMixedFixture) {
    const Dataset& data = SharedDataset();
    ASSERT_EQ(data.raw_base.cols, kDim);
    ASSERT_EQ(data.raw_query.cols, kDim);
    ASSERT_EQ(data.mixed_base.buf.size(), data.raw_base.buf.size());
    ASSERT_EQ(data.mixed_query.buf.size(), data.raw_query.buf.size());

    const AprRotationPlan plan = MakeMixedPlan();
    const std::vector<double> base_values = MixedBaseValues(data);
    const std::vector<double> query_values = MixedQueryValues(data);
    std::vector<float> rotated_base(data.raw_base.buf.size());
    std::vector<float> rotated_query(data.raw_query.buf.size());
    rotate_batch_hapr(
        plan, data.raw_base.data(), data.raw_base.rows, base_values.data(),
        rotated_base.data(), 0
    );
    rotate_batch_hapr(
        plan, data.raw_query.data(), data.raw_query.rows, query_values.data(),
        rotated_query.data(), 0
    );
    for (size_t i = 0; i < rotated_base.size(); ++i) {
        EXPECT_FLOAT_EQ(rotated_base[i], data.mixed_base.buf[i]) << "base " << i;
    }
    for (size_t i = 0; i < rotated_query.size(); ++i) {
        EXPECT_FLOAT_EQ(rotated_query[i], data.mixed_query.buf[i]) << "query " << i;
    }
}

TEST(HaprRotationTest, MatchesGoldenRangeFixture) {
    const Dataset& data = SharedDataset();
    ASSERT_EQ(data.range_base.buf.size(), data.raw_base.buf.size());
    ASSERT_EQ(data.range_query.buf.size(), data.raw_query.buf.size());

    const AprRotationPlan plan = MakeRangePlan();
    const std::vector<double> base_values = RangeBaseValues(data);
    const std::vector<double> query_values = RangeQueryValues(data);
    std::vector<float> rotated_base(data.raw_base.buf.size());
    std::vector<float> rotated_query(data.raw_query.buf.size());
    rotate_batch_hapr(
        plan, data.raw_base.data(), data.raw_base.rows, base_values.data(),
        rotated_base.data(), 0
    );
    rotate_batch_hapr(
        plan, data.raw_query.data(), data.raw_query.rows, query_values.data(),
        rotated_query.data(), 0
    );
    for (size_t i = 0; i < rotated_base.size(); ++i) {
        EXPECT_FLOAT_EQ(rotated_base[i], data.range_base.buf[i]) << "base " << i;
    }
    for (size_t i = 0; i < rotated_query.size(); ++i) {
        EXPECT_FLOAT_EQ(rotated_query[i], data.range_query.buf[i]) << "query " << i;
    }
}

TEST(HaprRotationTest, AllEqualityHaprMatchesClassicRotation) {
    const Dataset& data = SharedDataset();
    const AttributeKind layout[2] = {
        AttributeKind::kEquality, AttributeKind::kEquality
    };
    const AprConfig config{kDim, 2, PredicateKind::kEquality,
                           WildcardPolicy::kReject, 0.0};
    const AprRotationPlan hapr = AprRotationPlan::make_hapr_standalone(
        config, kSeed, layout, 2, 0.0
    );
    const AprRotationPlan classic = AprRotationPlan::make_standalone(config, kSeed);

    std::vector<double> values(data.raw_base.rows * 2);
    for (size_t i = 0; i < data.raw_base.rows; ++i) {
        values[2 * i] = static_cast<double>(data.base_attr(i, 0));
        values[2 * i + 1] = static_cast<double>(data.base_attr(i, 1));
    }
    std::vector<float> from_hapr(data.raw_base.buf.size());
    std::vector<float> from_classic(data.raw_base.buf.size());
    rotate_batch_hapr(
        hapr, data.raw_base.data(), data.raw_base.rows, values.data(),
        from_hapr.data(), 0
    );
    rotate_batch(
        classic, data.raw_base.data(), data.raw_base.rows, data.base_attr.data(),
        from_classic.data(), 0
    );
    EXPECT_EQ(from_hapr, from_classic);
    EXPECT_FALSE(hapr.has_range());
}

TEST(HaprRotationTest, PreservesNormsOnUnnormalisedInput) {
    const Dataset& data = SharedDataset();
    double mean_norm = 0.0;
    for (size_t i = 0; i < data.raw_base.rows; ++i) {
        const double before = SquaredNorm(data.raw_base.data() + i * kDim, kDim);
        const double after = SquaredNorm(data.mixed_base.data() + i * kDim, kDim);
        EXPECT_NEAR(after, before, 1e-4 * std::max(1.0, before)) << "row " << i;
        mean_norm += std::sqrt(before);
    }
    mean_norm /= static_cast<double>(data.raw_base.rows);
    // The fixture is deliberately NOT unit norm; normalising would change the
    // rerank metric and break comparability with the published experiments.
    EXPECT_GT(std::abs(mean_norm - 1.0), 0.1);
}

TEST(HaprRotationTest, RangePhaseUsesGlobalPairIndex) {
    constexpr size_t kSmallDim = 8;
    constexpr size_t kSmallPairs = kSmallDim / 2;
    const AprRotationPlan plan = MakeSmallRangePlan();
    const std::vector<float> src = PairBasis(kSmallDim);
    std::array<float, kSmallDim> dst{};
    const double value[1] = {25.0};
    rotate_vector_hapr(plan, src.data(), value, dst.data());
    for (size_t j = 0; j < kSmallPairs; ++j) {
        const double expected =
            kPi * (static_cast<double>(j) / kSmallPairs) * (25.0 / kRangeL);
        EXPECT_NEAR(dst[2 * j], std::cos(expected), 1e-6) << "cos pair " << j;
        EXPECT_NEAR(dst[2 * j + 1], std::sin(expected), 1e-6) << "sin pair " << j;
    }
    // j starts at zero: no +1 offset and no base frequency multiplier.
    EXPECT_FLOAT_EQ(dst[0], 1.0F);
    EXPECT_FLOAT_EQ(dst[1], 0.0F);
}

TEST(HaprRotationTest, HalfIntegerCentreIsNotTruncated) {
    constexpr size_t kSmallDim = 8;
    const AprRotationPlan plan = MakeSmallRangePlan();
    const std::vector<float> src = PairBasis(kSmallDim);
    std::array<float, kSmallDim> integral{};
    std::array<float, kSmallDim> half{};
    const double integral_value[1] = {20.0};
    const double half_value[1] = {20.5};
    rotate_vector_hapr(plan, src.data(), integral_value, integral.data());
    rotate_vector_hapr(plan, src.data(), half_value, half.data());
    EXPECT_NE(integral, half);
    const double expected = kPi * (1.0 / (kSmallDim / 2)) * (20.5 / kRangeL);
    EXPECT_NEAR(half[2], std::cos(expected), 1e-6);
    EXPECT_NEAR(half[3], std::sin(expected), 1e-6);
}

TEST(HaprRotationTest, WindowCentrePreservesDistanceAndOutsideDiffers) {
    const AprRotationPlan plan = MakeRangePlan();
    const std::vector<float> src = PairBasis(kDim);
    std::vector<float> query(kDim);
    std::vector<float> matched(kDim);
    std::vector<float> outside(kDim);
    const double centre[1] = {20.0};
    const double same[1] = {20.0};
    const double far[1] = {90.0};
    rotate_vector_hapr(plan, src.data(), centre, query.data());
    rotate_vector_hapr(plan, src.data(), same, matched.data());
    rotate_vector_hapr(plan, src.data(), far, outside.data());
    EXPECT_NEAR(SquaredDistance(matched.data(), query.data(), kDim), 0.0, 1e-8);
    EXPECT_GT(SquaredDistance(outside.data(), query.data(), kDim), 1.0);
}

TEST(HaprRotationTest, SoftFilteringGrowsWithValueOffset) {
    const AprRotationPlan plan = MakeRangePlan();
    const std::vector<float> src = PairBasis(kDim);
    std::vector<float> query(kDim);
    std::vector<float> candidate(kDim);
    const double centre[1] = {20.0};
    rotate_vector_hapr(plan, src.data(), centre, query.data());
    const double offsets[] = {0.0, 0.5, 1.0, 2.0, 5.0};
    double previous = -1.0;
    for (const double offset : offsets) {
        const double value[1] = {20.0 + offset};
        rotate_vector_hapr(plan, src.data(), value, candidate.data());
        const double distance = SquaredDistance(candidate.data(), query.data(), kDim);
        EXPECT_GT(distance, previous) << "offset " << offset;
        previous = distance;
    }
}

TEST(HaprRotationTest, ParallelMatchesSerial) {
    const Dataset& data = SharedDataset();
    const AprRotationPlan plan = MakeMixedPlan();
    const std::vector<double> values = MixedBaseValues(data);
    std::vector<float> serial(data.raw_base.buf.size());
    std::vector<float> parallel(data.raw_base.buf.size());
    rotate_batch_hapr(
        plan, data.raw_base.data(), data.raw_base.rows, values.data(),
        serial.data(), 0
    );
    rotate_batch_hapr(
        plan, data.raw_base.data(), data.raw_base.rows, values.data(),
        parallel.data(), 2
    );
    EXPECT_EQ(serial, parallel);
}

TEST(HaprRotationTest, RejectsInvalidLayoutsAndInputs) {
    const AttributeKind layout[1] = {AttributeKind::kRange};
    const AprConfig config{
        kDim, 1, PredicateKind::kRange, WildcardPolicy::kReject, kRangeL
    };
    EXPECT_THROW(
        (AprRotationPlan::make_hapr_standalone(config, kSeed, nullptr, 1, kRangeL)),
        std::invalid_argument
    );
    EXPECT_THROW(
        (AprRotationPlan::make_hapr_standalone(config, kSeed, layout, 2, kRangeL)),
        std::invalid_argument
    );
    EXPECT_THROW(
        (AprRotationPlan::make_hapr_standalone(config, kSeed, layout, 1, 0.0)),
        std::invalid_argument
    );
    EXPECT_THROW(
        (AprRotationPlan::make_hapr_standalone(config, kSeed, layout, 1, -1.0)),
        std::invalid_argument
    );
    const AttributeKind all_eq[1] = {AttributeKind::kEquality};
    EXPECT_THROW(
        (AprRotationPlan::make_hapr_standalone(config, kSeed, all_eq, 1, kRangeL)),
        std::invalid_argument
    );
    const AttributeKind unknown[1] = {static_cast<AttributeKind>(7)};
    EXPECT_THROW(
        (AprRotationPlan::make_hapr_standalone(config, kSeed, unknown, 1, kRangeL)),
        std::invalid_argument
    );
    const AprConfig odd_dim{
        7, 1, PredicateKind::kRange, WildcardPolicy::kReject, kRangeL
    };
    EXPECT_THROW(
        (AprRotationPlan::make_hapr_standalone(odd_dim, kSeed, layout, 1, kRangeL)),
        std::invalid_argument
    );
    const AprConfig bad_wildcard{
        kDim, 1, PredicateKind::kRange, static_cast<WildcardPolicy>(9), kRangeL
    };
    EXPECT_THROW(
        (AprRotationPlan::make_hapr_standalone(
            bad_wildcard, kSeed, layout, 1, kRangeL
        )),
        std::invalid_argument
    );
    // The classic factories must not silently accept a range layout.
    EXPECT_THROW(
        (AprRotationPlan::make_standalone(config, kSeed)), std::invalid_argument
    );

    const AprRotationPlan plan = MakeMixedPlan();
    const std::vector<float> src = PairBasis(kDim);
    std::vector<float> dst(kDim);
    const double negative[2] = {0.0, -1.0};
    const double fractional_eq[2] = {0.5, 20.0};
    EXPECT_THROW(
        (rotate_vector_hapr(plan, src.data(), negative, dst.data())),
        std::invalid_argument
    );
    EXPECT_THROW(
        (rotate_vector_hapr(plan, src.data(), fractional_eq, dst.data())),
        std::invalid_argument
    );
    EXPECT_THROW(
        (rotate_batch_hapr(plan, nullptr, 1, fractional_eq, dst.data(), 0)),
        std::invalid_argument
    );
    EXPECT_NO_THROW((rotate_batch_hapr(plan, src.data(), 0, fractional_eq,
                                       dst.data(), 0)));
}

}  // namespace
}  // namespace rabitqlib::attribute
