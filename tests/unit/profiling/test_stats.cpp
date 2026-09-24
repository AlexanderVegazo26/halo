// Statistics are pinned against numpy 2.x: np.percentile (default "linear" = Hyndman-Fan
// type 7) and np.std(ddof=1). Values generated once with /root/halo-py/.venv and hard-coded.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include <nlohmann/json.hpp>

#include "halo/profiling/stats.h"
#include "test_util.h"

using namespace halo::profiling;
using halo::ErrorCode;

namespace {

void expect_stats(const std::vector<double>& xs, std::size_t n, double mn, double mx, double mean, double median,
                  double sd, double p95, double p99, double cv) {
    const SampleStats s = compute_sample_stats(xs);
    EXPECT_EQ(s.n, n);
    EXPECT_DOUBLE_EQ(s.min, mn);
    EXPECT_DOUBLE_EQ(s.max, mx);
    EXPECT_NEAR(s.mean, mean, 1e-12);
    EXPECT_NEAR(s.median, median, 1e-12);
    EXPECT_NEAR(s.p50, median, 1e-12);
    EXPECT_NEAR(s.stddev, sd, 1e-12);
    EXPECT_NEAR(s.p95, p95, 1e-12);
    EXPECT_NEAR(s.p99, p99, 1e-12);
    if (std::isinf(cv)) {
        EXPECT_TRUE(std::isinf(s.cv));
    } else {
        EXPECT_NEAR(s.cv, cv, 1e-12);
    }
}

}  // namespace

TEST(Stats, OddCountType7LinearAndSampleStddev) {
    // sorted 1 2 3 5 7 9 10; p95: h = 6*0.95 = 5.7 -> 9 + 0.7*(10-9) = 9.7
    expect_stats({9, 1, 5, 3, 7, 2, 10}, 7, 1, 10, 5.2857142857142856, 5, 3.4982989063393708, 9.7, 9.94,
                 0.66184033363177286);
}

TEST(Stats, EvenCountMedianIsMeanOfMiddlePair) {
    // sorted 1..6; median = (3+4)/2; p95: h = 5*0.95 = 4.75 -> 5 + 0.75 = 5.75
    expect_stats({4, 1, 3, 2, 6, 5}, 6, 1, 6, 3.5, 3.5, 1.8708286933869707, 5.75, 5.95, 0.53452248382484879);
}

TEST(Stats, SingleSampleHasZeroSpread) {
    expect_stats({42.5}, 1, 42.5, 42.5, 42.5, 42.5, 0, 42.5, 42.5, 0);
}

TEST(Stats, ConstantSampleHasZeroCv) {
    expect_stats({3, 3, 3, 3}, 4, 3, 3, 3, 3, 0, 3, 3, 0);
}

TEST(Stats, ZeroMeanWithSpreadHasInfiniteCvSerializedAsNull) {
    expect_stats({-2, 2, -1, 1}, 4, -2, 2, 0, 0, 1.8257418583505538, 1.8499999999999996, 1.9699999999999998,
                 std::numeric_limits<double>::infinity());
    const nlohmann::json j = compute_sample_stats(std::vector<double>{-2, 2, -1, 1});
    EXPECT_TRUE(j.at("cv").is_null());
    SampleStats back = j.get<SampleStats>();
    EXPECT_TRUE(std::isinf(back.cv));
}

TEST(Stats, TwentySamplesWithOutlier) {
    const std::vector<double> xs{10, 12, 11, 13, 10, 14, 15, 11, 10, 12, 13, 12, 11, 10, 16, 12, 11, 13, 12, 30};
    expect_stats(xs, 20, 10, 30, 12.9, 12, 4.3516482055955716, 16.70000000000001, 27.339999999999982,
                 0.33733707020120707);
}

TEST(Stats, PercentileEndpointsAndErrors) {
    const std::vector<double> s{1, 2, 4};
    EXPECT_DOUBLE_EQ(percentile_sorted(s, 0.0), 1.0);
    EXPECT_DOUBLE_EQ(percentile_sorted(s, 1.0), 4.0);
    EXPECT_DOUBLE_EQ(percentile_sorted(s, 0.75), 3.0);  // h = 1.5 -> 2 + 0.5*2
    test::expect_error(ErrorCode::Config, [&] { (void)percentile_sorted(s, 1.01); });
    test::expect_error(ErrorCode::Config, [&] { (void)percentile_sorted(s, -0.1); });
    test::expect_error(ErrorCode::Config, [] { (void)percentile_sorted(std::vector<double>{}, 0.5); });
    test::expect_error(ErrorCode::Config, [] { (void)compute_sample_stats(std::vector<double>{}); });
    test::expect_error(ErrorCode::Config,
                       [] { (void)compute_sample_stats(std::vector<double>{1, std::nan("")}); });
}

TEST(Stats, StabilityClassifier) {
    std::vector<double> stable(20, 100.0);
    stable[0] = 101.0;  // cv ~ 0.002
    EXPECT_EQ(classify_stability(compute_sample_stats(stable)), Stability::Stable);

    std::vector<double> few(19, 100.0);
    EXPECT_EQ(classify_stability(compute_sample_stats(few)), Stability::TooFewSamples);

    std::vector<double> noisy(20, 100.0);
    for (std::size_t i = 0; i < noisy.size(); i += 2) noisy[i] = 120.0;  // cv ~ 0.095
    EXPECT_EQ(classify_stability(compute_sample_stats(noisy)), Stability::Unstable);

    StabilityPolicy loose;
    loose.max_cv = 0.2;
    EXPECT_EQ(classify_stability(compute_sample_stats(noisy), loose), Stability::Stable);

    SampleStats inf_cv = compute_sample_stats(std::vector<double>(20, 1.0));
    inf_cv.cv = std::numeric_limits<double>::infinity();
    EXPECT_EQ(classify_stability(inf_cv), Stability::Unstable);
    inf_cv.cv = std::nan("");
    EXPECT_EQ(classify_stability(inf_cv), Stability::Unstable);
    EXPECT_EQ(to_string(Stability::TooFewSamples), "too_few_samples");
}
