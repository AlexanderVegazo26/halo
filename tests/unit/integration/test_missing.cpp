// Registered when a component the integration tests need is not part of this build, so
// the omission is a visible skip rather than a shorter test list.

#include <gtest/gtest.h>

TEST(Integration, ComponentsPresent) {
    GTEST_SKIP() << "integration tests not built: missing targets " << HALO_INTEGRATION_MISSING;
}
