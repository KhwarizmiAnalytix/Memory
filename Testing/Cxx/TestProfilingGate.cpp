#include <gtest/gtest.h>

#include <thread>

#include "profiler/profiling_gate.h"

using memory::detail::profiling_gate;

// Plan 3.6 / R8: the profiler query takes a process-wide mutex, so the gate must
// keep it off the per-allocation path while profiling is off, and stay exact
// while a session is active.
class ProfilingGateTest : public ::testing::Test
{
protected:
    void SetUp() override { profiling_gate::reset_for_testing(); }
    void TearDown() override { profiling_gate::reset_for_testing(); }
};

TEST_F(ProfilingGateTest, IdleCallsQueryTheProfilerOnceEveryStride)
{
    int calls = 0;
    for (std::uint32_t i = 0; i < 10 * profiling_gate::kStride; ++i)
    {
        EXPECT_FALSE(profiling_gate::active([&] { ++calls; return false; }));
    }
    EXPECT_EQ(10, calls);
}

TEST_F(ProfilingGateTest, FirstCallOnAThreadAsksTheProfiler)
{
    int calls = 0;
    EXPECT_FALSE(profiling_gate::active([&] { ++calls; return false; }));
    EXPECT_EQ(1, calls);
}

TEST_F(ProfilingGateTest, SessionStartIsNoticedWithinOneStride)
{
    (void)profiling_gate::active([] { return false; });  // arm the countdown
    bool active  = false;
    int  noticed = -1;
    for (std::uint32_t i = 0; i < profiling_gate::kStride && !active; ++i)
    {
        active = profiling_gate::active([] { return true; });
        noticed = static_cast<int>(i);
    }
    EXPECT_TRUE(active);
    EXPECT_LT(noticed, static_cast<int>(profiling_gate::kStride));
}

TEST_F(ProfilingGateTest, ActiveSessionIsQueriedOnEveryCallAndEndsExactly)
{
    EXPECT_TRUE(profiling_gate::active([] { return true; }));
    int calls = 0;
    for (int i = 0; i < 100; ++i)
    {
        EXPECT_TRUE(profiling_gate::active([&] { ++calls; return true; }));
    }
    EXPECT_EQ(100, calls);  // exact while active: no event is skipped
    // The session ends: the very next call reports it.
    EXPECT_FALSE(profiling_gate::active([&] { ++calls; return false; }));
    EXPECT_EQ(101, calls);
}

TEST_F(ProfilingGateTest, EveryThreadStartsByAskingTheProfiler)
{
    int calls = 0;
    (void)profiling_gate::active([&] { ++calls; return false; });
    int other = 0;
    std::thread([&] { (void)profiling_gate::active([&] { ++other; return false; }); }).join();
    EXPECT_EQ(1, calls);
    EXPECT_EQ(1, other);
}
