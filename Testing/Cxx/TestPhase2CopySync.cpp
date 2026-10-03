/*
 * Unit tests for Phase 2: copy_sync() completion and token state
 *
 * Tests CPU-only paths; GPU validation deferred to hardware testing.
 */

#include <gtest/gtest.h>

#include "allocator.h"
#include "common/copy_token.h"
#include "common/data_ptr.h"
#include "common/execution_context.h"

namespace memory
{

class TestPhase2CopySync : public ::testing::Test
{
protected:
    static constexpr size_t size_floats = 1000;
};

// Test: CPU→CPU copy is synchronous
TEST_F(TestPhase2CopySync, CpuCopySynchronous)
{
    std::vector<float> src(size_floats, 3.14f);
    std::vector<float> dst(size_floats, 0.0f);

    execution_context ctx_cpu = execution_context::cpu();

    // Synchronous copy should return with data visible immediately
    allocator<float>::copy_sync(
        src.data(), size_floats, dst.data(),
        device_enum::CPU, device_enum::CPU);

    // Verify data was copied
    for (size_t i = 0; i < size_floats; ++i)
    {
        EXPECT_EQ(dst[i], 3.14f);
    }
}

// Test: copy_token default construction is immediately complete
TEST_F(TestPhase2CopySync, DefaultTokenIsComplete)
{
    copy_token token;

    EXPECT_TRUE(token.ready());
    EXPECT_EQ(token.state(), completion_state::complete);

    // wait() should be no-op
    token.wait();  // Should not throw
}

// Test: copy_token state() query
TEST_F(TestPhase2CopySync, TokenStateQuery)
{
    copy_token token;

    // Default token should always be complete
    EXPECT_EQ(token.state(), completion_state::complete);
    EXPECT_TRUE(token.ready());

    // Multiple calls should return same state
    EXPECT_EQ(token.state(), completion_state::complete);
    EXPECT_TRUE(token.ready());
}

// Test: copy_token from CPU context is complete
TEST_F(TestPhase2CopySync, TokenFromCpuContextIsComplete)
{
    execution_context ctx = execution_context::cpu();
    copy_token token(ctx);

    EXPECT_TRUE(token.ready());
    EXPECT_EQ(token.state(), completion_state::complete);
    token.wait();  // Should not throw
}

// Test: copy_token with zero-size operation
TEST_F(TestPhase2CopySync, ZeroSizeCopy)
{
    std::vector<float> src(10, 1.0f);
    std::vector<float> dst(10, 0.0f);

    // Zero-size copy should succeed and be no-op
    allocator<float>::copy_sync(
        src.data(), 0, dst.data(),
        device_enum::CPU, device_enum::CPU);

    // Destination should be unchanged
    for (const auto& val : dst)
    {
        EXPECT_EQ(val, 0.0f);
    }
}

// Test: copy_token error handling
TEST_F(TestPhase2CopySync, TokenWaitWithCpuContext)
{
    copy_token token(execution_context::cpu());

    // wait() should be no-op for CPU and not throw
    EXPECT_NO_THROW(token.wait());
}

// Test: copy_sync rejects a null endpoint with a positive count (task 1.5)
TEST_F(TestPhase2CopySync, NullptrHandling)
{
    std::vector<float> dst(size_floats, 1.0f);

    EXPECT_THROW(
        allocator<float>::copy_sync(
            nullptr, size_floats, dst.data(), device_enum::CPU, device_enum::CPU),
        std::invalid_argument);

    // Destination should be unchanged
    for (const auto& val : dst)
    {
        EXPECT_EQ(val, 1.0f);
    }

    std::vector<float> src(size_floats, 2.0f);
    EXPECT_THROW(
        allocator<float>::copy_sync(
            src.data(), size_floats, nullptr, device_enum::CPU, device_enum::CPU),
        std::invalid_argument);
}

// Test: multiple ready() calls return consistent results
TEST_F(TestPhase2CopySync, TokenReadyIdempotent)
{
    copy_token token(execution_context::cpu());

    bool first_ready = token.ready();
    bool second_ready = token.ready();
    bool third_ready = token.ready();

    EXPECT_EQ(first_ready, second_ready);
    EXPECT_EQ(second_ready, third_ready);
    EXPECT_TRUE(first_ready);
}

// Test: token copying (tokens are copyable)
TEST_F(TestPhase2CopySync, TokenCopyable)
{
    copy_token token1(execution_context::cpu());
    copy_token token2 = token1;  // Copy assignment

    EXPECT_EQ(token2.state(), completion_state::complete);
    EXPECT_TRUE(token2.ready());

    copy_token token3;
    token3 = token1;  // Move assignment
    EXPECT_EQ(token3.state(), completion_state::complete);
}

// Test: token context accessor
TEST_F(TestPhase2CopySync, TokenContextAccessor)
{
    execution_context ctx = execution_context::cpu();
    copy_token token(ctx);

    EXPECT_EQ(token.ctx().device_type, device_enum::CPU);
}

// Test: large buffer copy
TEST_F(TestPhase2CopySync, LargeBufferCopy)
{
    const size_t large_size = 1000000;
    std::vector<float> src(large_size);
    std::vector<float> dst(large_size);

    // Fill source with sequential values
    for (size_t i = 0; i < large_size; ++i)
    {
        src[i] = static_cast<float>(i);
    }

    allocator<float>::copy_sync(
        src.data(), large_size, dst.data(),
        device_enum::CPU, device_enum::CPU);

    // Spot-check several values
    EXPECT_EQ(dst[0], 0.0f);
    EXPECT_EQ(dst[large_size / 2], large_size / 2);
    EXPECT_EQ(dst[large_size - 1], static_cast<float>(large_size - 1));
}

// Test: overflow detection in copy size
TEST_F(TestPhase2CopySync, SizeOverflowDetection)
{
    std::vector<float> src(10, 1.0f);
    std::vector<float> dst(10, 0.0f);

    // Huge count that overflows when multiplied by sizeof(float)
    // This should throw std::overflow_error
    EXPECT_THROW(
        allocator<float>::copy_sync(
            src.data(),
            std::numeric_limits<size_t>::max(),  // Will overflow
            dst.data(),
            device_enum::CPU, device_enum::CPU),
        std::overflow_error);
}

}  // namespace memory
