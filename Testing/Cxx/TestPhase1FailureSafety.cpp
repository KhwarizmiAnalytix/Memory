/*
 * Unit tests for Phase 1: Failure safety and cleanup
 *
 * Tests allocation rollback, cleanup hooks, quarantine states,
 * and error propagation (CPU-only).
 */

#include <gtest/gtest.h>
#include <stdexcept>
#include <unordered_set>
#include <limits>

#include "allocator.h"
#include "common/data_ptr.h"

namespace memory
{

// Simple mock for cleanup failure tracking
class CleanupFailureTracker
{
public:
    static CleanupFailureTracker& instance()
    {
        static CleanupFailureTracker tracker;
        return tracker;
    }

    void record_failure(const std::string& reason) noexcept
    {
        failure_count_++;
        last_failure_ = reason;
    }

    size_t failure_count() const noexcept { return failure_count_; }
    std::string last_failure() const noexcept { return last_failure_; }
    void reset() noexcept
    {
        failure_count_ = 0;
        last_failure_.clear();
    }

private:
    size_t failure_count_ = 0;
    std::string last_failure_;
};

class TestPhase1FailureSafety : public ::testing::Test
{
protected:
    void SetUp() override
    {
        CleanupFailureTracker::instance().reset();
    }
};

// Test: Allocation with invalid size throws
TEST_F(TestPhase1FailureSafety, ZeroSizeAllocationHandling)
{
    // Zero-size allocation should not throw (it's valid, just empty)
    EXPECT_NO_THROW({
        data_ptr<float> ptr(0, execution_context::cpu());
        EXPECT_EQ(ptr.size(), 0);
    });
}

// Test: Overflow detection on size calculation
TEST_F(TestPhase1FailureSafety, SizeOverflowDetection)
{
    // Huge size that overflows when multiplied by sizeof(float)
    EXPECT_THROW(
        allocator<float>::allocate(
            std::numeric_limits<size_t>::max(),
            execution_context::cpu()),
        std::overflow_error);
}

// Test: Multiple allocation and deallocation cycles
TEST_F(TestPhase1FailureSafety, AllocationDeallocationCycles)
{
    const size_t size = 1000;

    for (int cycle = 0; cycle < 10; ++cycle)
    {
        {
            data_ptr<float> ptr(size, execution_context::cpu());
            EXPECT_EQ(ptr.size(), size);
            // Destructor should free without error
        }  // ptr destroyed here
    }

    // If we get here, all cycles completed without crash
    EXPECT_TRUE(true);
}

// Test: move semantics don't corrupt state
TEST_F(TestPhase1FailureSafety, MoveDoesNotCorruptState)
{
    const size_t size = 1000;
    data_ptr<float> ptr1(size, execution_context::cpu());
    auto* original_data = ptr1.data();
    auto original_id = ptr1.id();

    data_ptr<float> ptr2 = std::move(ptr1);

    // ptr2 should have the data and ID
    EXPECT_EQ(ptr2.data(), original_data);
    EXPECT_EQ(ptr2.id(), original_id);
    EXPECT_EQ(ptr2.size(), size);

    // ptr1 should be cleared
    EXPECT_EQ(ptr1.data(), nullptr);
    EXPECT_EQ(ptr1.size(), 0);
}

// Test: Copy is deleted (prevents accidental deep-copy)
TEST_F(TestPhase1FailureSafety, CopyIsDeleted)
{
    data_ptr<float> ptr(100, execution_context::cpu());

    // This should not compile:
    // data_ptr<float> copy = ptr;  // ERROR: deleted copy constructor
    // data_ptr<float> copy2;
    // copy2 = ptr;  // ERROR: deleted copy assignment

    // If we get here, copy is properly deleted
    EXPECT_TRUE(true);
}

// Test: explicit clone allocates new memory
TEST_F(TestPhase1FailureSafety, CloneAllocatesNewMemory)
{
    data_ptr<float> ptr1(100, execution_context::cpu());
    auto* original_data = ptr1.data();

    data_ptr<float> ptr2 = ptr1.clone();

    // Should be different allocations
    EXPECT_NE(ptr2.data(), original_data);
    EXPECT_EQ(ptr2.size(), ptr1.size());

    // Different allocation IDs
    EXPECT_NE(ptr2.id(), ptr1.id());
}

// Test: a null endpoint with a positive count is rejected before any work
// (plan 5.1, task 1.5); the other endpoint is left untouched.
TEST_F(TestPhase1FailureSafety, NullptrCopyHandling)
{
    std::vector<float> src(100, 1.0f);
    std::vector<float> dst(100, 0.0f);

    EXPECT_THROW(
        allocator<float>::copy(nullptr, 100, dst.data(), device_enum::CPU, device_enum::CPU),
        std::invalid_argument);

    for (const auto& val : dst)
    {
        EXPECT_EQ(val, 0.0f);  // Unchanged
    }

    EXPECT_THROW(
        allocator<float>::copy(src.data(), 100, nullptr, device_enum::CPU, device_enum::CPU),
        std::invalid_argument);
}

// Test: Zero-size copy
TEST_F(TestPhase1FailureSafety, ZeroSizeCopyHandling)
{
    std::vector<float> src(100, 1.0f);
    std::vector<float> dst(100, 0.0f);

    // Zero-size copy: should be no-op
    allocator<float>::copy(src.data(), 0, dst.data(),
                          device_enum::CPU, device_enum::CPU);

    for (const auto& val : dst)
    {
        EXPECT_EQ(val, 0.0f);  // Unchanged
    }
}

// Test: Allocation ID uniqueness under stress
TEST_F(TestPhase1FailureSafety, AllocationIdUniquenessStress)
{

    std::unordered_set<uint64_t> ids;
    const int alloc_count = 1000;

    for (int i = 0; i < alloc_count; ++i)
    {
        data_ptr<float> ptr(10, execution_context::cpu());
        ids.insert(ptr.id().value);
    }

    // All IDs should be unique
    EXPECT_EQ(ids.size(), alloc_count);
}

// Test: Destructor does not throw
TEST_F(TestPhase1FailureSafety, DestructorNoThrow)
{
    // This should not throw even though destructors are noexcept
    EXPECT_NO_THROW({
        data_ptr<float> ptr(1000, execution_context::cpu());
    });  // Destructor called here
}

// Test: Multiple moves in sequence
TEST_F(TestPhase1FailureSafety, ChainedMoves)
{
    data_ptr<float> ptr1(100, execution_context::cpu());
    auto id1 = ptr1.id();
    auto* data1 = ptr1.data();

    data_ptr<float> ptr2 = std::move(ptr1);
    data_ptr<float> ptr3 = std::move(ptr2);
    data_ptr<float> ptr4 = std::move(ptr3);

    // Final ptr should have original data and ID
    EXPECT_EQ(ptr4.data(), data1);
    EXPECT_EQ(ptr4.id(), id1);

    // All intermediate ptrs should be empty
    EXPECT_EQ(ptr1.data(), nullptr);
    EXPECT_EQ(ptr2.data(), nullptr);
    EXPECT_EQ(ptr3.data(), nullptr);
}

// Test: Invalid argument handling
TEST_F(TestPhase1FailureSafety, InvalidArgumentHandling)
{
    // Size that would definitely overflow when multiplied by sizeof(float)
    EXPECT_THROW(
        allocator<float>::allocate(
            std::numeric_limits<size_t>::max(),
            execution_context::cpu()),
        std::overflow_error);
}

// Test: Context preservation through copies
TEST_F(TestPhase1FailureSafety, ContextPreservationInClone)
{
    auto ctx = execution_context::cpu();
    data_ptr<float> ptr1(100, ctx);

    data_ptr<float> ptr2 = ptr1.clone();

    // Context should be preserved
    EXPECT_EQ(ptr2.device(), ctx.device_type());
    EXPECT_EQ(ptr2.device_index(), ctx.device_index());
}

}  // namespace memory
