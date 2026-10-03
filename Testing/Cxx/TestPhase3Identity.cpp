/*
 * Unit tests for Phase 3: Allocation identity and storage contracts
 *
 * Tests allocation ID generation, tracking, and preservation through
 * ownership transfers and slicing.
 */

#include <gtest/gtest.h>
#include <unordered_set>

#include "common/data_ptr.h"
#include "common/storage_identity.h"

namespace memory
{

class TestPhase3Identity : public ::testing::Test
{
protected:
    static constexpr size_t size_floats = 1000;
};

// Test: allocation_id default construction is invalid (zero)
TEST_F(TestPhase3Identity, AllocationIdDefaultInvalid)
{
    allocation_id id;
    EXPECT_FALSE(id.valid());
    EXPECT_EQ(id.value, 0);
}

// Test: allocation_id explicit construction
TEST_F(TestPhase3Identity, AllocationIdExplicitConstruction)
{
    allocation_id id(42);
    EXPECT_TRUE(id.valid());
    EXPECT_EQ(id.value, 42);
}

// Test: allocation_id equality
TEST_F(TestPhase3Identity, AllocationIdEquality)
{
    allocation_id id1(10);
    allocation_id id2(10);
    allocation_id id3(20);

    EXPECT_EQ(id1, id2);
    EXPECT_NE(id1, id3);
}

// Test: allocation_id ordering
TEST_F(TestPhase3Identity, AllocationIdOrdering)
{
    allocation_id id1(10);
    allocation_id id2(20);

    EXPECT_LT(id1, id2);
    EXPECT_LE(id1, id2);
    EXPECT_GT(id2, id1);
    EXPECT_GE(id2, id1);
}

// Test: next_allocation_id() generates unique IDs
TEST_F(TestPhase3Identity, NextAllocationIdUnique)
{

    allocation_id id1 = next_allocation_id();
    allocation_id id2 = next_allocation_id();
    allocation_id id3 = next_allocation_id();

    EXPECT_NE(id1, id2);
    EXPECT_NE(id2, id3);
    EXPECT_NE(id1, id3);

    // Should skip 0 (invalid)
    EXPECT_TRUE(id1.valid());
    EXPECT_TRUE(id2.valid());
    EXPECT_TRUE(id3.valid());
}

// Test: next_allocation_id() is monotonic
TEST_F(TestPhase3Identity, NextAllocationIdMonotonic)
{

    allocation_id id1 = next_allocation_id();
    allocation_id id2 = next_allocation_id();
    allocation_id id3 = next_allocation_id();

    EXPECT_LT(id1.value, id2.value);
    EXPECT_LT(id2.value, id3.value);
}

// Test: data_ptr has allocation ID
TEST_F(TestPhase3Identity, DataPtrHasAllocationId)
{
    data_ptr<float> ptr(size_floats, execution_context::cpu());

    allocation_id id = ptr.id();
    EXPECT_TRUE(id.valid());
}

// Test: each allocation gets unique ID
TEST_F(TestPhase3Identity, EachAllocationUnique)
{

    data_ptr<float> ptr1(100, execution_context::cpu());
    data_ptr<float> ptr2(200, execution_context::cpu());
    data_ptr<float> ptr3(300, execution_context::cpu());

    EXPECT_NE(ptr1.id(), ptr2.id());
    EXPECT_NE(ptr2.id(), ptr3.id());
    EXPECT_NE(ptr1.id(), ptr3.id());
}

// Test: ID survives move
TEST_F(TestPhase3Identity, IdSurvivesMove)
{
    data_ptr<float> ptr1(size_floats, execution_context::cpu());
    allocation_id original_id = ptr1.id();

    data_ptr<float> ptr2 = std::move(ptr1);

    EXPECT_EQ(ptr2.id(), original_id);
}

// Task 1.9: empty storage has no allocation lifetime, hence no valid ID.
TEST_F(TestPhase3Identity, ZeroSizeAllocationHasInvalidId)
{
    data_ptr<float> ptr(0, execution_context::cpu());

    EXPECT_FALSE(ptr.id().valid());
    EXPECT_EQ(ptr.size(), 0U);
    EXPECT_EQ(ptr.data(), nullptr);
    EXPECT_EQ(ptr.device(), device_enum::CPU);  // zero-size still knows its device
}

TEST_F(TestPhase3Identity, DefaultAndMovedFromHandlesHaveInvalidIds)
{
    data_ptr<float> empty;
    EXPECT_FALSE(empty.id().valid());

    data_ptr<float> source(16, execution_context::cpu());
    allocation_id   id = source.id();
    ASSERT_TRUE(id.valid());
    data_ptr<float> target = std::move(source);
    EXPECT_FALSE(source.id().valid());  // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(target.id(), id);

    data_ptr<float> assigned;
    assigned = std::move(target);
    EXPECT_FALSE(target.id().valid());  // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(assigned.id(), id);

    retained_ptr<float> shared(std::move(assigned));
    EXPECT_FALSE(assigned.id().valid());  // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(shared.identity().alloc_id, id.value);
    retained_ptr<float> moved = std::move(shared);
    EXPECT_FALSE(shared.identity().valid());  // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(moved.identity().alloc_id, id.value);

    retained_ptr<float> promoted_empty{data_ptr<float>{}};
    EXPECT_FALSE(promoted_empty.identity().valid());
}

// One allocation lifetime has one ID: slices of slices keep the owner's ID and base.
TEST_F(TestPhase3Identity, NestedSlicesCarryTheOwnersId)
{
    data_ptr<float>     owner(64, execution_context::cpu());
    uint64_t const      id = owner.id().value;
    retained_ptr<float> whole(std::move(owner));
    auto                outer = whole.slice(8, 40);
    auto                inner = outer.slice(4, 10);
    auto                leaf  = inner.slice(2, 3);
    EXPECT_EQ(outer.identity().alloc_id, id);
    EXPECT_EQ(inner.identity().alloc_id, id);
    EXPECT_EQ(leaf.identity().alloc_id, id);
    EXPECT_EQ(leaf.base(), whole.base());
    EXPECT_EQ(leaf.data(), whole.data() + 8 + 4 + 2);
    EXPECT_EQ(leaf.size(), 3U);
    EXPECT_EQ(whole.use_count(), 4);
}

// A new allocation at a recycled address is a new allocation lifetime.
TEST_F(TestPhase3Identity, AddressReuseGetsANewId)
{
    allocation_id previous;
    bool          saw_reuse = false;
    void*         previous_address = nullptr;
    for (int i = 0; i < 64; ++i)
    {
        data_ptr<float> block(32, execution_context::cpu());
        EXPECT_NE(block.id(), previous);
        if (block.data() == previous_address)
        {
            saw_reuse = true;
        }
        previous         = block.id();
        previous_address = block.data();
    }
    RecordProperty("address_reuse_observed", saw_reuse ? 1 : 0);
}

// Test: clone gets different ID
TEST_F(TestPhase3Identity, CloneGetsDifferentId)
{

    data_ptr<float> ptr1(size_floats, execution_context::cpu());
    allocation_id id1 = ptr1.id();

    data_ptr<float> ptr2 = ptr1.clone();
    allocation_id id2 = ptr2.id();

    // Different allocations = different IDs
    EXPECT_NE(id1, id2);
}

// Test: storage_identity with allocation_id
TEST_F(TestPhase3Identity, StorageIdentityIntegration)
{

    allocation_id id = next_allocation_id();
    void* ptr = reinterpret_cast<void*>(0x12345678);
    size_t cap = 1000;

    // Legacy API still works (backward compatibility)
    uint64_t next_legacy_id = storage_identity::next_id();
    EXPECT_NE(next_legacy_id, 0);
}

// Test: allocation_id can be used in hash table
TEST_F(TestPhase3Identity, AllocationIdHashable)
{

    std::unordered_set<allocation_id> ids;

    allocation_id id1 = next_allocation_id();
    allocation_id id2 = next_allocation_id();
    allocation_id id3 = next_allocation_id();

    ids.insert(id1);
    ids.insert(id2);
    ids.insert(id3);

    EXPECT_EQ(ids.size(), 3);
    EXPECT_TRUE(ids.count(id1));
    EXPECT_TRUE(ids.count(id2));
    EXPECT_TRUE(ids.count(id3));
    EXPECT_FALSE(ids.count(allocation_id(999)));
}

// Test: invalid ID (zero) works correctly
TEST_F(TestPhase3Identity, InvalidIdHandling)
{
    allocation_id invalid;
    allocation_id valid(1);

    EXPECT_FALSE(invalid.valid());
    EXPECT_TRUE(valid.valid());
    EXPECT_LT(invalid, valid);
}

// Test: multiple allocations and tracking
TEST_F(TestPhase3Identity, MultipleAllocationsTracking)
{

    std::vector<data_ptr<float>> ptrs;
    std::unordered_set<allocation_id> ids;

    // Create 10 allocations
    for (int i = 0; i < 10; ++i)
    {
        ptrs.emplace_back(100 + i * 10, execution_context::cpu());
        ids.insert(ptrs.back().id());
    }

    // All IDs should be unique
    EXPECT_EQ(ids.size(), 10);

    // Each ptr's ID should be in the set
    for (const auto& ptr : ptrs)
    {
        EXPECT_TRUE(ids.count(ptr.id()));
    }
}

}  // namespace memory

namespace memory
{
// Task 1.9 / A6: one generator per process. IDs minted inside the Memory library
// (data_ptr allocation) and IDs minted by header-inline code compiled into the
// client (retained_ptr adoption, next_allocation_id) share one sequence.
TEST_F(TestPhase3Identity, OneGeneratorAcrossTheLibraryBoundary)
{
    std::unordered_set<uint64_t> ids;
    std::vector<data_ptr<float>> owners;
    uint64_t                     last = 0;
    for (int i = 0; i < 500; ++i)
    {
        owners.emplace_back(8, execution_context::cpu());  // minted in the library
        uint64_t const from_library = owners.back().id().value;
        auto           adopted      = retained_ptr<int>::adopt(
            new int[2]{}, 2, execution_context::cpu(),
            [](int* p, size_t, execution_context const&) { delete[] p; });
        uint64_t const from_client = adopted.identity().alloc_id;  // minted in this binary
        uint64_t const direct      = next_allocation_id().value;

        EXPECT_TRUE(ids.insert(from_library).second) << "duplicate id " << from_library;
        EXPECT_TRUE(ids.insert(from_client).second) << "duplicate id " << from_client;
        EXPECT_TRUE(ids.insert(direct).second) << "duplicate id " << direct;
        EXPECT_GT(from_library, last);
        EXPECT_GT(from_client, from_library);  // one monotonic sequence
        EXPECT_GT(direct, from_client);
        last = direct;
    }
}
}  // namespace memory
