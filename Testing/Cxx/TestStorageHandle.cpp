/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * P2 gate tests for storage_handle and the data_ptr → retained_ptr promotion path.
 * Plan §4.2, §4.3, P2.2–P2.4.
 */

#include <gtest/gtest.h>

#include <cstring>
#include <stdexcept>

#include "common/data_ptr.h"
#include "common/execution_context.h"
#include "common/retained_ptr.h"
#include "common/storage_handle.h"

namespace memory
{

// ---------------------------------------------------------------------------
// storage_handle: layout and basic contract
// ---------------------------------------------------------------------------

TEST(StorageHandle, SizeIs48Bytes)
{
    // Plan §4.2: exact 48-byte layout on 64-bit platforms.
    static_assert(sizeof(storage_handle) == 48,
                  "storage_handle must be exactly 48 bytes (plan §4.2)");
    EXPECT_EQ(sizeof(storage_handle), 48u);
}

TEST(StorageHandle, DefaultIsEmpty)
{
    storage_handle h;
    EXPECT_TRUE(h.empty());
    EXPECT_EQ(h.get(), nullptr);
    EXPECT_EQ(h.nbytes(), 0u);
    EXPECT_FALSE(h.id().valid());
}

TEST(StorageHandle, DeleterRunsOnce)
{
    int calls = 0;
    {
        auto del = [](void* ctx, void* ptr, std::size_t) noexcept {
            (*static_cast<int*>(ctx))++;
            ::operator delete(ptr);
        };
        void* mem = ::operator new(64);
        storage_handle h(mem, 64, del, &calls, device::cpu(), next_allocation_id());
        EXPECT_FALSE(h.empty());
    }  // destructor fires here
    EXPECT_EQ(calls, 1);
}

TEST(StorageHandle, MoveSourceDisarmed)
{
    int calls = 0;
    auto del  = [](void* ctx, void* ptr, std::size_t) noexcept {
        (*static_cast<int*>(ctx))++;
        ::operator delete(ptr);
    };
    void* mem = ::operator new(32);
    storage_handle src(mem, 32, del, &calls, device::cpu(), next_allocation_id());
    {
        storage_handle dst(std::move(src));  // src is now empty
        EXPECT_TRUE(src.empty());
        EXPECT_FALSE(dst.empty());
    }
    // Only dst's destructor should have called the deleter.
    EXPECT_EQ(calls, 1);
}

TEST(StorageHandle, MoveAssignSourceDisarmed)
{
    int calls = 0;
    auto del  = [](void* ctx, void* ptr, std::size_t) noexcept {
        (*static_cast<int*>(ctx))++;
        ::operator delete(ptr);
    };
    void* mem = ::operator new(16);
    storage_handle src(mem, 16, del, &calls, device::cpu(), next_allocation_id());
    storage_handle dst;
    dst = std::move(src);
    EXPECT_TRUE(src.empty());
    EXPECT_FALSE(dst.empty());
    // No destructor has fired yet.
    EXPECT_EQ(calls, 0);
}

TEST(StorageHandle, ReleasedDisarmsDestructor)
{
    int calls = 0;
    auto del  = [](void* ctx, void* ptr, std::size_t) noexcept {
        (*static_cast<int*>(ctx))++;
        ::operator delete(ptr);
    };
    void* mem = ::operator new(8);
    {
        storage_handle h(mem, 8, del, &calls, device::cpu(), next_allocation_id());
        void* raw = h.release();
        EXPECT_EQ(raw, mem);
        EXPECT_TRUE(h.empty());
    }
    EXPECT_EQ(calls, 0);  // destructor was disarmed
    ::operator delete(mem);
}

// ---------------------------------------------------------------------------
// adopt_bytes: validation and deleter wiring
// ---------------------------------------------------------------------------

TEST(StorageHandle, AdoptBytesDeleterRuns)
{
    int calls = 0;
    auto del  = [](void* ctx, void* ptr, std::size_t) noexcept {
        (*static_cast<int*>(ctx))++;
        ::operator delete(ptr);
    };
    void* mem = ::operator new(128);
    {
        storage_handle h = adopt_bytes(mem, 128, device::cpu(), del, &calls);
        EXPECT_EQ(h.get(), mem);
        EXPECT_EQ(h.nbytes(), 128u);
    }
    EXPECT_EQ(calls, 1);
}

TEST(StorageHandle, AdoptBytesNullPtrThrows)
{
    auto del = [](void*, void*, std::size_t) noexcept {};
    EXPECT_THROW(adopt_bytes(nullptr, 64, device::cpu(), del, nullptr),
                 std::invalid_argument);
}

TEST(StorageHandle, AdoptBytesZeroNbytesThrows)
{
    void* mem = ::operator new(8);
    auto  del = [](void*, void*, std::size_t) noexcept {};
    EXPECT_THROW(adopt_bytes(mem, 0, device::cpu(), del, nullptr),
                 std::invalid_argument);
    ::operator delete(mem);
}

TEST(StorageHandle, AdoptBytesNullDeleterThrows)
{
    void* mem = ::operator new(8);
    EXPECT_THROW(adopt_bytes(mem, 8, device::cpu(), nullptr, nullptr),
                 std::invalid_argument);
    ::operator delete(mem);
}

// ---------------------------------------------------------------------------
// allocate_bytes (CPU path): round-trip alloc/free via storage_handle
// ---------------------------------------------------------------------------

TEST(StorageHandle, AllocateBytesCPU)
{
    execution_context ctx = execution_context::cpu();
    storage_handle    h   = allocate_bytes(256, alignof(std::max_align_t), ctx);
    EXPECT_FALSE(h.empty());
    EXPECT_EQ(h.nbytes(), 256u);
    EXPECT_EQ(h.dev().type, device_enum::CPU);
    EXPECT_TRUE(h.id().valid());
    // Handle destructs here and calls cpu_free_fn via deleter_.
}

TEST(StorageHandle, AllocateBytesZeroReturnsEmpty)
{
    storage_handle h = allocate_bytes(0, alignof(std::max_align_t), execution_context::cpu());
    EXPECT_TRUE(h.empty());
}

TEST(StorageHandle, AllocateBytesAssignsUniqueIds)
{
    auto h1 = allocate_bytes(64, alignof(std::max_align_t), execution_context::cpu());
    auto h2 = allocate_bytes(64, alignof(std::max_align_t), execution_context::cpu());
    EXPECT_NE(h1.id().value, h2.id().value);
}

// ---------------------------------------------------------------------------
// data_ptr CPU allocation: storage_handle-backed
// ---------------------------------------------------------------------------

TEST(DataPtrStorage, CpuAllocationRoundTrip)
{
    data_ptr<float> dp(16, device_enum::CPU);
    EXPECT_EQ(dp.size(), 16u);
    EXPECT_NE(dp.data(), nullptr);
    EXPECT_EQ(dp.device(), device_enum::CPU);
    EXPECT_TRUE(dp.id().valid());
}

TEST(DataPtrStorage, MoveEmptiesSource)
{
    data_ptr<int> src(8, device_enum::CPU);
    EXPECT_NE(src.data(), nullptr);
    data_ptr<int> dst(std::move(src));
    EXPECT_EQ(src.data(), nullptr);
    EXPECT_EQ(src.size(), 0u);
    EXPECT_NE(dst.data(), nullptr);
    EXPECT_EQ(dst.size(), 8u);
}

// ---------------------------------------------------------------------------
// retained_ptr promotion: data_ptr<T> → retained_ptr<T>
// ---------------------------------------------------------------------------

TEST(RetainedPtrPromotion, PromotionTransfersOwnership)
{
    data_ptr<float> dp(32, device_enum::CPU);
    float*          raw = dp.data();
    ASSERT_NE(raw, nullptr);

    retained_ptr<float> rp(std::move(dp));

    // dp is empty after promotion.
    EXPECT_EQ(dp.data(), nullptr);
    EXPECT_EQ(dp.size(), 0u);

    // rp sees the same allocation.
    EXPECT_EQ(rp.data(), raw);
    EXPECT_EQ(rp.size(), 32u);
    EXPECT_EQ(rp.use_count(), 1);
    EXPECT_TRUE(rp.identity().valid());
}

TEST(RetainedPtrPromotion, PromotedPtrSharesOwnership)
{
    data_ptr<int>   dp(10, device_enum::CPU);
    retained_ptr<int> rp1(std::move(dp));
    retained_ptr<int> rp2 = rp1;  // copy increments ref count

    EXPECT_EQ(rp1.use_count(), 2);
    EXPECT_EQ(rp2.use_count(), 2);
    EXPECT_EQ(rp1.data(), rp2.data());
}

TEST(RetainedPtrPromotion, PromotedPtrSliceSharesBlock)
{
    data_ptr<int>   dp(20, device_enum::CPU);
    int*            raw = dp.data();
    retained_ptr<int> rp(std::move(dp));
    retained_ptr<int> sl = rp.slice(5, 10);

    EXPECT_EQ(rp.use_count(), 2);
    EXPECT_EQ(sl.data(), raw + 5);
    EXPECT_EQ(sl.size(), 10u);
    // base() always points to the original allocation base.
    EXPECT_EQ(sl.base(), raw);
    EXPECT_EQ(sl.base(), rp.base());
}

TEST(RetainedPtrPromotion, EmptyDataPtrPromotesEmpty)
{
    data_ptr<float>   dp;
    retained_ptr<float> rp(std::move(dp));
    EXPECT_TRUE(rp.empty());
    EXPECT_EQ(rp.use_count(), 0);
}

TEST(RetainedPtrPromotion, IdentityPreservedAcrossPromotion)
{
    data_ptr<double> dp(4, device_enum::CPU);
    allocation_id    orig_id = dp.id();
    retained_ptr<double> rp(std::move(dp));
    EXPECT_EQ(rp.identity().alloc_id, orig_id.value);
}

}  // namespace memory
