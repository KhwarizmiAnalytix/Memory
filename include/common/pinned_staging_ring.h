/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

#include <chrono>
#include <cstddef>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "common/copy_token.h"
#include "common/pinned_buffer.h"

namespace memory
{

/// Fixed ring of pinned host staging slots for overlapping host work with
/// transfers (plan 6.4).
///
/// A slot is IDLE, ACQUIRED (the caller is filling it), IN FLIGHT (a transfer that
/// reads or writes it was submitted) or QUARANTINED (that transfer failed, so its
/// state is unknown and the slot is never reused). A slot returns to idle only
/// when the token the caller submitted for it reports complete: reuse is decided
/// by the transfer's own completion, never by elapsed time or by the stream.
///
///   pinned_staging_ring<float> ring(3, 1 << 20);
///   auto slot = ring.acquire(std::chrono::seconds(1));   // waits, or throws
///   fill(slot.data(), slot.size());
///   ring.submit(slot, pinned.copy_to_device_async(...));  // token of the transfer
///
/// `try_acquire()` is the fail-fast form (no slot idle: returns nullopt);
/// `acquire(timeout)` polls for completions until a slot frees up and throws
/// std::runtime_error on timeout, or at once when every slot is quarantined.
///
/// Thread-safe. The destructor waits for in-flight transfers; if one cannot be
/// proven complete its slot's pinned memory is deliberately leaked rather than
/// recycled under a running transfer.
template <typename T>
class pinned_staging_ring
{
public:
    class slot
    {
    public:
        T*          data() const noexcept { return data_; }
        std::size_t size() const noexcept { return size_; }
        std::size_t index() const noexcept { return index_; }

    private:
        friend class pinned_staging_ring;
        slot(T* data, std::size_t size, std::size_t index) noexcept
            : data_(data), size_(size), index_(index)
        {
        }
        T*          data_;
        std::size_t size_;
        std::size_t index_;
    };

    pinned_staging_ring(std::size_t slot_count, std::size_t elements_per_slot, int device = 0)
    {
        if (slot_count == 0 || elements_per_slot == 0)
        {
            throw std::invalid_argument("pinned_staging_ring: slots and slot size must be non-zero");
        }
        slots_.reserve(slot_count);
        for (std::size_t i = 0; i < slot_count; ++i)
        {
            slots_.emplace_back(elements_per_slot, device);
        }
        elements_ = elements_per_slot;
    }

    ~pinned_staging_ring()
    {
        for (auto& entry : slots_)
        {
            if (entry.state != slot_state::in_flight)
            {
                continue;
            }
            bool safe = false;
            try
            {
                entry.token.wait();
                safe = true;
            }
            catch (...)  // NOLINT(bugprone-empty-catch)
            {
                // Completion could not be proven: leak the buffer below.
            }
            if (!safe)
            {
                leak(entry);
            }
        }
        for (auto& entry : slots_)
        {
            if (entry.state == slot_state::quarantined)
            {
                leak(entry);
            }
        }
    }

    pinned_staging_ring(pinned_staging_ring const&)            = delete;
    pinned_staging_ring& operator=(pinned_staging_ring const&) = delete;

    /// An idle slot, or nullopt when every slot is acquired, in flight or
    /// quarantined. Never waits.
    std::optional<slot> try_acquire()
    {
        std::vector<copy_token> released;  // destroyed after the lock is dropped
        std::optional<slot>     found;
        {
            std::lock_guard<std::mutex> const lock(mutex_);
            refresh_locked(released);
            for (std::size_t i = 0; i < slots_.size(); ++i)
            {
                if (slots_[i].state == slot_state::idle)
                {
                    slots_[i].state = slot_state::acquired;
                    found.emplace(slot(slots_[i].buffer.data(), elements_, i));
                    break;
                }
            }
        }
        return found;
    }

    /// Waits up to @p timeout for a slot. Throws std::runtime_error on timeout, or
    /// immediately when no slot can ever become idle (all quarantined).
    slot acquire(std::chrono::milliseconds timeout)
    {
        auto const deadline = std::chrono::steady_clock::now() + timeout;
        while (true)
        {
            if (auto got = try_acquire())
            {
                return *got;
            }
            if (quarantined_count() == slots_.size())
            {
                throw std::runtime_error("pinned_staging_ring: every slot is quarantined");
            }
            if (std::chrono::steady_clock::now() >= deadline)
            {
                throw std::runtime_error("pinned_staging_ring: timed out waiting for a free slot");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    /// Mark an acquired slot as in flight until @p token completes. A token that is
    /// already complete frees the slot at once. Throws std::invalid_argument if the
    /// slot is not currently acquired (double submit, foreign slot).
    void submit(slot const& s, copy_token token)
    {
        copy_token                  displaced;
        std::lock_guard<std::mutex> const lock(mutex_);
        auto&                       entry = checked_acquired(s);
        if (token.ready())
        {
            entry.state = slot_state::idle;
            displaced    = std::move(token);
            return;
        }
        entry.token  = std::move(token);
        entry.state = slot_state::in_flight;
    }

    /// Give an acquired slot back without submitting a transfer for it.
    void release(slot const& s)
    {
        std::lock_guard<std::mutex> const lock(mutex_);
        checked_acquired(s).state = slot_state::idle;
    }

    std::size_t slot_count() const noexcept { return slots_.size(); }
    std::size_t slot_elements() const noexcept { return elements_; }
    std::size_t idle_count() { return count(slot_state::idle); }
    std::size_t in_flight_count() { return count(slot_state::in_flight); }
    std::size_t quarantined_count() { return count(slot_state::quarantined); }

private:
    enum class slot_state : unsigned char
    {
        idle,
        acquired,
        in_flight,
        quarantined
    };

    struct entry
    {
        entry(std::size_t elements, int device) : buffer(elements, device) {}
        pinned_buffer<T> buffer;
        copy_token       token;
        slot_state       state{slot_state::idle};
    };

    std::vector<entry> slots_;
    std::size_t        elements_{0};
    std::mutex         mutex_;

    entry& checked_acquired(slot const& s)
    {
        if (s.index() >= slots_.size() || slots_[s.index()].state != slot_state::acquired ||
            slots_[s.index()].buffer.data() != s.data())
        {
            throw std::invalid_argument("pinned_staging_ring: slot is not currently acquired");
        }
        return slots_[s.index()];
    }

    // Move finished transfers' slots back to idle; quarantine failed ones.
    void refresh_locked(std::vector<copy_token>& released)
    {
        for (auto& e : slots_)
        {
            if (e.state != slot_state::in_flight)
            {
                continue;
            }
            completion_state const state = e.token.state();
            if (state == completion_state::complete)
            {
                released.push_back(std::move(e.token));
                e.token  = copy_token{};
                e.state = slot_state::idle;
            }
            else if (state == completion_state::failed)
            {
                e.state = slot_state::quarantined;  // the token stays: its state is unknown
            }
        }
    }

    std::size_t count(slot_state wanted)
    {
        std::vector<copy_token> released;
        std::size_t             n = 0;
        {
            std::lock_guard<std::mutex> const lock(mutex_);
            refresh_locked(released);
            for (auto const& e : slots_)
            {
                n += (e.state == wanted) ? 1U : 0U;
            }
        }
        return n;
    }

    static void leak(entry& e)
    {
        // Never recycle memory a transfer of unknown state may still touch.
        (void)new pinned_buffer<T>(std::move(e.buffer));
    }
};

}  // namespace memory
