#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <optional>
#include <x86intrin.h>

namespace darkpool {

template<typename T, size_t Size>
class LockFreeQueue {
    static_assert((Size & (Size - 1)) == 0, "Size must be power of 2");
    
    struct alignas(CACHE_LINE_SIZE) Cell {
        std::atomic<size_t> sequence;
        T data;
    };
    
public:
    LockFreeQueue() : buffer_(new Cell[Size]), mask_(Size - 1) {
        for (size_t i = 0; i < Size; ++i) {
            buffer_[i].sequence.store(i, std::memory_order_relaxed);
        }
        
        enqueue_pos_.store(0, std::memory_order_relaxed);
        dequeue_pos_.store(0, std::memory_order_relaxed);
    }
    
    ~LockFreeQueue() = default;
    
    bool enqueue(T&& item) noexcept {
        Cell* cell;
        size_t pos = enqueue_pos_.load(std::memory_order_relaxed);
        
        for (;;) {
            cell = &buffer_[pos & mask_];
            size_t seq = cell->sequence.load(std::memory_order_acquire);
            intptr_t dif = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);
            
            if (dif == 0) {
                if (enqueue_pos_.compare_exchange_weak(pos, pos + 1, 
                                                       std::memory_order_relaxed)) {
                    break;
                }
            } else if (dif < 0) {
                return false; // Queue full
            } else {
                pos = enqueue_pos_.load(std::memory_order_relaxed);
            }
        }
        
        cell->data = std::move(item);
        cell->sequence.store(pos + 1, std::memory_order_release);
        
        return true;
    }
    
    bool enqueue(const T& item) noexcept {
        return enqueue(T(item));
    }
    
    std::optional<T> dequeue() noexcept {
        Cell* cell;
        size_t pos = dequeue_pos_.load(std::memory_order_relaxed);
        
        for (;;) {
            cell = &buffer_[pos & mask_];
            size_t seq = cell->sequence.load(std::memory_order_acquire);
            intptr_t dif = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);
            
            if (dif == 0) {
                if (dequeue_pos_.compare_exchange_weak(pos, pos + 1,
                                                       std::memory_order_relaxed)) {
                    break;
                }
            } else if (dif < 0) {
                return std::nullopt; // Queue empty
            } else {
                pos = dequeue_pos_.load(std::memory_order_relaxed);
            }
        }
        
        T item = std::move(cell->data);
        cell->sequence.store(pos + mask_ + 1, std::memory_order_release);
        
        return item;
    }
    
    size_t size_approx() const noexcept {
        size_t enqueue = enqueue_pos_.load(std::memory_order_relaxed);
        size_t dequeue = dequeue_pos_.load(std::memory_order_relaxed);
        return (enqueue > dequeue) ? (enqueue - dequeue) : 0;
    }
    
    bool empty() const noexcept {
        return size_approx() == 0;
    }
    
    static constexpr size_t capacity() noexcept {
        return Size;
    }
    
private:
    std::unique_ptr<Cell[]> buffer_;
    const size_t mask_;
    
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> enqueue_pos_;
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> dequeue_pos_;
};

// Single Producer Single Consumer queue for better performance
template<typename T, size_t Size>
class SPSCQueue {
    static_assert((Size & (Size - 1)) == 0, "Size must be power of 2");
    
public:
    SPSCQueue() : buffer_(new T[Size]), mask_(Size - 1) {
        write_pos_.store(0, std::memory_order_relaxed);
        read_pos_.store(0, std::memory_order_relaxed);
    }
    
    bool push(T&& item) noexcept {
        const size_t write = write_pos_.load(std::memory_order_relaxed);
        const size_t next_write = (write + 1) & mask_;
        
        if (next_write == read_pos_.load(std::memory_order_acquire)) {
            return false; // Queue full
        }
        
        buffer_[write] = std::move(item);
        write_pos_.store(next_write, std::memory_order_release);
        
        return true;
    }
    
    std::optional<T> pop() noexcept {
        const size_t read = read_pos_.load(std::memory_order_relaxed);
        
        if (read == write_pos_.load(std::memory_order_acquire)) {
            return std::nullopt; // Queue empty
        }
        
        T item = std::move(buffer_[read]);
        read_pos_.store((read + 1) & mask_, std::memory_order_release);
        
        return item;
    }
    
    size_t size_approx() const noexcept {
        const size_t write = write_pos_.load(std::memory_order_relaxed);
        const size_t read = read_pos_.load(std::memory_order_relaxed);
        
        if (write >= read) {
            return write - read;
        } else {
            return Size - read + write;
        }
    }
    
    bool empty() const noexcept {
        return read_pos_.load(std::memory_order_relaxed) == 
               write_pos_.load(std::memory_order_relaxed);
    }
    
private:
    std::unique_ptr<T[]> buffer_;
    const size_t mask_;
    
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> write_pos_;
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> read_pos_;
};

}
