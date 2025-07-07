#pragma once

#include <atomic>
#include <cstring>
#include <memory>
#include <vector>
#include <immintrin.h>

namespace darkpool {

// Disruptor-style ring buffer for ultra-low latency
template<typename T>
class RingBuffer {
    static constexpr size_t CACHE_LINE_SIZE = 64;
    
public:
    explicit RingBuffer(size_t size) 
        : size_(next_power_of_two(size)),
          mask_(size_ - 1),
          buffer_(allocate_aligned_buffer()) {
        
        cursor_.store(-1, std::memory_order_relaxed);
        cached_gating_sequence_.store(-1, std::memory_order_relaxed);
    }
    
    // Single producer claim
    int64_t claim_next() noexcept {
        return claim(1);
    }
    
    int64_t claim(size_t n) noexcept {
        const int64_t current = cursor_.load(std::memory_order_relaxed);
        const int64_t next = current + n;
        
        const int64_t wrap_point = next - size_;
        const int64_t cached_gating = cached_gating_sequence_.load(std::memory_order_relaxed);
        
        if (wrap_point > cached_gating || cached_gating > current) {
            int64_t min_sequence = get_minimum_sequence(current);
            cached_gating_sequence_.store(min_sequence, std::memory_order_relaxed);
            
            if (wrap_point > min_sequence) {
                // Buffer full, would need to wait
                return -1;
            }
        }
        
        cursor_.store(next, std::memory_order_relaxed);
        return next;
    }
    
    void publish(int64_t sequence) noexcept {
        // Memory barrier to ensure data is written before publishing
        std::atomic_thread_fence(std::memory_order_release);
    }
    
    T& operator[](int64_t sequence) noexcept {
        return buffer_[sequence & mask_];
    }
    
    const T& operator[](int64_t sequence) const noexcept {
        return buffer_[sequence & mask_];
    }
    
    // Consumer sequence tracking
    class Sequence {
    public:
        Sequence() : value_(-1) {}
        
        int64_t get() const noexcept {
            return value_.load(std::memory_order_acquire);
        }
        
        void set(int64_t value) noexcept {
            value_.store(value, std::memory_order_release);
        }
        
        bool compare_and_set(int64_t expected, int64_t value) noexcept {
            return value_.compare_exchange_strong(expected, value,
                                                  std::memory_order_acq_rel);
        }
        
    private:
        alignas(CACHE_LINE_SIZE) std::atomic<int64_t> value_;
    };
    
    std::unique_ptr<Sequence> new_barrier() {
        auto seq = std::make_unique<Sequence>();
        gating_sequences_.push_back(seq.get());
        return seq;
    }
    
    int64_t get_cursor() const noexcept {
        return cursor_.load(std::memory_order_acquire);
    }
    
    size_t size() const noexcept {
        return size_;
    }
    
    // Wait strategy for consumers
    class BusySpinWaitStrategy {
    public:
        int64_t wait_for(int64_t sequence, const RingBuffer& buffer, 
                        Sequence& consumer_sequence) noexcept {
            int64_t available_sequence;
            
            while ((available_sequence = buffer.get_cursor()) < sequence) {
                _mm_pause(); // CPU hint for spin-wait loop
            }
            
            return available_sequence;
        }
    };
    
    class YieldingWaitStrategy {
    public:
        int64_t wait_for(int64_t sequence, const RingBuffer& buffer,
                        Sequence& consumer_sequence) noexcept {
            int64_t available_sequence;
            int counter = 100;
            
            while ((available_sequence = buffer.get_cursor()) < sequence) {
                if (--counter == 0) {
                    std::this_thread::yield();
                    counter = 100;
                } else {
                    _mm_pause();
                }
            }
            
            return available_sequence;
        }
    };
    
private:
    static size_t next_power_of_two(size_t n) {
        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        n |= n >> 32;
        n++;
        return n;
    }
    
    T* allocate_aligned_buffer() {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, CACHE_LINE_SIZE, size_ * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }
    
    int64_t get_minimum_sequence(int64_t minimum) const noexcept {
        for (auto* seq : gating_sequences_) {
            int64_t value = seq->get();
            if (value < minimum) {
                minimum = value;
            }
        }
        return minimum;
    }
    
    const size_t size_;
    const size_t mask_;
    T* buffer_;
    
    alignas(CACHE_LINE_SIZE) std::atomic<int64_t> cursor_;
    alignas(CACHE_LINE_SIZE) std::atomic<int64_t> cached_gating_sequence_;
    
    std::vector<Sequence*> gating_sequences_;
};

// Batch event processor for consumers
template<typename T, typename Handler>
class BatchEventProcessor {
public:
    using WaitStrategy = typename RingBuffer<T>::YieldingWaitStrategy;
    
    BatchEventProcessor(RingBuffer<T>& buffer, Handler& handler)
        : buffer_(buffer),
          handler_(handler),
          sequence_(buffer.new_barrier()),
          running_(false) {}
    
    void start() {
        running_.store(true, std::memory_order_relaxed);
        thread_ = std::thread(&BatchEventProcessor::run, this);
    }
    
    void stop() {
        running_.store(false, std::memory_order_relaxed);
        if (thread_.joinable()) {
            thread_.join();
        }
    }
    
private:
    void run() {
        WaitStrategy wait_strategy;
        int64_t next_sequence = sequence_->get() + 1;
        
        while (running_.load(std::memory_order_relaxed)) {
            try {
                const int64_t available_sequence = wait_strategy.wait_for(
                    next_sequence, buffer_, *sequence_);
                
                while (next_sequence <= available_sequence) {
                    handler_.on_event(buffer_[next_sequence], next_sequence);
                    next_sequence++;
                }
                
                sequence_->set(available_sequence);
            } catch (const std::exception& e) {
                handler_.on_error(e);
            }
        }
    }
    
    RingBuffer<T>& buffer_;
    Handler& handler_;
    std::unique_ptr<typename RingBuffer<T>::Sequence> sequence_;
    std::atomic<bool> running_;
    std::thread thread_;
};

}
