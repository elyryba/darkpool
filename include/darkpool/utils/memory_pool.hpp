#pragma once

#include <atomic>
#include <cstddef>
#include <cstring>
#include <memory>
#include <vector>
#include <sys/mman.h>
#include <numa.h>

namespace darkpool {

template<typename T>
class MemoryPool {
    static_assert(std::is_trivially_destructible_v<T>, 
                  "MemoryPool only supports trivially destructible types");
    
    struct alignas(CACHE_LINE_SIZE) Block {
        std::atomic<Block*> next;
        char data[sizeof(T)];
    };
    
public:
    explicit MemoryPool(size_t capacity, int numa_node = -1, bool use_huge_pages = true) 
        : capacity_(capacity), numa_node_(numa_node) {
        
        const size_t total_size = capacity * sizeof(Block);
        
        // Allocate memory with NUMA awareness and huge pages
        int flags = MAP_PRIVATE | MAP_ANONYMOUS;
        if (use_huge_pages) {
            flags |= MAP_HUGETLB;
        }
        
        void* mem = nullptr;
        if (numa_node >= 0 && numa_available() >= 0) {
            mem = numa_alloc_onnode(total_size, numa_node);
            if (!mem) {
                throw std::bad_alloc();
            }
        } else {
            mem = mmap(nullptr, total_size, PROT_READ | PROT_WRITE, flags, -1, 0);
            if (mem == MAP_FAILED) {
                throw std::bad_alloc();
            }
        }
        
        // Prefault pages
        std::memset(mem, 0, total_size);
        
        memory_ = static_cast<Block*>(mem);
        memory_size_ = total_size;
        
        // Initialize free list
        for (size_t i = 0; i < capacity - 1; ++i) {
            memory_[i].next.store(&memory_[i + 1], std::memory_order_relaxed);
        }
        memory_[capacity - 1].next.store(nullptr, std::memory_order_relaxed);
        
        free_list_.store(&memory_[0], std::memory_order_relaxed);
    }
    
    ~MemoryPool() {
        if (numa_node_ >= 0 && numa_available() >= 0) {
            numa_free(memory_, memory_size_);
        } else {
            munmap(memory_, memory_size_);
        }
    }
    
    [[nodiscard]] T* allocate() noexcept {
        Block* block = free_list_.load(std::memory_order_acquire);
        
        while (block) {
            Block* next = block->next.load(std::memory_order_relaxed);
            
            if (free_list_.compare_exchange_weak(block, next,
                                                  std::memory_order_release,
                                                  std::memory_order_acquire)) {
                return reinterpret_cast<T*>(block->data);
            }
        }
        
        return nullptr; // Pool exhausted
    }
    
    void deallocate(T* ptr) noexcept {
        if (!ptr) return;
        
        Block* block = reinterpret_cast<Block*>(
            reinterpret_cast<char*>(ptr) - offsetof(Block, data)
        );
        
        Block* current_head = free_list_.load(std::memory_order_relaxed);
        
        do {
            block->next.store(current_head, std::memory_order_relaxed);
        } while (!free_list_.compare_exchange_weak(current_head, block,
                                                    std::memory_order_release,
                                                    std::memory_order_relaxed));
    }
    
    size_t capacity() const noexcept { return capacity_; }
    
    size_t available() const noexcept {
        size_t count = 0;
        Block* current = free_list_.load(std::memory_order_acquire);
        
        while (current) {
            ++count;
            current = current->next.load(std::memory_order_relaxed);
        }
        
        return count;
    }
    
private:
    std::atomic<Block*> free_list_;
    Block* memory_;
    size_t memory_size_;
    size_t capacity_;
    int numa_node_;
};

// Pooled allocator for STL containers
template<typename T>
class PoolAllocator {
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    
    explicit PoolAllocator(MemoryPool<T>& pool) : pool_(&pool) {}
    
    template<typename U>
    PoolAllocator(const PoolAllocator<U>& other) : pool_(other.pool_) {}
    
    T* allocate(size_type n) {
        if (n != 1) {
            throw std::bad_alloc(); // Only single allocations supported
        }
        
        T* ptr = pool_->allocate();
        if (!ptr) {
            throw std::bad_alloc();
        }
        
        return ptr;
    }
    
    void deallocate(T* ptr, size_type n) {
        if (n == 1 && ptr) {
            pool_->deallocate(ptr);
        }
    }
    
    template<typename U>
    bool operator==(const PoolAllocator<U>& other) const {
        return pool_ == other.pool_;
    }
    
    template<typename U>
    bool operator!=(const PoolAllocator<U>& other) const {
        return !(*this == other);
    }
    
private:
    MemoryPool<T>* pool_;
    
    template<typename U>
    friend class PoolAllocator;
};

}
