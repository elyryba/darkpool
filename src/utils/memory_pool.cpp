#include "darkpool/utils/memory_pool.hpp"
#include "darkpool/utils/cpu_affinity.hpp"
#include <cstring>
#include <algorithm>
#include <thread>

namespace darkpool::utils {

// Global memory pool instance
MemoryPool g_memory_pool(1024 * 1024 * 1024); // 1GB default

MemoryPool::MemoryPool(size_t total_size, int numa_node)
    : total_size_(total_size)
    , numa_node_(numa_node)
    , allocated_(0)
    , deallocated_(0)
    , high_water_mark_(0) {
    
    initialize();
}

MemoryPool::~MemoryPool() {
    // Free all chunks
    for (auto& chunk : chunks_) {
        if (numa_node_ >= 0) {
            free_numa_memory(chunk.memory, chunk.size);
        } else {
            std::free(chunk.memory);
        }
    }
}

void MemoryPool::initialize() noexcept {
    // Determine number of size classes
    size_classes_.clear();
    free_lists_.clear();
    
    // Create size classes: 64B, 128B, 256B, ..., up to 64KB
    size_t size = 64;
    while (size <= 65536) {
        size_classes_.push_back(size);
        free_lists_.emplace_back(std::make_unique<FreeList>());
        size *= 2;
    }
    
    // Pre-allocate initial chunks
    allocate_chunk(total_size_ / 4); // Start with 25% of total
}

void* MemoryPool::allocate(size_t size) noexcept {
    if (size == 0) return nullptr;
    
    // Add alignment padding
    size = align_size(size);
    
    // Find appropriate size class
    size_t class_idx = get_size_class_index(size);
    
    if (class_idx < size_classes_.size()) {
        // Small allocation - use free list
        return allocate_from_free_list(class_idx, size_classes_[class_idx]);
    } else {
        // Large allocation - allocate directly
        return allocate_large(size);
    }
}

void MemoryPool::deallocate(void* ptr, size_t size) noexcept {
    if (!ptr) return;
    
    size = align_size(size);
    size_t class_idx = get_size_class_index(size);
    
    if (class_idx < size_classes_.size()) {
        // Return to free list
        deallocate_to_free_list(ptr, class_idx);
    } else {
        // Large allocation - track but don't actually free (for performance)
        deallocated_.fetch_add(size, std::memory_order_relaxed);
    }
}

void* MemoryPool::allocate_from_free_list(size_t class_idx, size_t actual_size) noexcept {
    auto& free_list = *free_lists_[class_idx];
    
    // Try to pop from free list
    void* ptr = nullptr;
    if (free_list.free_blocks.pop(ptr)) {
        return ptr;
    }
    
    // Free list empty - allocate new block from chunk
    std::lock_guard<std::mutex> lock(chunk_mutex_);
    
    // Find chunk with space
    for (auto& chunk : chunks_) {
        if (chunk.used + actual_size <= chunk.size) {
            ptr = static_cast<char*>(chunk.memory) + chunk.used;
            chunk.used += actual_size;
            
            // Update statistics
            size_t total_allocated = allocated_.fetch_add(actual_size, std::memory_order_relaxed) + actual_size;
            update_high_water_mark(total_allocated);
            
            return ptr;
        }
    }
    
    // No space in existing chunks - allocate new chunk
    size_t chunk_size = std::max(size_t(1024 * 1024), actual_size * 1024); // At least 1MB
    if (allocate_chunk(chunk_size)) {
        // Retry allocation
        return allocate_from_free_list(class_idx, actual_size);
    }
    
    return nullptr;
}

void MemoryPool::deallocate_to_free_list(void* ptr, size_t class_idx) noexcept {
    auto& free_list = *free_lists_[class_idx];
    
    // Clear memory for security
    std::memset(ptr, 0, size_classes_[class_idx]);
    
    // Push to free list
    free_list.free_blocks.push(ptr);
    
    // Update statistics
    deallocated_.fetch_add(size_classes_[class_idx], std::memory_order_relaxed);
}

void* MemoryPool::allocate_large(size_t size) noexcept {
    std::lock_guard<std::mutex> lock(chunk_mutex_);
    
    // Allocate dedicated chunk for large allocation
    void* ptr = nullptr;
    if (numa_node_ >= 0) {
        ptr = allocate_numa_memory(size, numa_node_);
    } else {
        ptr = std::aligned_alloc(64, size);
    }
    
    if (ptr) {
        Chunk chunk;
        chunk.memory = ptr;
        chunk.size = size;
        chunk.used = size;
        chunks_.push_back(chunk);
        
        // Update statistics
        size_t total_allocated = allocated_.fetch_add(size, std::memory_order_relaxed) + size;
        update_high_water_mark(total_allocated);
    }
    
    return ptr;
}

bool MemoryPool::allocate_chunk(size_t size) noexcept {
    void* memory = nullptr;
    
    if (numa_node_ >= 0) {
        memory = allocate_numa_memory(size, numa_node_);
    } else {
        memory = std::aligned_alloc(64, size);
    }
    
    if (!memory) return false;
    
    Chunk chunk;
    chunk.memory = memory;
    chunk.size = size;
    chunk.used = 0;
    
    chunks_.push_back(chunk);
    return true;
}

size_t MemoryPool::get_size_class_index(size_t size) const noexcept {
    // Binary search for size class
    auto it = std::lower_bound(size_classes_.begin(), size_classes_.end(), size);
    if (it != size_classes_.end()) {
        return std::distance(size_classes_.begin(), it);
    }
    return size_classes_.size();
}

void MemoryPool::update_high_water_mark(size_t allocated) noexcept {
    size_t current = high_water_mark_.load(std::memory_order_relaxed);
    while (allocated > current && 
           !high_water_mark_.compare_exchange_weak(current, allocated)) {
        // Retry
    }
}

MemoryPool::Stats MemoryPool::get_stats() const noexcept {
    Stats stats;
    stats.total_allocated = allocated_.load(std::memory_order_relaxed);
    stats.total_deallocated = deallocated_.load(std::memory_order_relaxed);
    stats.current_usage = stats.total_allocated - stats.total_deallocated;
    stats.high_water_mark = high_water_mark_.load(std::memory_order_relaxed);
    stats.num_chunks = chunks_.size();
    
    // Calculate fragmentation
    size_t total_chunk_size = 0;
    size_t total_used = 0;
    {
        std::lock_guard<std::mutex> lock(chunk_mutex_);
        for (const auto& chunk : chunks_) {
            total_chunk_size += chunk.size;
            total_used += chunk.used;
        }
    }
    
    if (total_chunk_size > 0) {
        stats.fragmentation_ratio = 1.0 - (static_cast<double>(total_used) / total_chunk_size);
    }
    
    return stats;
}

// Thread-local memory pool implementation
namespace {
    thread_local std::unique_ptr<MemoryPool> tl_pool;
}

MemoryPool& get_thread_local_pool() noexcept {
    if (!tl_pool) {
        // Create thread-local pool on current NUMA node
        int cpu = get_current_cpu();
        int numa_node = get_numa_node_for_cpu(cpu);
        tl_pool = std::make_unique<MemoryPool>(64 * 1024 * 1024, numa_node); // 64MB per thread
    }
    return *tl_pool;
}

void* allocate_thread_local(size_t size) noexcept {
    return get_thread_local_pool().allocate(size);
}

void deallocate_thread_local(void* ptr, size_t size) noexcept {
    get_thread_local_pool().deallocate(ptr, size);
}

// Aligned allocation helpers
void* allocate_aligned(size_t size, size_t alignment) noexcept {
    // Ensure alignment is power of 2
    if (alignment & (alignment - 1)) return nullptr;
    
    // Adjust size for alignment
    size_t adjusted_size = size + alignment - 1;
    
    void* ptr = g_memory_pool.allocate(adjusted_size);
    if (!ptr) return nullptr;
    
    // Align pointer
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t aligned_addr = (addr + alignment - 1) & ~(alignment - 1);
    
    return reinterpret_cast<void*>(aligned_addr);
}

void deallocate_aligned(void* ptr, size_t size, size_t alignment) noexcept {
    // For simplicity, we don't track the original pointer
    // In production, would need to store offset
    g_memory_pool.deallocate(ptr, size);
}

// Cache-aligned allocation
void* allocate_cache_aligned(size_t size) noexcept {
    return allocate_aligned(size, get_cache_line_size());
}

void deallocate_cache_aligned(void* ptr, size_t size) noexcept {
    deallocate_aligned(ptr, size, get_cache_line_size());
}

// Huge page allocation support
void* allocate_huge_pages(size_t size) noexcept {
#ifdef __linux__
    // Round up to huge page size (2MB on x86_64)
    const size_t huge_page_size = 2 * 1024 * 1024;
    size = (size + huge_page_size - 1) & ~(huge_page_size - 1);
    
    void* ptr = mmap(nullptr, size, 
                    PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                    -1, 0);
    
    if (ptr == MAP_FAILED) {
        // Fallback to regular allocation
        return allocate_aligned(size, huge_page_size);
    }
    
    return ptr;
#else
    // Fallback to regular large allocation
    return g_memory_pool.allocate(size);
#endif
}

void deallocate_huge_pages(void* ptr, size_t size) noexcept {
#ifdef __linux__
    const size_t huge_page_size = 2 * 1024 * 1024;
    size = (size + huge_page_size - 1) & ~(huge_page_size - 1);
    munmap(ptr, size);
#else
    g_memory_pool.deallocate(ptr, size);
#endif
}

} 
