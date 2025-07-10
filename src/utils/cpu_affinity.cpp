#include "darkpool/utils/cpu_affinity.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>

#ifdef __linux__
#include <sched.h>
#include <pthread.h>
#include <numa.h>
#include <unistd.h>
#include <sys/syscall.h>
#endif

#ifdef _WIN32
#include <windows.h>
#endif

namespace darkpool::utils {

namespace {
    // Cache line size detection
    size_t detect_cache_line_size() {
#ifdef __linux__
        // Try to read from sysfs
        std::ifstream cache_file("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size");
        if (cache_file.is_open()) {
            size_t size;
            cache_file >> size;
            if (size > 0) return size;
        }
#endif
        // Default to 64 bytes (common for x86_64)
        return 64;
    }
    
    // Global cache for CPU topology
    CPUTopology g_topology;
    std::once_flag g_topology_initialized;
}

bool set_thread_affinity(std::thread::native_handle_type handle, int cpu_id) noexcept {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    
    int result = pthread_setaffinity_np(handle, sizeof(cpu_set_t), &cpuset);
    return result == 0;
    
#elif defined(_WIN32)
    DWORD_PTR mask = 1ULL << cpu_id;
    DWORD_PTR result = SetThreadAffinityMask(reinterpret_cast<HANDLE>(handle), mask);
    return result != 0;
    
#else
    // Platform not supported
    return false;
#endif
}

bool set_current_thread_affinity(int cpu_id) noexcept {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    
    int result = sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
    return result == 0;
    
#elif defined(_WIN32)
    DWORD_PTR mask = 1ULL << cpu_id;
    DWORD_PTR result = SetThreadAffinityMask(GetCurrentThread(), mask);
    return result != 0;
    
#else
    return false;
#endif
}

bool set_numa_memory_policy(int numa_node) noexcept {
#ifdef __linux__
    if (numa_available() < 0) {
        return false; // NUMA not available
    }
    
    // Create nodemask for specified node
    struct bitmask* nodemask = numa_allocate_nodemask();
    if (!nodemask) return false;
    
    numa_bitmask_clearall(nodemask);
    numa_bitmask_setbit(nodemask, numa_node);
    
    // Set memory policy to bind to this node
    int result = numa_set_membind(nodemask);
    numa_free_nodemask(nodemask);
    
    return result == 0;
#else
    // NUMA not supported on this platform
    return false;
#endif
}

void* allocate_numa_memory(size_t size, int numa_node) noexcept {
#ifdef __linux__
    if (numa_available() < 0) {
        // Fallback to regular aligned allocation
        void* ptr = nullptr;
        if (posix_memalign(&ptr, 64, size) == 0) {
            return ptr;
        }
        return nullptr;
    }
    
    // Allocate on specific NUMA node
    void* ptr = numa_alloc_onnode(size, numa_node);
    if (ptr) {
        // Touch pages to ensure they're allocated
        std::memset(ptr, 0, size);
    }
    return ptr;
#else
    // Fallback to regular aligned allocation
    return std::aligned_alloc(64, size);
#endif
}

void free_numa_memory(void* ptr, size_t size) noexcept {
#ifdef __linux__
    if (numa_available() >= 0) {
        numa_free(ptr, size);
    } else {
        std::free(ptr);
    }
#else
    std::free(ptr);
#endif
}

CPUTopology get_cpu_topology() noexcept {
    std::call_once(g_topology_initialized, []() {
        g_topology = detect_cpu_topology();
    });
    return g_topology;
}

CPUTopology detect_cpu_topology() noexcept {
    CPUTopology topology;
    
#ifdef __linux__
    // Get total CPU count
    topology.total_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    
    // Detect NUMA nodes
    if (numa_available() >= 0) {
        topology.numa_nodes = numa_max_node() + 1;
        
        // Map CPUs to NUMA nodes
        topology.cpu_to_numa.resize(topology.total_cpus);
        for (int cpu = 0; cpu < topology.total_cpus; ++cpu) {
            topology.cpu_to_numa[cpu] = numa_node_of_cpu(cpu);
        }
        
        // Get CPUs per node
        topology.cpus_per_node.resize(topology.numa_nodes);
        for (int node = 0; node < topology.numa_nodes; ++node) {
            struct bitmask* cpumask = numa_allocate_cpumask();
            if (numa_node_to_cpus(node, cpumask) == 0) {
                for (int cpu = 0; cpu < topology.total_cpus; ++cpu) {
                    if (numa_bitmask_isbitset(cpumask, cpu)) {
                        topology.cpus_per_node[node].push_back(cpu);
                    }
                }
            }
            numa_free_cpumask(cpumask);
        }
    } else {
        // No NUMA support - treat as single node
        topology.numa_nodes = 1;
        topology.cpu_to_numa.resize(topology.total_cpus, 0);
        topology.cpus_per_node.resize(1);
        for (int cpu = 0; cpu < topology.total_cpus; ++cpu) {
            topology.cpus_per_node[0].push_back(cpu);
        }
    }
    
    // Detect CPU cache hierarchy
    for (int cpu = 0; cpu < topology.total_cpus; ++cpu) {
        CPUTopology::CacheInfo cache_info;
        
        // L1 Data Cache
        std::string l1d_path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + 
                              "/cache/index0/size";
        std::ifstream l1d_file(l1d_path);
        if (l1d_file.is_open()) {
            std::string size_str;
            l1d_file >> size_str;
            cache_info.l1d_cache = parse_size_string(size_str);
        }
        
        // L1 Instruction Cache
        std::string l1i_path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + 
                              "/cache/index1/size";
        std::ifstream l1i_file(l1i_path);
        if (l1i_file.is_open()) {
            std::string size_str;
            l1i_file >> size_str;
            cache_info.l1i_cache = parse_size_string(size_str);
        }
        
        // L2 Cache
        std::string l2_path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + 
                             "/cache/index2/size";
        std::ifstream l2_file(l2_path);
        if (l2_file.is_open()) {
            std::string size_str;
            l2_file >> size_str;
            cache_info.l2_cache = parse_size_string(size_str);
        }
        
        // L3 Cache (shared)
        std::string l3_path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + 
                             "/cache/index3/size";
        std::ifstream l3_file(l3_path);
        if (l3_file.is_open()) {
            std::string size_str;
            l3_file >> size_str;
            cache_info.l3_cache = parse_size_string(size_str);
            
            // Find sharing CPUs
            std::string shared_path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + 
                                    "/cache/index3/shared_cpu_list";
            std::ifstream shared_file(shared_path);
            if (shared_file.is_open()) {
                std::string shared_str;
                shared_file >> shared_str;
                cache_info.l3_sharing_cpus = parse_cpu_list(shared_str);
            }
        }
        
        topology.cache_hierarchy.push_back(cache_info);
    }
    
    // Detect hyperthreading
    std::set<std::vector<int>> unique_l1_sharing;
    for (const auto& cache : topology.cache_hierarchy) {
        if (!cache.l1_sharing_cpus.empty()) {
            unique_l1_sharing.insert(cache.l1_sharing_cpus);
        }
    }
    topology.hyperthreading_enabled = unique_l1_sharing.size() < topology.total_cpus;
    
#elif defined(_WIN32)
    // Windows implementation
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    topology.total_cpus = sysinfo.dwNumberOfProcessors;
    
    // Simplified topology for Windows
    topology.numa_nodes = 1;
    topology.cpu_to_numa.resize(topology.total_cpus, 0);
    topology.cpus_per_node.resize(1);
    for (int i = 0; i < topology.total_cpus; ++i) {
        topology.cpus_per_node[0].push_back(i);
    }
    
#else
    // Fallback for other platforms
    topology.total_cpus = std::thread::hardware_concurrency();
    topology.numa_nodes = 1;
    topology.cpu_to_numa.resize(topology.total_cpus, 0);
#endif
    
    topology.cache_line_size = detect_cache_line_size();
    
    return topology;
}

int get_current_cpu() noexcept {
#ifdef __linux__
    return sched_getcpu();
#elif defined(_WIN32)
    return GetCurrentProcessorNumber();
#else
    return -1;
#endif
}

size_t get_cache_line_size() noexcept {
    static size_t cache_line_size = detect_cache_line_size();
    return cache_line_size;
}

std::vector<int> get_numa_node_cpus(int numa_node) noexcept {
    auto topology = get_cpu_topology();
    if (numa_node >= 0 && numa_node < topology.numa_nodes) {
        return topology.cpus_per_node[numa_node];
    }
    return {};
}

int get_numa_node_for_cpu(int cpu_id) noexcept {
    auto topology = get_cpu_topology();
    if (cpu_id >= 0 && cpu_id < static_cast<int>(topology.cpu_to_numa.size())) {
        return topology.cpu_to_numa[cpu_id];
    }
    return -1;
}

// Helper function to parse size strings like "32K" or "8M"
size_t parse_size_string(const std::string& size_str) noexcept {
    if (size_str.empty()) return 0;
    
    size_t value = 0;
    char unit = '\0';
    std::istringstream iss(size_str);
    iss >> value >> unit;
    
    switch (unit) {
        case 'K': case 'k': return value * 1024;
        case 'M': case 'm': return value * 1024 * 1024;
        case 'G': case 'g': return value * 1024 * 1024 * 1024;
        default: return value;
    }
}

// Helper function to parse CPU lists like "0-3,8-11"
std::vector<int> parse_cpu_list(const std::string& cpu_list) noexcept {
    std::vector<int> cpus;
    std::istringstream iss(cpu_list);
    std::string range;
    
    while (std::getline(iss, range, ',')) {
        size_t dash_pos = range.find('-');
        if (dash_pos != std::string::npos) {
            // Range like "0-3"
            int start = std::stoi(range.substr(0, dash_pos));
            int end = std::stoi(range.substr(dash_pos + 1));
            for (int i = start; i <= end; ++i) {
                cpus.push_back(i);
            }
        } else {
            // Single CPU
            cpus.push_back(std::stoi(range));
        }
    }
    
    return cpus;
}

} 
