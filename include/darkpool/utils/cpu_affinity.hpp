#pragma once

#include <thread>
#include <vector>
#include <sched.h>
#include <pthread.h>
#include <numa.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace darkpool {

class CPUAffinity {
public:
    static bool set_thread_affinity(int cpu_id) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_id, &cpuset);
        
        pthread_t thread = pthread_self();
        return pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset) == 0;
    }
    
    static bool set_thread_affinity(const std::vector<int>& cpu_ids) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        
        for (int cpu_id : cpu_ids) {
            CPU_SET(cpu_id, &cpuset);
        }
        
        pthread_t thread = pthread_self();
        return pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset) == 0;
    }
    
    static int get_current_cpu() {
        return sched_getcpu();
    }
    
    static bool set_numa_node(int node) {
        if (numa_available() < 0) {
            return false;
        }
        
        struct bitmask* mask = numa_allocate_nodemask();
        numa_bitmask_setbit(mask, node);
        
        int result = numa_run_on_node_mask(mask);
        numa_free_nodemask(mask);
        
        return result == 0;
    }
    
    static int get_numa_node_for_cpu(int cpu_id) {
        if (numa_available() < 0) {
            return -1;
        }
        
        return numa_node_of_cpu(cpu_id);
    }
    
    static void set_thread_name(const std::string& name) {
        pthread_setname_np(pthread_self(), name.substr(0, 15).c_str());
    }
    
    static void set_scheduler_fifo(int priority = 99) {
        struct sched_param param;
        param.sched_priority = priority;
        
        pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
    }
    
    static void prefetch_memory(void* addr, size_t size) {
        const char* p = static_cast<const char*>(addr);
        const char* end = p + size;
        
        while (p < end) {
            __builtin_prefetch(p, 0, 3); // Read, high temporal locality
            p += CACHE_LINE_SIZE;
        }
    }
    
    static void disable_cpu_frequency_scaling() {
        // Requires root or CAP_SYS_ADMIN
        system("echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor");
    }
    
    static void isolate_cpus(const std::vector<int>& cpu_ids) {
        // Add to kernel boot parameters: isolcpus=0,2,4,6
        // This is informational only - actual isolation requires boot config
        std::string isolated;
        for (size_t i = 0; i < cpu_ids.size(); ++i) {
            if (i > 0) isolated += ",";
            isolated += std::to_string(cpu_ids[i]);
        }
        
        // Log recommendation
        printf("Recommend adding to kernel boot: isolcpus=%s\n", isolated.c_str());
    }
    
    class ScopedAffinity {
    public:
        explicit ScopedAffinity(int cpu_id) {
            pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &original_);
            set_thread_affinity(cpu_id);
        }
        
        ~ScopedAffinity() {
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &original_);
        }
        
    private:
        cpu_set_t original_;
    };
};

// Thread pool with CPU affinity
class AffinityThreadPool {
public:
    AffinityThreadPool(const std::vector<int>& cpu_ids, const std::string& name_prefix = "darkpool")
        : cpu_ids_(cpu_ids) {
        
        threads_.reserve(cpu_ids.size());
        
        for (size_t i = 0; i < cpu_ids.size(); ++i) {
            threads_.emplace_back([this, i, name_prefix] {
                CPUAffinity::set_thread_affinity(cpu_ids_[i]);
                CPUAffinity::set_thread_name(name_prefix + std::to_string(i));
                CPUAffinity::set_scheduler_fifo();
                
                worker_loop(i);
            });
        }
    }
    
    virtual ~AffinityThreadPool() {
        stop();
    }
    
    void stop() {
        running_.store(false, std::memory_order_relaxed);
        
        for (auto& thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }
    
protected:
    virtual void worker_loop(size_t worker_id) = 0;
    
    std::atomic<bool> running_{true};
    
private:
    std::vector<int> cpu_ids_;
    std::vector<std::thread> threads_;
};

} 
