#pragma once

#include <atomic>
#include <chrono>
#include <map>
#include <string>
#include <vector>
#include <prometheus/counter.h>
#include <prometheus/histogram.h>
#include <prometheus/exposer.h>
#include <prometheus/registry.h>

namespace darkpool {

class MetricsCollector {
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    
public:
    explicit MetricsCollector(uint16_t prometheus_port = 9090) 
        : exposer_(std::make_unique<prometheus::Exposer>("0.0.0.0:" + std::to_string(prometheus_port))),
          registry_(std::make_shared<prometheus::Registry>()) {
        
        exposer_->RegisterCollectable(registry_);
        
        // Initialize metrics
        messages_processed_ = &prometheus::BuildCounter()
            .Name("darkpool_messages_processed_total")
            .Help("Total number of messages processed")
            .Register(*registry_)
            .Add({});
            
        anomalies_detected_ = &prometheus::BuildCounter()
            .Name("darkpool_anomalies_detected_total")
            .Help("Total number of anomalies detected")
            .Register(*registry_)
            .Add({});
            
        message_latency_ = &prometheus::BuildHistogram()
            .Name("darkpool_message_latency_nanoseconds")
            .Help("Message processing latency in nanoseconds")
            .Buckets({100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000})
            .Register(*registry_)
            .Add({});
            
        ml_inference_latency_ = &prometheus::BuildHistogram()
            .Name("darkpool_ml_inference_latency_microseconds")
            .Help("ML inference latency in microseconds")
            .Buckets({100, 500, 1000, 2000, 5000, 10000, 20000})
            .Register(*registry_)
            .Add({});
    }
    
    class ScopedTimer {
    public:
        ScopedTimer(prometheus::Histogram& histogram) 
            : histogram_(histogram), start_(Clock::now()) {}
        
        ~ScopedTimer() {
            auto duration = Clock::now() - start_;
            histogram_.Observe(std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count());
        }
        
    private:
        prometheus::Histogram& histogram_;
        TimePoint start_;
    };
    
    void increment_messages() {
        messages_processed_->Increment();
    }
    
    void increment_anomalies() {
        anomalies_detected_->Increment();
    }
    
    ScopedTimer time_message_processing() {
        return ScopedTimer(*message_latency_);
    }
    
    ScopedTimer time_ml_inference() {
        return ScopedTimer(*ml_inference_latency_);
    }
    
    // High-frequency metrics using atomics for minimal overhead
    struct FastMetrics {
        alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> messages{0};
        alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> anomalies{0};
        alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> total_latency_ns{0};
        alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> max_latency_ns{0};
        
        void record_latency(uint64_t latency_ns) {
            total_latency_ns.fetch_add(latency_ns, std::memory_order_relaxed);
            
            uint64_t current_max = max_latency_ns.load(std::memory_order_relaxed);
            while (latency_ns > current_max && 
                   !max_latency_ns.compare_exchange_weak(current_max, latency_ns,
                                                          std::memory_order_relaxed)) {}
        }
        
        void reset() {
            messages.store(0, std::memory_order_relaxed);
            anomalies.store(0, std::memory_order_relaxed);
            total_latency_ns.store(0, std::memory_order_relaxed);
            max_latency_ns.store(0, std::memory_order_relaxed);
        }
    };
    
    FastMetrics& fast_metrics() { return fast_metrics_; }
    
    // Periodic flush to Prometheus
    void flush_fast_metrics() {
        uint64_t messages = fast_metrics_.messages.exchange(0, std::memory_order_relaxed);
        uint64_t anomalies = fast_metrics_.anomalies.exchange(0, std::memory_order_relaxed);
        
        messages_processed_->Increment(messages);
        anomalies_detected_->Increment(anomalies);
        
        // Can add more sophisticated histogram updates here
    }
    
private:
    std::unique_ptr<prometheus::Exposer> exposer_;
    std::shared_ptr<prometheus::Registry> registry_;
    
    prometheus::Counter* messages_processed_;
    prometheus::Counter* anomalies_detected_;
    prometheus::Histogram* message_latency_;
    prometheus::Histogram* ml_inference_latency_;
    
    FastMetrics fast_metrics_;
};

// Inline performance counter for hot path
class PerfCounter {
public:
    PerfCounter() : start_(rdtsc()) {}
    
    uint64_t elapsed_cycles() const {
        return rdtsc() - start_;
    }
    
    uint64_t elapsed_ns() const {
        // Assuming 3GHz CPU for now
        return elapsed_cycles() / 3;
    }
    
private:
    static uint64_t rdtsc() {
        unsigned int lo, hi;
        __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
        return ((uint64_t)hi << 32) | lo;
    }
    
    uint64_t start_;
};

} 
