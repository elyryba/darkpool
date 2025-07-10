#include "darkpool/utils/metrics_collector.hpp"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <chrono>

namespace darkpool::utils {

// Global metrics instance
MetricsCollector g_metrics;

MetricsCollector::MetricsCollector() {
    // Pre-create common metrics
    create_counter("messages_processed", "Total messages processed");
    create_counter("anomalies_detected", "Total anomalies detected");
    create_gauge("active_connections", "Number of active connections");
    create_histogram("processing_latency_ns", "Message processing latency in nanoseconds");
    create_histogram("ml_inference_latency_us", "ML inference latency in microseconds");
}

void MetricsCollector::create_counter(const std::string& name, const std::string& help) {
    std::lock_guard<std::shared_mutex> lock(mutex_);
    
    auto counter = std::make_unique<Counter>();
    counter->name = name;
    counter->help = help;
    counter->value = 0;
    
    counters_[name] = std::move(counter);
}

void MetricsCollector::create_gauge(const std::string& name, const std::string& help) {
    std::lock_guard<std::shared_mutex> lock(mutex_);
    
    auto gauge = std::make_unique<Gauge>();
    gauge->name = name;
    gauge->help = help;
    gauge->value = 0.0;
    
    gauges_[name] = std::move(gauge);
}

void MetricsCollector::create_histogram(const std::string& name, const std::string& help,
                                       const std::vector<double>& buckets) {
    std::lock_guard<std::shared_mutex> lock(mutex_);
    
    auto hist = std::make_unique<Histogram>();
    hist->name = name;
    hist->help = help;
    
    // Set up buckets
    if (buckets.empty()) {
        // Default exponential buckets
        hist->buckets = {0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000};
    } else {
        hist->buckets = buckets;
    }
    
    // Initialize bucket counts
    hist->bucket_counts.resize(hist->buckets.size() + 1, 0); // +1 for infinity bucket
    hist->sum = 0.0;
    hist->count = 0;
    
    // Initialize circular buffer for percentile calculation
    hist->values.reserve(10000);
    hist->current_index = 0;
    
    histograms_[name] = std::move(hist);
}

void MetricsCollector::increment_counter(const std::string& name, double value) {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    auto it = counters_.find(name);
    if (it != counters_.end()) {
        it->second->value.fetch_add(static_cast<uint64_t>(value), std::memory_order_relaxed);
    }
}

void MetricsCollector::set_gauge(const std::string& name, double value) {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    auto it = gauges_.find(name);
    if (it != gauges_.end()) {
        // Store as uint64_t bit pattern for atomic operations
        uint64_t bits;
        std::memcpy(&bits, &value, sizeof(double));
        it->second->value.store(bits, std::memory_order_relaxed);
    }
}

void MetricsCollector::record_histogram(const std::string& name, double value) {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    auto it = histograms_.find(name);
    if (it != histograms_.end()) {
        auto& hist = *it->second;
        
        // Update sum and count
        hist.sum.fetch_add(static_cast<uint64_t>(value * 1000000), std::memory_order_relaxed);
        hist.count.fetch_add(1, std::memory_order_relaxed);
        
        // Update bucket counts
        size_t bucket_idx = 0;
        for (size_t i = 0; i < hist.buckets.size(); ++i) {
            if (value <= hist.buckets[i]) {
                bucket_idx = i;
                break;
            }
            bucket_idx = i + 1; // infinity bucket
        }
        hist.bucket_counts[bucket_idx].fetch_add(1, std::memory_order_relaxed);
        
        // Store value for percentile calculation
        size_t idx = hist.current_index.fetch_add(1, std::memory_order_relaxed) % 10000;
        if (idx < hist.values.size()) {
            hist.values[idx] = value;
        } else {
            std::lock_guard<std::mutex> value_lock(hist.values_mutex);
            if (hist.values.size() < 10000) {
                hist.values.push_back(value);
            }
        }
    }
}

void MetricsCollector::record_latency(const std::string& name, 
                                    std::chrono::high_resolution_clock::time_point start) {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    record_histogram(name, static_cast<double>(duration));
}

std::string MetricsCollector::expose_prometheus() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    std::ostringstream oss;
    
    // Counters
    for (const auto& [name, counter] : counters_) {
        oss << "# HELP " << name << " " << counter->help << "\n";
        oss << "# TYPE " << name << " counter\n";
        oss << name << " " << counter->value.load(std::memory_order_relaxed) << "\n\n";
    }
    
    // Gauges
    for (const auto& [name, gauge] : gauges_) {
        oss << "# HELP " << name << " " << gauge->help << "\n";
        oss << "# TYPE " << name << " gauge\n";
        
        uint64_t bits = gauge->value.load(std::memory_order_relaxed);
        double value;
        std::memcpy(&value, &bits, sizeof(double));
        oss << name << " " << value << "\n\n";
    }
    
    // Histograms
    for (const auto& [name, hist] : histograms_) {
        oss << "# HELP " << name << " " << hist->help << "\n";
        oss << "# TYPE " << name << " histogram\n";
        
        uint64_t total_count = 0;
        
        // Bucket counts
        for (size_t i = 0; i < hist->buckets.size(); ++i) {
            total_count += hist->bucket_counts[i].load(std::memory_order_relaxed);
            oss << name << "_bucket{le=\"" << hist->buckets[i] << "\"} " << total_count << "\n";
        }
        
        // Infinity bucket
        total_count += hist->bucket_counts.back().load(std::memory_order_relaxed);
        oss << name << "_bucket{le=\"+Inf\"} " << total_count << "\n";
        
        // Sum and count
        double sum = hist->sum.load(std::memory_order_relaxed) / 1000000.0; // Convert back from fixed point
        oss << name << "_sum " << sum << "\n";
        oss << name << "_count " << hist->count.load(std::memory_order_relaxed) << "\n\n";
    }
    
    return oss.str();
}

std::string MetricsCollector::expose_json() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    std::ostringstream oss;
    
    oss << "{\n";
    
    // Counters
    oss << "  \"counters\": {\n";
    bool first_counter = true;
    for (const auto& [name, counter] : counters_) {
        if (!first_counter) oss << ",\n";
        oss << "    \"" << name << "\": " << counter->value.load(std::memory_order_relaxed);
        first_counter = false;
    }
    oss << "\n  },\n";
    
    // Gauges
    oss << "  \"gauges\": {\n";
    bool first_gauge = true;
    for (const auto& [name, gauge] : gauges_) {
        if (!first_gauge) oss << ",\n";
        
        uint64_t bits = gauge->value.load(std::memory_order_relaxed);
        double value;
        std::memcpy(&value, &bits, sizeof(double));
        oss << "    \"" << name << "\": " << value;
        first_gauge = false;
    }
    oss << "\n  },\n";
    
    // Histograms with percentiles
    oss << "  \"histograms\": {\n";
    bool first_hist = true;
    for (const auto& [name, hist] : histograms_) {
        if (!first_hist) oss << ",\n";
        
        auto percentiles = calculate_percentiles(name, {0.5, 0.95, 0.99, 0.999});
        
        oss << "    \"" << name << "\": {\n";
        oss << "      \"count\": " << hist->count.load(std::memory_order_relaxed) << ",\n";
        oss << "      \"sum\": " << (hist->sum.load(std::memory_order_relaxed) / 1000000.0) << ",\n";
        oss << "      \"p50\": " << percentiles[0] << ",\n";
        oss << "      \"p95\": " << percentiles[1] << ",\n";
        oss << "      \"p99\": " << percentiles[2] << ",\n";
        oss << "      \"p999\": " << percentiles[3] << "\n";
        oss << "    }";
        first_hist = false;
    }
    oss << "\n  }\n";
    
    oss << "}\n";
    
    return oss.str();
}

std::vector<double> MetricsCollector::calculate_percentiles(const std::string& name,
                                                           const std::vector<double>& percentiles) const {
    auto it = histograms_.find(name);
    if (it == histograms_.end()) {
        return std::vector<double>(percentiles.size(), 0.0);
    }
    
    auto& hist = *it->second;
    std::vector<double> results;
    
    // Get copy of values for sorting
    std::vector<double> values;
    {
        std::lock_guard<std::mutex> lock(hist.values_mutex);
        values = hist.values;
    }
    
    if (values.empty()) {
        return std::vector<double>(percentiles.size(), 0.0);
    }
    
    // Sort values
    std::sort(values.begin(), values.end());
    
    // Calculate percentiles
    for (double p : percentiles) {
        size_t idx = static_cast<size_t>(p * (values.size() - 1));
        results.push_back(values[idx]);
    }
    
    return results;
}

void MetricsCollector::reset() {
    std::lock_guard<std::shared_mutex> lock(mutex_);
    
    // Reset counters
    for (auto& [name, counter] : counters_) {
        counter->value.store(0, std::memory_order_relaxed);
    }
    
    // Reset histograms
    for (auto& [name, hist] : histograms_) {
        hist->sum.store(0, std::memory_order_relaxed);
        hist->count.store(0, std::memory_order_relaxed);
        for (auto& bucket : hist->bucket_counts) {
            bucket.store(0, std::memory_order_relaxed);
        }
        hist->current_index.store(0, std::memory_order_relaxed);
        
        std::lock_guard<std::mutex> value_lock(hist->values_mutex);
        hist->values.clear();
    }
}

void MetricsCollector::reset_metric(const std::string& name) {
    std::lock_guard<std::shared_mutex> lock(mutex_);
    
    // Check counters
    auto counter_it = counters_.find(name);
    if (counter_it != counters_.end()) {
        counter_it->second->value.store(0, std::memory_order_relaxed);
        return;
    }
    
    // Check histograms
    auto hist_it = histograms_.find(name);
    if (hist_it != histograms_.end()) {
        auto& hist = *hist_it->second;
        hist.sum.store(0, std::memory_order_relaxed);
        hist.count.store(0, std::memory_order_relaxed);
        for (auto& bucket : hist.bucket_counts) {
            bucket.store(0, std::memory_order_relaxed);
        }
        hist.current_index.store(0, std::memory_order_relaxed);
        
        std::lock_guard<std::mutex> value_lock(hist.values_mutex);
        hist.values.clear();
    }
}

// Timer implementation
MetricsTimer::MetricsTimer(MetricsCollector& collector, const std::string& metric_name)
    : collector_(collector)
    , metric_name_(metric_name)
    , start_time_(std::chrono::high_resolution_clock::now()) {
}

MetricsTimer::~MetricsTimer() {
    if (!stopped_) {
        stop();
    }
}

void MetricsTimer::stop() {
    if (!stopped_) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
            end_time - start_time_).count();
        collector_.record_histogram(metric_name_, static_cast<double>(duration));
        stopped_ = true;
    }
}

// Convenience functions
void increment_counter(const std::string& name, double value) {
    g_metrics.increment_counter(name, value);
}

void set_gauge(const std::string& name, double value) {
    g_metrics.set_gauge(name, value);
}

void record_histogram(const std::string& name, double value) {
    g_metrics.record_histogram(name, value);
}

std::unique_ptr<MetricsTimer> start_timer(const std::string& metric_name) {
    return std::make_unique<MetricsTimer>(g_metrics, metric_name);
}

}
