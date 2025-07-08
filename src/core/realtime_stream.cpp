#include "darkpool/core/realtime_stream.hpp"
#include "darkpool/detector.hpp"
#include "darkpool/protocols/protocol_normalizer.hpp"
#include "darkpool/utils/cpu_affinity.hpp"
#include <thread>
#include <chrono>

namespace darkpool {

class RealTimeStream {
public:
    struct Config {
        size_t backpressure_threshold = 10000;    // Messages before applying backpressure
        size_t latency_check_interval = 1000;     // Check latency every N messages
        double max_latency_ms = 1.0;              // Maximum acceptable latency
        bool enable_flow_control = true;          // Enable adaptive flow control
        size_t batch_size = 100;                  // Process messages in batches
    };
    
    RealTimeStream(const Config& config = Config{});
    ~RealTimeStream();
    
    // Connect components
    void connect(std::shared_ptr<ProtocolNormalizer> normalizer,
                std::shared_ptr<Detector> detector);
    
    // Start/stop streaming
    void start();
    void stop();
    bool is_running() const { return running_.load(); }
    
    // Performance monitoring
    struct StreamMetrics {
        std::atomic<uint64_t> messages_processed{0};
        std::atomic<uint64_t> messages_dropped{0};
        std::atomic<uint64_t> backpressure_events{0};
        std::atomic<uint64_t> total_latency_us{0};
        std::atomic<uint64_t> max_latency_us{0};
        std::atomic<double> throughput_mps{0.0};    // Messages per second
        std::atomic<double> avg_latency_us{0.0};
    };
    
    const StreamMetrics& metrics() const { return metrics_; }
    
    // Flow control
    void pause();
    void resume();
    void set_rate_limit(double messages_per_second);
    
private:
    // Message processing pipeline
    class StreamProcessor : public MessageHandler {
    public:
        StreamProcessor(RealTimeStream* stream);
        
        void on_message(const MarketMessage& message) override;
        void on_error(const std::string& error) override;
        
        void process_batch();
        
    private:
        RealTimeStream* stream_;
        std::vector<MarketMessage> batch_;
        std::chrono::high_resolution_clock::time_point last_batch_time_;
    };
    
    // Monitoring thread
    void monitor_performance();
    
    // Apply backpressure if needed
    void check_backpressure();
    
    // Update throughput metrics
    void update_throughput();
    
    Config config_;
    std::shared_ptr<ProtocolNormalizer> normalizer_;
    std::shared_ptr<Detector> detector_;
    std::unique_ptr<StreamProcessor> processor_;
    
    // Performance monitoring
    StreamMetrics metrics_;
    std::thread monitor_thread_;
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point last_throughput_update_;
    
    // Flow control
    std::atomic<bool> running_{false};
    std::atomic<bool> paused_{false};
    std::atomic<double> rate_limit_{0.0};  // 0 = unlimited
    
    // Backpressure detection
    std::atomic<size_t> pending_messages_{0};
    std::atomic<bool> backpressure_active_{false};
};

// Implementation

RealTimeStream::RealTimeStream(const Config& config) : config_(config) {
    processor_ = std::make_unique<StreamProcessor>(this);
}

RealTimeStream::~RealTimeStream() {
    stop();
}

void RealTimeStream::connect(std::shared_ptr<ProtocolNormalizer> normalizer,
                            std::shared_ptr<Detector> detector) {
    normalizer_ = normalizer;
    detector_ = detector;
    
    // Set up message flow: Normalizer -> StreamProcessor -> Detector
    normalizer_->set_handler(processor_);
}

void RealTimeStream::start() {
    if (running_.load()) return;
    
    running_.store(true);
    start_time_ = std::chrono::high_resolution_clock::now();
    last_throughput_update_ = start_time_;
    
    // Start monitoring thread
    monitor_thread_ = std::thread(&RealTimeStream::monitor_performance, this);
    
    // Set thread properties
    CPUAffinity::set_thread_name("darkpool_monitor");
    
    // Start components
    if (normalizer_) {
        normalizer_->start();
    }
    
    if (detector_) {
        detector_->start();
    }
}

void RealTimeStream::stop() {
    if (!running_.load()) return;
    
    running_.store(false);
    
    // Stop components
    if (detector_) {
        detector_->stop();
    }
    
    if (normalizer_) {
        normalizer_->stop();
    }
    
    // Stop monitoring
    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }
}

void RealTimeStream::pause() {
    paused_.store(true);
}

void RealTimeStream::resume() {
    paused_.store(false);
}

void RealTimeStream::set_rate_limit(double messages_per_second) {
    rate_limit_.store(messages_per_second);
}

void RealTimeStream::monitor_performance() {
    while (running_.load()) {
        // Update metrics every second
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        update_throughput();
        check_backpressure();
        
        // Calculate average latency
        uint64_t total_messages = metrics_.messages_processed.load();
        uint64_t total_latency = metrics_.total_latency_us.load();
        
        if (total_messages > 0) {
            metrics_.avg_latency_us.store(
                static_cast<double>(total_latency) / total_messages
            );
        }
        
        // Log if latency exceeds threshold
        double avg_latency_ms = metrics_.avg_latency_us.load() / 1000.0;
        if (avg_latency_ms > config_.max_latency_ms) {
            // Would log warning in production
            metrics_.backpressure_events.fetch_add(1);
        }
    }
}

void RealTimeStream::check_backpressure() {
    size_t pending = pending_messages_.load();
    
    if (pending > config_.backpressure_threshold) {
        if (!backpressure_active_.load()) {
            backpressure_active_.store(true);
            metrics_.backpressure_events.fetch_add(1);
            
            // Apply flow control
            if (config_.enable_flow_control) {
                // Temporarily slow down processing
                pause();
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                resume();
            }
        }
    } else {
        backpressure_active_.store(false);
    }
}

void RealTimeStream::update_throughput() {
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        now - last_throughput_update_
    ).count();
    
    if (elapsed > 0) {
        uint64_t messages = metrics_.messages_processed.load();
        double throughput = static_cast<double>(messages) / elapsed;
        metrics_.throughput_mps.store(throughput);
        
        // Reset for next interval
        metrics_.messages_processed.store(0);
        last_throughput_update_ = now;
    }
}

// StreamProcessor implementation

RealTimeStream::StreamProcessor::StreamProcessor(RealTimeStream* stream)
    : stream_(stream) {
    batch_.reserve(stream->config_.batch_size);
    last_batch_time_ = std::chrono::high_resolution_clock::now();
}

void RealTimeStream::StreamProcessor::on_message(const MarketMessage& message) {
    // Check if paused
    if (stream_->paused_.load()) {
        stream_->metrics_.messages_dropped.fetch_add(1);
        return;
    }
    
    // Rate limiting
    double rate_limit = stream_->rate_limit_.load();
    if (rate_limit > 0) {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            now - last_batch_time_
        ).count();
        
        double expected_interval_us = 1000000.0 / rate_limit;
        if (elapsed < expected_interval_us) {
            // Too fast, drop message
            stream_->metrics_.messages_dropped.fetch_add(1);
            return;
        }
    }
    
    // Track pending messages
    stream_->pending_messages_.fetch_add(1);
    
    // Measure latency
    auto start = std::chrono::high_resolution_clock::now();
    
    // Add to batch
    batch_.push_back(message);
    
    // Process batch if full
    if (batch_.size() >= stream_->config_.batch_size) {
        process_batch();
    }
    
    // Calculate processing latency
    auto end = std::chrono::high_resolution_clock::now();
    auto latency_us = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start
    ).count();
    
    stream_->metrics_.total_latency_us.fetch_add(latency_us);
    
    // Update max latency
    uint64_t current_max = stream_->metrics_.max_latency_us.load();
    while (latency_us > current_max && 
           !stream_->metrics_.max_latency_us.compare_exchange_weak(current_max, latency_us)) {}
}

void RealTimeStream::StreamProcessor::on_error(const std::string& error) {
    // Forward to detector
    if (stream_->detector_) {
        // Would handle error in production
    }
}

void RealTimeStream::StreamProcessor::process_batch() {
    if (batch_.empty()) return;
    
    // Process all messages in batch
    for (const auto& message : batch_) {
        if (stream_->detector_) {
            stream_->detector_->process_message(message);
        }
        
        stream_->metrics_.messages_processed.fetch_add(1);
        stream_->pending_messages_.fetch_sub(1);
    }
    
    // Clear batch
    batch_.clear();
    last_batch_time_ = std::chrono::high_resolution_clock::now();
}

// Factory function
std::unique_ptr<RealTimeStream> create_realtime_stream(
    const Config& config,
    const RealTimeStream::Config& stream_config) {
    
    auto stream = std::make_unique<RealTimeStream>(stream_config);
    
    // Create components
    auto normalizer = std::make_shared<ProtocolNormalizer>();
    auto detector = DetectorFactory::create_with_config("config/production.yaml");
    
    // Connect pipeline
    stream->connect(normalizer, detector);
    
    // Configure data sources from config
    for (const auto& source : config.market_data_sources) {
        if (source.type == "FIX") {
            normalizer->add_fix_source(source.host, source.port, source.symbols);
        } else if (source.type == "ITCH") {
            normalizer->add_itch_source(source.host, source.port, source.symbols);
        } else if (source.type == "OUCH") {
            normalizer->add_ouch_source(source.host, source.port, source.symbols);
        }
    }
    
    return stream;
}

} 
