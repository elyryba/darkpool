#include <benchmark/benchmark.h>
#include <thread>
#include <atomic>
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>

#include "darkpool/detector.hpp"
#include "darkpool/core/detector_impl.hpp"
#include "darkpool/core/realtime_stream.hpp"
#include "darkpool/utils/cpu_affinity.hpp"
#include "darkpool/utils/ring_buffer.hpp"
#include "darkpool/protocols/fix_parser.hpp"
#include "darkpool/protocols/itch_parser.hpp"

using namespace darkpool;
using namespace darkpool::core;
using namespace darkpool::protocols;
using namespace darkpool::utils;

// Global stats
struct ThroughputStats {
    std::atomic<uint64_t> messages_processed{0};
    std::atomic<uint64_t> anomalies_detected{0};
    std::atomic<uint64_t> parse_failures{0};
    std::atomic<uint64_t> queue_overflows{0};
    std::atomic<uint64_t> total_latency_ns{0};
    std::atomic<uint64_t> max_latency_ns{0};
    
    void reset() {
        messages_processed = 0;
        anomalies_detected = 0;
        parse_failures = 0;
        queue_overflows = 0;
        total_latency_ns = 0;
        max_latency_ns = 0;
    }
    
    void print(double duration_seconds) {
        double throughput = messages_processed.load() / duration_seconds;
        double avg_latency = total_latency_ns.load() / 
                           std::max(1UL, messages_processed.load());
        
        std::cout << "\nThroughput Statistics:\n";
        std::cout << "Duration: " << duration_seconds << " seconds\n";
        std::cout << "Messages processed: " << messages_processed.load() << "\n";
        std::cout << "Throughput: " << std::fixed << std::setprecision(2) 
                  << throughput / 1e6 << " M msgs/sec\n";
        std::cout << "Anomalies detected: " << anomalies_detected.load() << "\n";
        std::cout << "Parse failures: " << parse_failures.load() << "\n";
        std::cout << "Queue overflows: " << queue_overflows.load() << "\n";
        std::cout << "Avg latency: " << avg_latency << " ns\n";
        std::cout << "Max latency: " << max_latency_ns.load() << " ns\n";
        
        if (throughput < 20e6) {
            std::cout << "WARNING: Throughput " << throughput / 1e6 
                      << " M msgs/sec is below 20M target!\n";
        } else {
            std::cout << "SUCCESS: Throughput " << throughput / 1e6 
                      << " M msgs/sec meets target!\n";
        }
    }
};

// Message generator thread
class MessageGenerator {
public:
    MessageGenerator(RingBuffer<MarketMessage>& buffer, 
                    const std::string& name,
                    int cpu_id)
        : buffer_(buffer), name_(name), cpu_id_(cpu_id) {}
    
    void start() {
        running_ = true;
        thread_ = std::thread(&MessageGenerator::run, this);
    }
    
    void stop() {
        running_ = false;
        if (thread_.joinable()) {
            thread_.join();
        }
    }
    
    uint64_t get_generated() const { return generated_.load(); }
    
private:
    void run() {
        utils::set_cpu_affinity(cpu_id_);
        
        std::vector<MarketMessage> batch;
        batch.reserve(1000);
        
        // Pre-generate messages
        for (int i = 0; i < 1000; ++i) {
            MarketMessage msg;
            msg.type = (i % 3 == 0) ? MessageType::TRADE : MessageType::QUOTE;
            msg.symbol = "SYM" + std::to_string(i % 10);
            msg.price = 100.0 + (i % 100) * 0.01;
            msg.quantity = 100 + (i % 10) * 100;
            msg.timestamp = std::chrono::system_clock::now();
            
            if (msg.type == MessageType::QUOTE) {
                msg.bid_price = msg.price - 0.01;
                msg.ask_price = msg.price + 0.01;
                msg.bid_size = msg.quantity;
                msg.ask_size = msg.quantity;
            }
            
            batch.push_back(msg);
        }
        
        size_t idx = 0;
        while (running_) {
            // Try to push batch
            for (size_t i = 0; i < 100 && running_; ++i) {
                auto& msg = batch[idx % batch.size()];
                msg.timestamp = std::chrono::system_clock::now();
                
                if (buffer_.try_push(msg)) {
                    generated_++;
                } else {
                    // Back off on overflow
                    std::this_thread::yield();
                }
                idx++;
            }
        }
    }
    
    RingBuffer<MarketMessage>& buffer_;
    std::string name_;
    int cpu_id_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> generated_{0};
};

// Processing thread
class ProcessingThread {
public:
    ProcessingThread(RingBuffer<MarketMessage>& buffer,
                    ThroughputStats& stats,
                    const std::string& name,
                    int cpu_id)
        : buffer_(buffer), stats_(stats), name_(name), cpu_id_(cpu_id) {
        detector_ = std::make_unique<DetectorImpl>();
    }
    
    void start() {
        running_ = true;
        thread_ = std::thread(&ProcessingThread::run, this);
    }
    
    void stop() {
        running_ = false;
        if (thread_.joinable()) {
            thread_.join();
        }
    }
    
private:
    void run() {
        utils::set_cpu_affinity(cpu_id_);
        
        MarketMessage msg;
        while (running_) {
            // Process messages in batches
            for (int i = 0; i < 100 && running_; ++i) {
                if (buffer_.try_pop(msg)) {
                    auto start = std::chrono::high_resolution_clock::now();
                    
                    auto anomalies = detector_->process(msg);
                    
                    auto end = std::chrono::high_resolution_clock::now();
                    auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>
                                  (end - start).count();
                    
                    stats_.messages_processed++;
                    stats_.anomalies_detected += anomalies.size();
                    stats_.total_latency_ns += latency;
                    
                    uint64_t current_max = stats_.max_latency_ns.load();
                    while (latency > current_max && 
                           !stats_.max_latency_ns.compare_exchange_weak(current_max, latency));
                } else {
                    // No messages, yield
                    std::this_thread::yield();
                }
            }
        }
    }
    
    RingBuffer<MarketMessage>& buffer_;
    ThroughputStats& stats_;
    std::string name_;
    int cpu_id_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    std::unique_ptr<DetectorImpl> detector_;
};

// Single-threaded throughput test
static void BM_SingleThreadThroughput(benchmark::State& state) {
    auto detector = std::make_unique<DetectorImpl>();
    
    // Pre-generate messages
    std::vector<MarketMessage> messages;
    for (int i = 0; i < 100000; ++i) {
        MarketMessage msg;
        msg.type = (i % 3 == 0) ? MessageType::TRADE : MessageType::QUOTE;
        msg.symbol = "AAPL";
        msg.price = 150.00 + (i % 100) * 0.01;
        msg.quantity = 100;
        msg.timestamp = std::chrono::system_clock::now();
        messages.push_back(msg);
    }
    
    uint64_t processed = 0;
    size_t idx = 0;
    
    for (auto _ : state) {
        auto anomalies = detector->process(messages[idx % messages.size()]);
        idx++;
        processed++;
        benchmark::DoNotOptimize(anomalies);
    }
    
    state.SetItemsProcessed(processed);
    state.SetLabel("Single-threaded baseline");
}

// Multi-threaded scaling test
static void BM_MultiThreadedScaling(benchmark::State& state) {
    const int num_threads = state.range(0);
    
    // Create ring buffer
    RingBuffer<MarketMessage> buffer(1 << 20); // 1M messages
    ThroughputStats stats;
    
    // Create processing threads
    std::vector<std::unique_ptr<ProcessingThread>> processors;
    for (int i = 0; i < num_threads; ++i) {
        processors.push_back(std::make_unique<ProcessingThread>(
            buffer, stats, "Processor" + std::to_string(i), i + 1));
    }
    
    // Create generator thread
    MessageGenerator generator(buffer, "Generator", 0);
    
    // Start all threads
    generator.start();
    for (auto& p : processors) {
        p->start();
    }
    
    // Run for benchmark duration
    auto start = std::chrono::steady_clock::now();
    
    for (auto _ : state) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    auto end = std::chrono::steady_clock::now();
    double duration = std::chrono::duration<double>(end - start).count();
    
    // Stop all threads
    generator.stop();
    for (auto& p : processors) {
        p->stop();
    }
    
    // Report results
    stats.print(duration);
    
    state.SetItemsProcessed(stats.messages_processed.load());
    state.SetLabel(std::to_string(num_threads) + " threads");
}

// Pipeline throughput test
static void BM_PipelineThroughput(benchmark::State& state) {
    // Create pipeline: Parser -> Normalizer -> Detector
    auto stream = std::make_unique<RealtimeStream>();
    ThroughputStats stats;
    
    // FIX parser thread
    std::thread fix_thread([&]() {
        utils::set_cpu_affinity(1);
        FIXParser parser;
        
        // Pre-generate FIX messages
        std::vector<std::string> fix_messages;
        for (int i = 0; i < 10000; ++i) {
            std::stringstream ss;
            ss << "8=FIX.4.2\x019=100\x0135=D\x0149=TEST\x0156=DARK\x01"
               << "11=ORD" << i << "\x0155=AAPL\x0154=1\x0138=100\x01"
               << "44=150.50\x0140=2\x0110=100\x01";
            fix_messages.push_back(ss.str());
        }
        
        size_t idx = 0;
        while (state.KeepRunning()) {
            const auto& msg = fix_messages[idx % fix_messages.size()];
            
            FIXMessage parsed;
            if (parser.parse(msg.c_str(), msg.length(), parsed)) {
                stream->publish(parsed);
                stats.messages_processed++;
            } else {
                stats.parse_failures++;
            }
            idx++;
        }
    });
    
    // ITCH parser thread
    std::thread itch_thread([&]() {
        utils::set_cpu_affinity(2);
        ITCHParser parser;
        
        // Pre-generate ITCH messages
        std::vector<std::vector<uint8_t>> itch_messages;
        for (int i = 0; i < 10000; ++i) {
            // Simplified ITCH Add Order
            std::vector<uint8_t> msg(36);
            msg[0] = 0;
            msg[1] = 35;
            msg[2] = 'A';
            // ... (fill rest of message)
            itch_messages.push_back(msg);
        }
        
        size_t idx = 0;
        while (state.KeepRunning()) {
            const auto& msg = itch_messages[idx % itch_messages.size()];
            
            ITCHMessage parsed;
            if (parser.parse(msg.data(), msg.size(), parsed)) {
                stream->publish(parsed);
                stats.messages_processed++;
            } else {
                stats.parse_failures++;
            }
            idx++;
        }
    });
    
    // Wait for completion
    if (fix_thread.joinable()) fix_thread.join();
    if (itch_thread.joinable()) itch_thread.join();
    
    state.SetItemsProcessed(stats.messages_processed.load());
}

// Memory bandwidth test
static void BM_MemoryBandwidth(benchmark::State& state) {
    const size_t message_size = sizeof(MarketMessage);
    const size_t buffer_size = 1 << 24; // 16M messages
    
    std::vector<MarketMessage> buffer(buffer_size);
    
    // Initialize buffer
    for (size_t i = 0; i < buffer_size; ++i) {
        buffer[i].type = MessageType::TRADE;
        buffer[i].price = 100.0 + i * 0.01;
        buffer[i].quantity = 100;
    }
    
    size_t read_idx = 0;
    size_t write_idx = buffer_size / 2;
    uint64_t bytes_transferred = 0;
    
    for (auto _ : state) {
        // Simulate read-modify-write pattern
        auto msg = buffer[read_idx % buffer_size];
        msg.price += 0.01;
        buffer[write_idx % buffer_size] = msg;
        
        read_idx++;
        write_idx++;
        bytes_transferred += message_size * 2; // Read + write
        
        benchmark::DoNotOptimize(msg);
    }
    
    state.SetBytesProcessed(bytes_transferred);
    state.SetLabel("Memory bandwidth utilization");
}

// Queue overflow test
static void BM_QueueOverflow(benchmark::State& state) {
    const int burst_size = state.range(0);
    RingBuffer<MarketMessage> buffer(1024); // Small buffer
    
    ThroughputStats stats;
    
    // Processing thread
    std::atomic<bool> running{true};
    std::thread processor([&]() {
        utils::set_cpu_affinity(1);
        auto detector = std::make_unique<DetectorImpl>();
        
        MarketMessage msg;
        while (running) {
            if (buffer.try_pop(msg)) {
                auto anomalies = detector->process(msg);
                stats.messages_processed++;
                stats.anomalies_detected += anomalies.size();
            }
        }
    });
    
    // Generate bursts
    for (auto _ : state) {
        // Create burst
        for (int i = 0; i < burst_size; ++i) {
            MarketMessage msg;
            msg.type = MessageType::TRADE;
            msg.symbol = "BURST";
            msg.price = 100.0 + i * 0.01;
            msg.quantity = 1000;
            msg.timestamp = std::chrono::system_clock::now();
            
            if (!buffer.try_push(msg)) {
                stats.queue_overflows++;
            }
        }
        
        // Let processor catch up
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    
    running = false;
    processor.join();
    
    state.SetItemsProcessed(stats.messages_processed.load());
    state.SetLabel("Burst size: " + std::to_string(burst_size) + 
                   ", Overflows: " + std::to_string(stats.queue_overflows.load()));
}

// Custom main for throughput testing
int main(int argc, char** argv) {
    // Set up environment
    std::cout << "Dark Pool Detector - Throughput Benchmark\n";
    std::cout << "=========================================\n";
    std::cout << "Target: 20M messages/second\n";
    std::cout << "CPU cores available: " << std::thread::hardware_concurrency() << "\n";
    
    // Initialize benchmark
    ::benchmark::Initialize(&argc, argv);
    
    // Register benchmarks
    BENCHMARK(BM_SingleThreadThroughput)->Iterations(10000000);
    BENCHMARK(BM_MultiThreadedScaling)->Range(1, 16)->Iterations(100);
    BENCHMARK(BM_PipelineThroughput)->Iterations(100);
    BENCHMARK(BM_MemoryBandwidth)->Iterations(100000000);
    BENCHMARK(BM_QueueOverflow)->Range(100, 10000)->Iterations(1000);
    
    // Run benchmarks
    ::benchmark::RunSpecifiedBenchmarks();
    
    std::cout << "\n=========================================\n";
    std::cout << "THROUGHPUT BENCHMARK SUMMARY\n";
    std::cout << "=========================================\n";
    std::cout << "Check results above for throughput metrics\n";
    std::cout << "Target: 20M messages/second sustained\n";
    std::cout << "=========================================\n";
    
    return 0;
}
