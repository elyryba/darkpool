#include <benchmark/benchmark.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <fstream>
#include <thread>
#include <sched.h>
#include <unistd.h>
#include <sys/mman.h>

#include "darkpool/detector.hpp"
#include "darkpool/protocols/fix_parser.hpp"
#include "darkpool/protocols/itch_parser.hpp"
#include "darkpool/protocols/ouch_parser.hpp"
#include "darkpool/protocols/protocol_normalizer.hpp"
#include "darkpool/core/detector_impl.hpp"
#include "darkpool/utils/cpu_affinity.hpp"

using namespace darkpool;
using namespace darkpool::protocols;
using namespace darkpool::core;

// Global detector instance
static std::unique_ptr<Detector> g_detector;

// CPU frequency for cycle to nanosecond conversion
static double g_cpu_freq_ghz = 3.0;

// Performance counter helpers
inline uint64_t rdtsc() {
    uint32_t lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

inline uint64_t rdtscp() {
    uint32_t lo, hi, aux;
    __asm__ __volatile__ ("rdtscp" : "=a"(lo), "=d"(hi), "=c"(aux));
    return ((uint64_t)hi << 32) | lo;
}

// Memory fence for accurate timing
inline void memory_fence() {
    __asm__ __volatile__ ("mfence" ::: "memory");
}

// Measure CPU frequency
double measure_cpu_frequency() {
    const int iterations = 10;
    std::vector<double> frequencies;
    
    for (int i = 0; i < iterations; ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();
        uint64_t start_cycles = rdtsc();
        
        // Busy wait for 100ms
        while (std::chrono::high_resolution_clock::now() - start_time < std::chrono::milliseconds(100)) {
            __asm__ __volatile__ ("nop");
        }
        
        uint64_t end_cycles = rdtsc();
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
        double cycles = end_cycles - start_cycles;
        double freq_ghz = cycles / duration_ns;
        
        frequencies.push_back(freq_ghz);
    }
    
    // Return median
    std::sort(frequencies.begin(), frequencies.end());
    return frequencies[iterations / 2];
}

// Test message generators
std::string generate_fix_message() {
    static uint64_t seq = 1;
    std::stringstream ss;
    ss << "8=FIX.4.2\x01"
       << "9=130\x01"
       << "35=D\x01"
       << "49=SENDER\x01"
       << "56=TARGET\x01"
       << "34=" << seq++ << "\x01"
       << "52=20240315-12:00:00.000\x01"
       << "11=ORDER" << seq << "\x01"
       << "55=AAPL\x01"
       << "54=1\x01"
       << "38=100\x01"
       << "44=150.50\x01"
       << "40=2\x01"
       << "10=100\x01";
    return ss.str();
}

std::vector<uint8_t> generate_itch_message() {
    static uint64_t order_ref = 1000000;
    std::vector<uint8_t> msg(36);
    
    // Add Order message
    msg[0] = 0;
    msg[1] = 35;
    msg[2] = 'A';
    
    // Stock locate
    uint16_t stock_locate = 1;
    *reinterpret_cast<uint16_t*>(&msg[3]) = htons(stock_locate);
    
    // Tracking number
    uint16_t tracking = 1;
    *reinterpret_cast<uint16_t*>(&msg[5]) = htons(tracking);
    
    // Timestamp (6 bytes)
    uint64_t timestamp = 34200000000000ULL; // 9:30 AM
    msg[7] = (timestamp >> 40) & 0xFF;
    msg[8] = (timestamp >> 32) & 0xFF;
    msg[9] = (timestamp >> 24) & 0xFF;
    msg[10] = (timestamp >> 16) & 0xFF;
    msg[11] = (timestamp >> 8) & 0xFF;
    msg[12] = timestamp & 0xFF;
    
    // Order reference
    *reinterpret_cast<uint64_t*>(&msg[13]) = htobe64(order_ref++);
    
    // Buy/sell
    msg[21] = 'B';
    
    // Shares
    *reinterpret_cast<uint32_t*>(&msg[22]) = htonl(100);
    
    // Stock
    std::memcpy(&msg[26], "AAPL    ", 8);
    
    // Price
    *reinterpret_cast<uint32_t*>(&msg[34]) = htonl(15050);
    
    return msg;
}

// Latency histogram for detailed analysis
class LatencyHistogram {
public:
    void add(uint64_t cycles) {
        samples_.push_back(cycles);
    }
    
    void analyze(const std::string& name) {
        if (samples_.empty()) return;
        
        std::sort(samples_.begin(), samples_.end());
        
        double p50 = percentile(0.50);
        double p90 = percentile(0.90);
        double p95 = percentile(0.95);
        double p99 = percentile(0.99);
        double p999 = percentile(0.999);
        double p9999 = percentile(0.9999);
        
        double mean = std::accumulate(samples_.begin(), samples_.end(), 0.0) / samples_.size();
        
        std::cout << "\n" << name << " Latency Analysis:\n";
        std::cout << "Samples: " << samples_.size() << "\n";
        std::cout << "Mean:    " << cycles_to_ns(mean) << " ns\n";
        std::cout << "P50:     " << cycles_to_ns(p50) << " ns\n";
        std::cout << "P90:     " << cycles_to_ns(p90) << " ns\n";
        std::cout << "P95:     " << cycles_to_ns(p95) << " ns\n";
        std::cout << "P99:     " << cycles_to_ns(p99) << " ns\n";
        std::cout << "P99.9:   " << cycles_to_ns(p999) << " ns\n";
        std::cout << "P99.99:  " << cycles_to_ns(p9999) << " ns\n";
        
        // Check if we meet the <500ns target at P99
        if (cycles_to_ns(p99) > 500.0) {
            std::cout << "WARNING: P99 latency " << cycles_to_ns(p99) 
                      << " ns exceeds 500ns target!\n";
        } else {
            std::cout << "SUCCESS: P99 latency " << cycles_to_ns(p99) 
                      << " ns meets <500ns target!\n";
        }
    }
    
private:
    double percentile(double p) {
        size_t idx = static_cast<size_t>(samples_.size() * p);
        return samples_[std::min(idx, samples_.size() - 1)];
    }
    
    double cycles_to_ns(double cycles) {
        return cycles / g_cpu_freq_ghz;
    }
    
    std::vector<uint64_t> samples_;
};

// End-to-end latency benchmark
static void BM_EndToEndLatency_FIX(benchmark::State& state) {
    FIXParser parser;
    auto normalizer = std::make_unique<ProtocolNormalizer>();
    auto detector = std::make_unique<DetectorImpl>();
    
    // Pre-generate messages
    std::vector<std::string> messages;
    for (int i = 0; i < 10000; ++i) {
        messages.push_back(generate_fix_message());
    }
    
    LatencyHistogram histogram;
    size_t msg_idx = 0;
    
    for (auto _ : state) {
        const auto& msg = messages[msg_idx % messages.size()];
        msg_idx++;
        
        memory_fence();
        uint64_t start = rdtscp();
        
        // Parse FIX message
        FIXMessage fix_msg;
        parser.parse(msg.c_str(), msg.length(), fix_msg);
        
        // Normalize to market message
        MarketMessage market_msg;
        normalizer->normalize(fix_msg, market_msg);
        
        // Detect anomalies
        auto anomalies = detector->process(market_msg);
        
        memory_fence();
        uint64_t end = rdtscp();
        
        uint64_t cycles = end - start;
        histogram.add(cycles);
        
        benchmark::DoNotOptimize(anomalies);
    }
    
    histogram.analyze("FIX End-to-End");
}

static void BM_EndToEndLatency_ITCH(benchmark::State& state) {
    ITCHParser parser;
    auto normalizer = std::make_unique<ProtocolNormalizer>();
    auto detector = std::make_unique<DetectorImpl>();
    
    // Pre-generate messages
    std::vector<std::vector<uint8_t>> messages;
    for (int i = 0; i < 10000; ++i) {
        messages.push_back(generate_itch_message());
    }
    
    LatencyHistogram histogram;
    size_t msg_idx = 0;
    
    for (auto _ : state) {
        const auto& msg = messages[msg_idx % messages.size()];
        msg_idx++;
        
        memory_fence();
        uint64_t start = rdtscp();
        
        // Parse ITCH message
        ITCHMessage itch_msg;
        parser.parse(msg.data(), msg.size(), itch_msg);
        
        // Normalize to market message
        MarketMessage market_msg;
        normalizer->normalize(itch_msg, market_msg);
        
        // Detect anomalies
        auto anomalies = detector->process(market_msg);
        
        memory_fence();
        uint64_t end = rdtscp();
        
        uint64_t cycles = end - start;
        histogram.add(cycles);
        
        benchmark::DoNotOptimize(anomalies);
    }
    
    histogram.analyze("ITCH End-to-End");
}

// Component-level latency benchmarks
static void BM_ComponentLatency_FIXParsing(benchmark::State& state) {
    FIXParser parser;
    auto msg = generate_fix_message();
    LatencyHistogram histogram;
    
    for (auto _ : state) {
        memory_fence();
        uint64_t start = rdtscp();
        
        FIXMessage parsed;
        parser.parse(msg.c_str(), msg.length(), parsed);
        
        memory_fence();
        uint64_t end = rdtscp();
        
        histogram.add(end - start);
        benchmark::DoNotOptimize(parsed);
    }
    
    histogram.analyze("FIX Parsing Component");
}

static void BM_ComponentLatency_ITCHParsing(benchmark::State& state) {
    ITCHParser parser;
    auto msg = generate_itch_message();
    LatencyHistogram histogram;
    
    for (auto _ : state) {
        memory_fence();
        uint64_t start = rdtscp();
        
        ITCHMessage parsed;
        parser.parse(msg.data(), msg.size(), parsed);
        
        memory_fence();
        uint64_t end = rdtscp();
        
        histogram.add(end - start);
        benchmark::DoNotOptimize(parsed);
    }
    
    histogram.analyze("ITCH Parsing Component");
}

static void BM_ComponentLatency_Normalization(benchmark::State& state) {
    auto normalizer = std::make_unique<ProtocolNormalizer>();
    FIXParser parser;
    auto fix_str = generate_fix_message();
    
    FIXMessage fix_msg;
    parser.parse(fix_str.c_str(), fix_str.length(), fix_msg);
    
    LatencyHistogram histogram;
    
    for (auto _ : state) {
        memory_fence();
        uint64_t start = rdtscp();
        
        MarketMessage market_msg;
        normalizer->normalize(fix_msg, market_msg);
        
        memory_fence();
        uint64_t end = rdtscp();
        
        histogram.add(end - start);
        benchmark::DoNotOptimize(market_msg);
    }
    
    histogram.analyze("Normalization Component");
}

static void BM_ComponentLatency_Detection(benchmark::State& state) {
    auto detector = std::make_unique<DetectorImpl>();
    
    // Create test market message
    MarketMessage msg;
    msg.type = MessageType::TRADE;
    msg.symbol = "AAPL";
    msg.price = 150.50;
    msg.quantity = 100;
    msg.timestamp = std::chrono::system_clock::now();
    
    LatencyHistogram histogram;
    
    for (auto _ : state) {
        memory_fence();
        uint64_t start = rdtscp();
        
        auto anomalies = detector->process(msg);
        
        memory_fence();
        uint64_t end = rdtscp();
        
        histogram.add(end - start);
        benchmark::DoNotOptimize(anomalies);
    }
    
    histogram.analyze("Detection Component");
}

// Cache analysis benchmark
static void BM_CacheEfficiency(benchmark::State& state) {
    auto detector = std::make_unique<DetectorImpl>();
    
    // Generate messages that will stress cache
    std::vector<MarketMessage> messages;
    std::vector<std::string> symbols = {"AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
                                       "META", "NVDA", "JPM", "JNJ", "V"};
    
    for (int i = 0; i < 1000; ++i) {
        MarketMessage msg;
        msg.type = (i % 3 == 0) ? MessageType::TRADE : MessageType::QUOTE;
        msg.symbol = symbols[i % symbols.size()];
        msg.price = 100.0 + (i % 100) * 0.01;
        msg.quantity = 100 + (i % 10) * 100;
        msg.timestamp = std::chrono::system_clock::now();
        messages.push_back(msg);
    }
    
    size_t idx = 0;
    
    for (auto _ : state) {
        auto anomalies = detector->process(messages[idx % messages.size()]);
        idx++;
        benchmark::DoNotOptimize(anomalies);
    }
    
    state.SetLabel("Multi-symbol cache stress test");
}

// Burst latency test
static void BM_BurstLatency(benchmark::State& state) {
    const int burst_size = state.range(0);
    auto detector = std::make_unique<DetectorImpl>();
    
    // Generate burst of messages
    std::vector<MarketMessage> burst;
    for (int i = 0; i < burst_size; ++i) {
        MarketMessage msg;
        msg.type = MessageType::TRADE;
        msg.symbol = "AAPL";
        msg.price = 150.00 + i * 0.01;
        msg.quantity = 100;
        msg.timestamp = std::chrono::system_clock::now();
        burst.push_back(msg);
    }
    
    LatencyHistogram histogram;
    
    for (auto _ : state) {
        memory_fence();
        uint64_t start = rdtscp();
        
        for (const auto& msg : burst) {
            auto anomalies = detector->process(msg);
            benchmark::DoNotOptimize(anomalies);
        }
        
        memory_fence();
        uint64_t end = rdtscp();
        
        uint64_t cycles_per_msg = (end - start) / burst_size;
        histogram.add(cycles_per_msg);
    }
    
    histogram.analyze("Burst Processing (per message)");
}

// Custom main to set up environment
int main(int argc, char** argv) {
    // Pin to CPU 0
    utils::set_cpu_affinity(0);
    
    // Set high priority
    nice(-20);
    
    // Lock memory
    mlockall(MCL_CURRENT | MCL_FUTURE);
    
    // Measure CPU frequency
    std::cout << "Measuring CPU frequency...\n";
    g_cpu_freq_ghz = measure_cpu_frequency();
    std::cout << "CPU frequency: " << g_cpu_freq_ghz << " GHz\n";
    
    // Warm up
    std::cout << "Warming up...\n";
    volatile uint64_t dummy = 0;
    for (int i = 0; i < 100000000; ++i) {
        dummy += i;
    }
    
    // Initialize benchmark
    ::benchmark::Initialize(&argc, argv);
    
    // Register benchmarks
    BENCHMARK(BM_EndToEndLatency_FIX)->Iterations(100000);
    BENCHMARK(BM_EndToEndLatency_ITCH)->Iterations(100000);
    BENCHMARK(BM_ComponentLatency_FIXParsing)->Iterations(100000);
    BENCHMARK(BM_ComponentLatency_ITCHParsing)->Iterations(100000);
    BENCHMARK(BM_ComponentLatency_Normalization)->Iterations(100000);
    BENCHMARK(BM_ComponentLatency_Detection)->Iterations(100000);
    BENCHMARK(BM_CacheEfficiency)->Iterations(100000);
    BENCHMARK(BM_BurstLatency)->Range(1, 100)->Iterations(10000);
    
    // Run benchmarks
    ::benchmark::RunSpecifiedBenchmarks();
    
    std::cout << "\n========================================\n";
    std::cout << "LATENCY BENCHMARK SUMMARY\n";
    std::cout << "========================================\n";
    std::cout << "Target: <500ns end-to-end latency at P99\n";
    std::cout << "Check the analysis above for PASS/FAIL status\n";
    std::cout << "========================================\n";
    
    return 0;
}
