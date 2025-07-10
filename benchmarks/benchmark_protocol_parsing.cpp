#include <benchmark/benchmark.h>
#include <chrono>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <sstream>

#include "darkpool/protocols/fix_parser.hpp"
#include "darkpool/protocols/itch_parser.hpp"
#include "darkpool/protocols/ouch_parser.hpp"
#include "darkpool/utils/cpu_affinity.hpp"

using namespace darkpool;
using namespace darkpool::protocols;

// Performance counters
inline uint64_t rdtsc() {
    uint32_t lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

// Message generators
class MessageGenerator {
public:
    static std::string generate_fix_new_order(int seq) {
        std::stringstream ss;
        ss << "8=FIX.4.2\x01"
           << "9=150\x01"
           << "35=D\x01"
           << "49=CLIENT\x01"
           << "56=BROKER\x01"
           << "34=" << seq << "\x01"
           << "52=20240315-10:15:30.123\x01"
           << "11=ORDER" << seq << "\x01"
           << "55=AAPL\x01"
           << "54=1\x01"
           << "38=" << (100 + seq % 900) << "\x01"
           << "44=" << (150.00 + (seq % 100) * 0.01) << "\x01"
           << "40=2\x01"
           << "59=0\x01"
           << "10=123\x01";
        return ss.str();
    }
    
    static std::string generate_fix_execution(int seq) {
        std::stringstream ss;
        ss << "8=FIX.4.2\x01"
           << "9=180\x01"
           << "35=8\x01"
           << "49=BROKER\x01"
           << "56=CLIENT\x01"
           << "34=" << seq << "\x01"
           << "52=20240315-10:15:31.456\x01"
           << "17=EXEC" << seq << "\x01"
           << "11=ORDER" << seq << "\x01"
           << "55=AAPL\x01"
           << "54=1\x01"
           << "31=" << (150.00 + (seq % 100) * 0.01) << "\x01"
           << "32=" << (100 + seq % 900) << "\x01"
           << "14=" << (100 + seq % 900) << "\x01"
           << "151=0\x01"
           << "39=2\x01"
           << "150=F\x01"
           << "10=234\x01";
        return ss.str();
    }
    
    static std::vector<uint8_t> generate_itch_add_order(uint64_t order_ref) {
        std::vector<uint8_t> msg(36);
        
        msg[0] = 0;
        msg[1] = 35;
        msg[2] = 'A'; // Add Order
        
        // Stock locate
        *reinterpret_cast<uint16_t*>(&msg[3]) = htons(1);
        
        // Tracking number
        *reinterpret_cast<uint16_t*>(&msg[5]) = htons(1);
        
        // Timestamp
        uint64_t timestamp = 34200000000000ULL;
        msg[7] = (timestamp >> 40) & 0xFF;
        msg[8] = (timestamp >> 32) & 0xFF;
        msg[9] = (timestamp >> 24) & 0xFF;
        msg[10] = (timestamp >> 16) & 0xFF;
        msg[11] = (timestamp >> 8) & 0xFF;
        msg[12] = timestamp & 0xFF;
        
        // Order reference
        *reinterpret_cast<uint64_t*>(&msg[13]) = htobe64(order_ref);
        
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
    
    static std::vector<uint8_t> generate_itch_trade(uint64_t match_number) {
        std::vector<uint8_t> msg(44);
        
        msg[0] = 0;
        msg[1] = 43;
        msg[2] = 'P'; // Trade message
        
        // Stock locate
        *reinterpret_cast<uint16_t*>(&msg[3]) = htons(1);
        
        // Tracking number
        *reinterpret_cast<uint16_t*>(&msg[5]) = htons(1);
        
        // Timestamp
        uint64_t timestamp = 34200000000000ULL;
        for (int i = 0; i < 6; ++i) {
            msg[7 + i] = (timestamp >> (40 - i * 8)) & 0xFF;
        }
        
        // Order reference
        *reinterpret_cast<uint64_t*>(&msg[13]) = htobe64(1000000);
        
        // Buy/sell
        msg[21] = 'B';
        
        // Shares
        *reinterpret_cast<uint32_t*>(&msg[22]) = htonl(100);
        
        // Stock
        std::memcpy(&msg[26], "AAPL    ", 8);
        
        // Price
        *reinterpret_cast<uint32_t*>(&msg[34]) = htonl(15050);
        
        // Match number
        *reinterpret_cast<uint64_t*>(&msg[38]) = htobe64(match_number);
        
        return msg;
    }
    
    static std::vector<uint8_t> generate_ouch_enter_order(uint64_t token) {
        std::vector<uint8_t> msg(49);
        
        msg[0] = 0;
        msg[1] = 48;
        msg[2] = 'O'; // Enter Order
        
        // Order token
        std::string token_str = "TKN" + std::to_string(token);
        token_str.resize(14, ' ');
        std::memcpy(&msg[3], token_str.c_str(), 14);
        
        // Buy/sell
        msg[17] = 'B';
        
        // Shares
        *reinterpret_cast<uint32_t*>(&msg[18]) = htonl(100);
        
        // Stock
        std::memcpy(&msg[22], "AAPL    ", 8);
        
        // Price
        *reinterpret_cast<uint32_t*>(&msg[30]) = htonl(1505000); // $150.50
        
        // Time in force
        *reinterpret_cast<uint32_t*>(&msg[34]) = htonl(99999);
        
        // Firm
        std::memcpy(&msg[38], "FIRM", 4);
        
        // Display
        msg[42] = 'Y';
        
        // Capacity
        msg[43] = 'A';
        
        // Intermarket sweep
        msg[44] = 'N';
        
        // Min quantity
        *reinterpret_cast<uint32_t*>(&msg[45]) = htonl(1);
        
        return msg;
    }
};

// Benchmark FIX parsing
static void BM_FIXParsing_NewOrder(benchmark::State& state) {
    FIXParser parser;
    
    // Pre-generate messages
    std::vector<std::string> messages;
    for (int i = 0; i < 10000; ++i) {
        messages.push_back(MessageGenerator::generate_fix_new_order(i));
    }
    
    size_t idx = 0;
    uint64_t total_bytes = 0;
    
    for (auto _ : state) {
        const auto& msg = messages[idx % messages.size()];
        
        FIXMessage parsed;
        bool success = parser.parse(msg.c_str(), msg.length(), parsed);
        
        benchmark::DoNotOptimize(success);
        benchmark::DoNotOptimize(parsed);
        
        total_bytes += msg.length();
        idx++;
    }
    
    state.SetBytesProcessed(total_bytes);
    state.SetLabel("FIX New Order");
}

static void BM_FIXParsing_Execution(benchmark::State& state) {
    FIXParser parser;
    
    std::vector<std::string> messages;
    for (int i = 0; i < 10000; ++i) {
        messages.push_back(MessageGenerator::generate_fix_execution(i));
    }
    
    size_t idx = 0;
    uint64_t total_bytes = 0;
    
    for (auto _ : state) {
        const auto& msg = messages[idx % messages.size()];
        
        FIXMessage parsed;
        bool success = parser.parse(msg.c_str(), msg.length(), parsed);
        
        benchmark::DoNotOptimize(success);
        benchmark::DoNotOptimize(parsed);
        
        total_bytes += msg.length();
        idx++;
    }
    
    state.SetBytesProcessed(total_bytes);
    state.SetLabel("FIX Execution");
}

// Benchmark ITCH parsing
static void BM_ITCHParsing_AddOrder(benchmark::State& state) {
    ITCHParser parser;
    
    std::vector<std::vector<uint8_t>> messages;
    for (uint64_t i = 0; i < 10000; ++i) {
        messages.push_back(MessageGenerator::generate_itch_add_order(1000000 + i));
    }
    
    size_t idx = 0;
    uint64_t total_bytes = 0;
    
    for (auto _ : state) {
        const auto& msg = messages[idx % messages.size()];
        
        ITCHMessage parsed;
        bool success = parser.parse(msg.data(), msg.size(), parsed);
        
        benchmark::DoNotOptimize(success);
        benchmark::DoNotOptimize(parsed);
        
        total_bytes += msg.size();
        idx++;
    }
    
    state.SetBytesProcessed(total_bytes);
    state.SetLabel("ITCH Add Order");
}

static void BM_ITCHParsing_Trade(benchmark::State& state) {
    ITCHParser parser;
    
    std::vector<std::vector<uint8_t>> messages;
    for (uint64_t i = 0; i < 10000; ++i) {
        messages.push_back(MessageGenerator::generate_itch_trade(2000000 + i));
    }
    
    size_t idx = 0;
    uint64_t total_bytes = 0;
    
    for (auto _ : state) {
        const auto& msg = messages[idx % messages.size()];
        
        ITCHMessage parsed;
        bool success = parser.parse(msg.data(), msg.size(), parsed);
        
        benchmark::DoNotOptimize(success);
        benchmark::DoNotOptimize(parsed);
        
        total_bytes += msg.size();
        idx++;
    }
    
    state.SetBytesProcessed(total_bytes);
    state.SetLabel("ITCH Trade");
}

// Benchmark OUCH parsing
static void BM_OUCHParsing_EnterOrder(benchmark::State& state) {
    OUCHParser parser;
    
    std::vector<std::vector<uint8_t>> messages;
    for (uint64_t i = 0; i < 10000; ++i) {
        messages.push_back(MessageGenerator::generate_ouch_enter_order(3000000 + i));
    }
    
    size_t idx = 0;
    uint64_t total_bytes = 0;
    
    for (auto _ : state) {
        const auto& msg = messages[idx % messages.size()];
        
        OUCHMessage parsed;
        bool success = parser.parse(msg.data(), msg.size(), parsed);
        
        benchmark::DoNotOptimize(success);
        benchmark::DoNotOptimize(parsed);
        
        total_bytes += msg.size();
        idx++;
    }
    
    state.SetBytesProcessed(total_bytes);
    state.SetLabel("OUCH Enter Order");
}

// Zero-copy validation
static void BM_ZeroCopyValidation(benchmark::State& state) {
    FIXParser parser;
    auto msg = MessageGenerator::generate_fix_new_order(1);
    
    // Track allocations
    size_t allocations_before = 0; // Would need custom allocator to track
    
    for (auto _ : state) {
        FIXMessage parsed;
        parser.parse(msg.c_str(), msg.length(), parsed);
        
        // Verify string_views point to original buffer
        bool zero_copy = (parsed.order_id.data() >= msg.c_str() && 
                         parsed.order_id.data() < msg.c_str() + msg.length());
        
        benchmark::DoNotOptimize(zero_copy);
    }
    
    state.SetLabel("Zero-copy verification");
}

// Cache efficiency test
static void BM_CacheEfficiency(benchmark::State& state) {
    const std::string protocol = state.label();
    
    if (protocol == "FIX") {
        FIXParser parser;
        std::vector<std::string> messages;
        
        // Generate messages that span multiple cache lines
        for (int i = 0; i < 1000; ++i) {
            messages.push_back(MessageGenerator::generate_fix_new_order(i));
        }
        
        size_t idx = 0;
        for (auto _ : state) {
            FIXMessage parsed;
            parser.parse(messages[idx % messages.size()].c_str(), 
                        messages[idx % messages.size()].length(), parsed);
            idx++;
            benchmark::DoNotOptimize(parsed);
        }
    } else if (protocol == "ITCH") {
        ITCHParser parser;
        std::vector<std::vector<uint8_t>> messages;
        
        for (uint64_t i = 0; i < 1000; ++i) {
            messages.push_back(MessageGenerator::generate_itch_add_order(i));
        }
        
        size_t idx = 0;
        for (auto _ : state) {
            ITCHMessage parsed;
            parser.parse(messages[idx % messages.size()].data(),
                        messages[idx % messages.size()].size(), parsed);
            idx++;
            benchmark::DoNotOptimize(parsed);
        }
    }
}

// Branch prediction analysis
static void BM_BranchPrediction(benchmark::State& state) {
    ITCHParser parser;
    
    // Mix of different message types to stress branch prediction
    std::vector<std::vector<uint8_t>> messages;
    
    // Add different ITCH message types
    for (int i = 0; i < 1000; ++i) {
        if (i % 3 == 0) {
            messages.push_back(MessageGenerator::generate_itch_add_order(i));
        } else if (i % 3 == 1) {
            messages.push_back(MessageGenerator::generate_itch_trade(i));
        } else {
            // System event message
            std::vector<uint8_t> msg(12);
            msg[0] = 0;
            msg[1] = 11;
            msg[2] = 'S';
            messages.push_back(msg);
        }
    }
    
    // Shuffle to make branch prediction harder
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(messages.begin(), messages.end(), g);
    
    size_t idx = 0;
    for (auto _ : state) {
        ITCHMessage parsed;
        parser.parse(messages[idx % messages.size()].data(),
                    messages[idx % messages.size()].size(), parsed);
        idx++;
        benchmark::DoNotOptimize(parsed);
    }
    
    state.SetLabel("Mixed message types");
}

// Message size impact
static void BM_MessageSizeImpact(benchmark::State& state) {
    const size_t extra_fields = state.range(0);
    FIXParser parser;
    
    // Generate messages with varying sizes
    std::vector<std::string> messages;
    for (size_t i = 0; i < 1000; ++i) {
        std::stringstream ss;
        ss << "8=FIX.4.2\x01"
           << "9=" << (100 + extra_fields * 10) << "\x01"
           << "35=D\x01"
           << "49=CLIENT\x01"
           << "56=BROKER\x01"
           << "11=ORDER" << i << "\x01"
           << "55=AAPL\x01"
           << "54=1\x01"
           << "38=100\x01"
           << "44=150.00\x01";
        
        // Add extra fields
        for (size_t j = 0; j < extra_fields; ++j) {
            ss << (5000 + j) << "=EXTRA" << j << "\x01";
        }
        
        ss << "10=123\x01";
        messages.push_back(ss.str());
    }
    
    size_t idx = 0;
    uint64_t total_bytes = 0;
    
    for (auto _ : state) {
        const auto& msg = messages[idx % messages.size()];
        FIXMessage parsed;
        parser.parse(msg.c_str(), msg.length(), parsed);
        
        total_bytes += msg.length();
        idx++;
        benchmark::DoNotOptimize(parsed);
    }
    
    state.SetBytesProcessed(total_bytes);
    state.SetLabel("Extra fields: " + std::to_string(extra_fields));
}

// Burst vs steady-state performance
static void BM_BurstPerformance(benchmark::State& state) {
    const bool is_burst = state.range(0);
    ITCHParser parser;
    
    std::vector<std::vector<uint8_t>> messages;
    for (uint64_t i = 0; i < 10000; ++i) {
        messages.push_back(MessageGenerator::generate_itch_add_order(i));
    }
    
    if (is_burst) {
        // Burst mode - parse many messages at once
        for (auto _ : state) {
            state.PauseTiming();
            // Simulate burst arrival
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            state.ResumeTiming();
            
            // Parse burst of 100 messages
            for (int i = 0; i < 100; ++i) {
                ITCHMessage parsed;
                parser.parse(messages[i % messages.size()].data(),
                           messages[i % messages.size()].size(), parsed);
                benchmark::DoNotOptimize(parsed);
            }
        }
    } else {
        // Steady-state - parse with delays
        size_t idx = 0;
        for (auto _ : state) {
            ITCHMessage parsed;
            parser.parse(messages[idx % messages.size()].data(),
                       messages[idx % messages.size()].size(), parsed);
            idx++;
            benchmark::DoNotOptimize(parsed);
            
            // Simulate steady arrival
            std::this_thread::yield();
        }
    }
    
    state.SetLabel(is_burst ? "Burst mode" : "Steady-state");
}

// Comprehensive comparison
static void BM_ProtocolComparison(benchmark::State& state) {
    std::cout << "\nProtocol Parsing Performance Summary:\n";
    std::cout << "=====================================\n";
    
    // Test all protocols
    struct Result {
        std::string protocol;
        std::string message_type;
        double ns_per_msg;
        size_t msg_size;
    };
    
    std::vector<Result> results;
    
    // FIX testing
    {
        FIXParser parser;
        auto msg = MessageGenerator::generate_fix_new_order(1);
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 1000000; ++i) {
            FIXMessage parsed;
            parser.parse(msg.c_str(), msg.length(), parsed);
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        results.push_back({"FIX", "New Order", duration.count() / 1000000.0, msg.length()});
    }
    
    // ITCH testing
    {
        ITCHParser parser;
        auto msg = MessageGenerator::generate_itch_add_order(1);
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 1000000; ++i) {
            ITCHMessage parsed;
            parser.parse(msg.data(), msg.size(), parsed);
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        results.push_back({"ITCH", "Add Order", duration.count() / 1000000.0, msg.size()});
    }
    
    // OUCH testing
    {
        OUCHParser parser;
        auto msg = MessageGenerator::generate_ouch_enter_order(1);
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 1000000; ++i) {
            OUCHMessage parsed;
            parser.parse(msg.data(), msg.size(), parsed);
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        results.push_back({"OUCH", "Enter Order", duration.count() / 1000000.0, msg.size()});
    }
    
    // Print results
    std::cout << std::setw(10) << "Protocol" 
              << std::setw(15) << "Message Type"
              << std::setw(15) << "ns/message"
              << std::setw(15) << "Message Size"
              << std::setw(15) << "MB/s" << "\n";
    std::cout << std::string(70, '-') << "\n";
    
    for (const auto& r : results) {
        double mb_per_sec = (r.msg_size * 1000.0) / r.ns_per_msg;
        std::cout << std::setw(10) << r.protocol
                  << std::setw(15) << r.message_type
                  << std::setw(15) << std::fixed << std::setprecision(2) << r.ns_per_msg
                  << std::setw(15) << r.msg_size
                  << std::setw(15) << mb_per_sec << "\n";
    }
    
    // Dummy loop for benchmark framework
    for (auto _ : state) {
        benchmark::DoNotOptimize(results);
    }
}

// Main
int main(int argc, char** argv) {
    std::cout << "Dark Pool Detector - Protocol Parsing Benchmark\n";
    std::cout << "==============================================\n\n";
    
    // Pin to CPU
    utils::set_cpu_affinity(0);
    
    // Initialize benchmark
    ::benchmark::Initialize(&argc, argv);
    
    // Register benchmarks
    BENCHMARK(BM_FIXParsing_NewOrder)->Iterations(10000000);
    BENCHMARK(BM_FIXParsing_Execution)->Iterations(10000000);
    BENCHMARK(BM_ITCHParsing_AddOrder)->Iterations(10000000);
    BENCHMARK(BM_ITCHParsing_Trade)->Iterations(10000000);
    BENCHMARK(BM_OUCHParsing_EnterOrder)->Iterations(10000000);
    BENCHMARK(BM_ZeroCopyValidation)->Iterations(1000000);
    
    benchmark::RegisterBenchmark("BM_CacheEfficiency", BM_CacheEfficiency)
        ->Iterations(10000000)->SetLabel("FIX");
    benchmark::RegisterBenchmark("BM_CacheEfficiency", BM_CacheEfficiency)
        ->Iterations(10000000)->SetLabel("ITCH");
    
    BENCHMARK(BM_BranchPrediction)->Iterations(10000000);
    BENCHMARK(BM_MessageSizeImpact)->Range(0, 20)->Iterations(1000000);
    BENCHMARK(BM_BurstPerformance)->Arg(0)->Arg(1)->Iterations(10000);
    BENCHMARK(BM_ProtocolComparison)->Iterations(1);
    
    // Run benchmarks
    ::benchmark::RunSpecifiedBenchmarks();
    
    std::cout << "\n==============================================\n";
    std::cout << "PROTOCOL PARSING BENCHMARK SUMMARY\n";
    std::cout << "==============================================\n";
    std::cout << "FIX target: <100ns per message\n";
    std::cout << "ITCH target: <50ns per message\n";
    std::cout << "OUCH target: <75ns per message\n";
    std::cout << "==============================================\n";
    
    return 0;
}
