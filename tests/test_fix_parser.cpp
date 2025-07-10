#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <thread>
#include <vector>
#include <cstring>
#include "darkpool/protocols/fix_parser.hpp"
#include "darkpool/utils/cpu_affinity.hpp"

using namespace darkpool;
using namespace darkpool::protocols;

// Custom allocator to track allocations
static thread_local size_t g_allocation_count = 0;
static thread_local size_t g_allocation_bytes = 0;

template<typename T>
class TrackingAllocator {
public:
    using value_type = T;
    
    TrackingAllocator() = default;
    template<typename U>
    TrackingAllocator(const TrackingAllocator<U>&) {}
    
    T* allocate(size_t n) {
        g_allocation_count++;
        g_allocation_bytes += n * sizeof(T);
        return static_cast<T*>(::operator new(n * sizeof(T)));
    }
    
    void deallocate(T* p, size_t) {
        ::operator delete(p);
    }
};

class FIXParserTest : public ::testing::Test {
protected:
    void SetUp() override {
        parser_ = std::make_unique<FIXParser>();
        g_allocation_count = 0;
        g_allocation_bytes = 0;
        
        // Pin to CPU for consistent timing
        utils::set_cpu_affinity(0);
        
        // Warm up CPU
        volatile uint64_t dummy = 0;
        for (int i = 0; i < 1000000; ++i) {
            dummy += i;
        }
    }
    
    void TearDown() override {
        parser_.reset();
    }
    
    // Helper to create valid FIX message with checksum
    std::string create_fix_message(const std::string& msg_type,
                                  const std::vector<std::pair<int, std::string>>& fields) {
        std::string msg = "8=FIX.4.2\x01";
        msg += "9=";
        
        // Build body
        std::string body = "35=" + msg_type + "\x01";
        body += "49=SENDER\x01";
        body += "56=TARGET\x01";
        body += "34=1\x01";
        body += "52=" + get_timestamp() + "\x01";
        
        for (const auto& [tag, value] : fields) {
            body += std::to_string(tag) + "=" + value + "\x01";
        }
        
        // Add body length
        msg += std::to_string(body.length()) + "\x01";
        msg += body;
        
        // Calculate checksum
        uint8_t checksum = 0;
        for (size_t i = 0; i < msg.length(); ++i) {
            checksum += static_cast<uint8_t>(msg[i]);
        }
        
        msg += "10=" + std::to_string(checksum % 256).substr(0, 3) + "\x01";
        return msg;
    }
    
    std::string get_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        char buffer[32];
        strftime(buffer, sizeof(buffer), "%Y%m%d-%H:%M:%S", gmtime(&time_t));
        return std::string(buffer);
    }
    
    // CPU cycle counter
    inline uint64_t rdtsc() {
        uint32_t lo, hi;
        __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
        return ((uint64_t)hi << 32) | lo;
    }
    
    std::unique_ptr<FIXParser> parser_;
};

// Test basic message parsing
TEST_F(FIXParserTest, ParseNewOrderSingle) {
    auto msg = create_fix_message("D", {
        {11, "ORDER123"},
        {55, "AAPL"},
        {54, "1"},     // Buy
        {38, "100"},   // Quantity
        {44, "150.50"}, // Price
        {40, "2"}      // Limit
    });
    
    FIXMessage parsed;
    ASSERT_TRUE(parser_->parse(msg.c_str(), msg.length(), parsed));
    
    EXPECT_EQ(parsed.msg_type, MessageType::NEW_ORDER_SINGLE);
    EXPECT_EQ(parsed.order_id, "ORDER123");
    EXPECT_EQ(parsed.symbol, "AAPL");
    EXPECT_EQ(parsed.side, Side::BUY);
    EXPECT_EQ(parsed.quantity, 100);
    EXPECT_DOUBLE_EQ(parsed.price, 150.50);
    EXPECT_EQ(parsed.order_type, OrderType::LIMIT);
}

TEST_F(FIXParserTest, ParseOrderCancelRequest) {
    auto msg = create_fix_message("F", {
        {11, "CANCEL123"},
        {41, "ORIG123"},
        {55, "MSFT"},
        {54, "2"}      // Sell
    });
    
    FIXMessage parsed;
    ASSERT_TRUE(parser_->parse(msg.c_str(), msg.length(), parsed));
    
    EXPECT_EQ(parsed.msg_type, MessageType::ORDER_CANCEL_REQUEST);
    EXPECT_EQ(parsed.order_id, "CANCEL123");
    EXPECT_EQ(parsed.orig_order_id, "ORIG123");
    EXPECT_EQ(parsed.symbol, "MSFT");
    EXPECT_EQ(parsed.side, Side::SELL);
}

TEST_F(FIXParserTest, ParseOrderCancelReplaceRequest) {
    auto msg = create_fix_message("G", {
        {11, "REPLACE123"},
        {41, "ORIG456"},
        {55, "GOOGL"},
        {54, "1"},
        {38, "200"},
        {44, "2750.00"}
    });
    
    FIXMessage parsed;
    ASSERT_TRUE(parser_->parse(msg.c_str(), msg.length(), parsed));
    
    EXPECT_EQ(parsed.msg_type, MessageType::ORDER_CANCEL_REPLACE_REQUEST);
    EXPECT_EQ(parsed.order_id, "REPLACE123");
    EXPECT_EQ(parsed.orig_order_id, "ORIG456");
    EXPECT_EQ(parsed.symbol, "GOOGL");
    EXPECT_EQ(parsed.quantity, 200);
    EXPECT_DOUBLE_EQ(parsed.price, 2750.00);
}

TEST_F(FIXParserTest, ParseExecutionReport) {
    auto msg = create_fix_message("8", {
        {17, "EXEC123"},
        {11, "ORDER789"},
        {55, "SPY"},
        {54, "1"},
        {31, "440.25"},   // Last price
        {32, "50"},       // Last quantity
        {14, "50"},       // Cum quantity
        {151, "50"},      // Leaves quantity
        {39, "1"},        // Partially filled
        {150, "F"}        // Exec type: partial fill
    });
    
    FIXMessage parsed;
    ASSERT_TRUE(parser_->parse(msg.c_str(), msg.length(), parsed));
    
    EXPECT_EQ(parsed.msg_type, MessageType::EXECUTION_REPORT);
    EXPECT_EQ(parsed.exec_id, "EXEC123");
    EXPECT_EQ(parsed.order_id, "ORDER789");
    EXPECT_DOUBLE_EQ(parsed.last_price, 440.25);
    EXPECT_EQ(parsed.last_quantity, 50);
    EXPECT_EQ(parsed.cum_quantity, 50);
    EXPECT_EQ(parsed.leaves_quantity, 50);
}

// Test zero-copy behavior
TEST_F(FIXParserTest, ZeroCopyVerification) {
    auto msg = create_fix_message("D", {
        {11, "ZEROCOPY123"},
        {55, "TSLA"},
        {54, "1"},
        {38, "1000"},
        {44, "900.00"}
    });
    
    // Parse should not allocate
    FIXMessage parsed;
    g_allocation_count = 0;
    g_allocation_bytes = 0;
    
    ASSERT_TRUE(parser_->parse(msg.c_str(), msg.length(), parsed));
    
    // Verify no allocations during parse
    EXPECT_EQ(g_allocation_count, 0);
    EXPECT_EQ(g_allocation_bytes, 0);
    
    // Verify string_views point to original buffer
    const char* msg_ptr = msg.c_str();
    EXPECT_GE(parsed.order_id.data(), msg_ptr);
    EXPECT_LT(parsed.order_id.data(), msg_ptr + msg.length());
    EXPECT_GE(parsed.symbol.data(), msg_ptr);
    EXPECT_LT(parsed.symbol.data(), msg_ptr + msg.length());
}

// Test malformed messages
TEST_F(FIXParserTest, MalformedMessageMissingChecksum) {
    std::string msg = "8=FIX.4.2\x01" "9=50\x01" "35=D\x01";
    FIXMessage parsed;
    EXPECT_FALSE(parser_->parse(msg.c_str(), msg.length(), parsed));
}

TEST_F(FIXParserTest, MalformedMessageBadChecksum) {
    auto msg = create_fix_message("D", {{55, "AAPL"}});
    // Corrupt checksum
    msg[msg.length() - 2] = '9';
    
    FIXMessage parsed;
    EXPECT_FALSE(parser_->parse(msg.c_str(), msg.length(), parsed));
}

TEST_F(FIXParserTest, MalformedMessageMissingRequiredField) {
    // Message without symbol (tag 55)
    std::string msg = "8=FIX.4.2\x01" "9=30\x01" "35=D\x01" "49=SENDER\x01" 
                      "56=TARGET\x01" "34=1\x01" "52=20240101-12:00:00\x01" "10=000\x01";
    
    FIXMessage parsed;
    EXPECT_FALSE(parser_->parse(msg.c_str(), msg.length(), parsed));
}

TEST_F(FIXParserTest, EmptyMessage) {
    FIXMessage parsed;
    EXPECT_FALSE(parser_->parse("", 0, parsed));
}

TEST_F(FIXParserTest, TruncatedMessage) {
    auto msg = create_fix_message("D", {{55, "AAPL"}});
    FIXMessage parsed;
    EXPECT_FALSE(parser_->parse(msg.c_str(), msg.length() / 2, parsed));
}

// Test different FIX versions
TEST_F(FIXParserTest, FIX42Support) {
    auto msg = create_fix_message("D", {{55, "IBM"}});
    EXPECT_TRUE(msg.find("8=FIX.4.2") != std::string::npos);
    
    FIXMessage parsed;
    EXPECT_TRUE(parser_->parse(msg.c_str(), msg.length(), parsed));
}

TEST_F(FIXParserTest, FIX44Support) {
    auto msg = create_fix_message("D", {{55, "IBM"}});
    // Replace version
    size_t pos = msg.find("FIX.4.2");
    if (pos != std::string::npos) {
        msg.replace(pos, 7, "FIX.4.4");
    }
    
    FIXMessage parsed;
    EXPECT_TRUE(parser_->parse(msg.c_str(), msg.length(), parsed));
}

// Performance tests
TEST_F(FIXParserTest, PerformanceSingleMessage) {
    auto msg = create_fix_message("D", {
        {11, "PERF123"},
        {55, "AAPL"},
        {54, "1"},
        {38, "100"},
        {44, "150.50"},
        {40, "2"}
    });
    
    FIXMessage parsed;
    
    // Warm up
    for (int i = 0; i < 10000; ++i) {
        parser_->parse(msg.c_str(), msg.length(), parsed);
    }
    
    // Measure
    const int iterations = 1000000;
    std::vector<uint64_t> cycles;
    cycles.reserve(iterations);
    
    for (int i = 0; i < iterations; ++i) {
        uint64_t start = rdtsc();
        parser_->parse(msg.c_str(), msg.length(), parsed);
        uint64_t end = rdtsc();
        cycles.push_back(end - start);
    }
    
    // Calculate percentiles
    std::sort(cycles.begin(), cycles.end());
    uint64_t p50 = cycles[iterations * 0.50];
    uint64_t p95 = cycles[iterations * 0.95];
    uint64_t p99 = cycles[iterations * 0.99];
    uint64_t p999 = cycles[iterations * 0.999];
    
    // Assuming 3GHz CPU for ns conversion
    const double cycles_per_ns = 3.0;
    
    std::cout << "FIX Parser Performance (single message):\n"
              << "  P50:  " << p50 / cycles_per_ns << " ns\n"
              << "  P95:  " << p95 / cycles_per_ns << " ns\n"
              << "  P99:  " << p99 / cycles_per_ns << " ns\n"
              << "  P99.9: " << p999 / cycles_per_ns << " ns\n";
    
    // Verify <100ns target at P99
    EXPECT_LT(p99 / cycles_per_ns, 100.0);
}

TEST_F(FIXParserTest, PerformanceMixedMessages) {
    // Create different message types
    std::vector<std::string> messages;
    messages.push_back(create_fix_message("D", {{11, "NEW1"}, {55, "AAPL"}, {54, "1"}, {38, "100"}}));
    messages.push_back(create_fix_message("F", {{11, "CXL1"}, {41, "ORIG1"}, {55, "MSFT"}}));
    messages.push_back(create_fix_message("G", {{11, "RPL1"}, {41, "ORIG2"}, {55, "GOOGL"}, {38, "200"}}));
    messages.push_back(create_fix_message("8", {{17, "EXEC1"}, {11, "ORD1"}, {31, "100.50"}, {32, "50"}}));
    
    FIXMessage parsed;
    
    // Warm up
    for (int i = 0; i < 10000; ++i) {
        const auto& msg = messages[i % messages.size()];
        parser_->parse(msg.c_str(), msg.length(), parsed);
    }
    
    // Measure
    const int iterations = 1000000;
    uint64_t total_cycles = 0;
    
    uint64_t start = rdtsc();
    for (int i = 0; i < iterations; ++i) {
        const auto& msg = messages[i % messages.size()];
        parser_->parse(msg.c_str(), msg.length(), parsed);
    }
    uint64_t end = rdtsc();
    
    total_cycles = end - start;
    double avg_cycles = static_cast<double>(total_cycles) / iterations;
    double avg_ns = avg_cycles / 3.0; // Assuming 3GHz
    
    std::cout << "FIX Parser Performance (mixed messages):\n"
              << "  Average: " << avg_ns << " ns per message\n";
    
    EXPECT_LT(avg_ns, 100.0);
}

// Test concurrent parsing (thread safety)
TEST_F(FIXParserTest, ThreadSafety) {
    const int num_threads = 4;
    const int messages_per_thread = 100000;
    
    std::vector<std::thread> threads;
    std::atomic<int> errors{0};
    
    auto worker = [&](int thread_id) {
        utils::set_cpu_affinity(thread_id);
        FIXParser local_parser;
        
        for (int i = 0; i < messages_per_thread; ++i) {
            auto msg = create_fix_message("D", {
                {11, "THR" + std::to_string(thread_id) + "_" + std::to_string(i)},
                {55, "AAPL"},
                {54, "1"},
                {38, std::to_string(100 + i)},
                {44, std::to_string(150.0 + i * 0.01)}
            });
            
            FIXMessage parsed;
            if (!local_parser.parse(msg.c_str(), msg.length(), parsed)) {
                errors++;
            }
        }
    };
    
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker, i);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    EXPECT_EQ(errors.load(), 0);
}

// Test edge cases
TEST_F(FIXParserTest, VeryLongOrderID) {
    std::string long_id(256, 'X');
    auto msg = create_fix_message("D", {
        {11, long_id},
        {55, "AAPL"},
        {54, "1"},
        {38, "100"}
    });
    
    FIXMessage parsed;
    EXPECT_TRUE(parser_->parse(msg.c_str(), msg.length(), parsed));
    EXPECT_EQ(parsed.order_id.length(), 256);
}

TEST_F(FIXParserTest, SpecialCharactersInFields) {
    auto msg = create_fix_message("D", {
        {11, "ORDER|123"},
        {55, "BRK.A"},
        {54, "1"},
        {38, "100"}
    });
    
    FIXMessage parsed;
    EXPECT_TRUE(parser_->parse(msg.c_str(), msg.length(), parsed));
    EXPECT_EQ(parsed.order_id, "ORDER|123");
    EXPECT_EQ(parsed.symbol, "BRK.A");
}

TEST_F(FIXParserTest, RepeatingGroupsHandling) {
    // Test with NoPartyIDs repeating group
    auto msg = create_fix_message("D", {
        {11, "ORDER123"},
        {55, "AAPL"},
        {54, "1"},
        {38, "100"},
        {453, "2"},  // NoPartyIDs
        {448, "BROKER1"},
        {447, "D"},
        {448, "BROKER2"},
        {447, "D"}
    });
    
    FIXMessage parsed;
    EXPECT_TRUE(parser_->parse(msg.c_str(), msg.length(), parsed));
}

// Benchmark different message sizes
TEST_F(FIXParserTest, BenchmarkMessageSizes) {
    struct TestCase {
        const char* name;
        int num_fields;
    };
    
    TestCase cases[] = {
        {"Small (5 fields)", 5},
        {"Medium (15 fields)", 15},
        {"Large (30 fields)", 30}
    };
    
    for (const auto& tc : cases) {
        std::vector<std::pair<int, std::string>> fields;
        fields.push_back({11, "ORDER123"});
        fields.push_back({55, "AAPL"});
        fields.push_back({54, "1"});
        
        // Add extra fields
        for (int i = 3; i < tc.num_fields; ++i) {
            fields.push_back({1000 + i, "VALUE" + std::to_string(i)});
        }
        
        auto msg = create_fix_message("D", fields);
        FIXMessage parsed;
        
        // Warm up
        for (int i = 0; i < 10000; ++i) {
            parser_->parse(msg.c_str(), msg.length(), parsed);
        }
        
        // Measure
        const int iterations = 100000;
        uint64_t start = rdtsc();
        for (int i = 0; i < iterations; ++i) {
            parser_->parse(msg.c_str(), msg.length(), parsed);
        }
        uint64_t end = rdtsc();
        
        double avg_cycles = static_cast<double>(end - start) / iterations;
        double avg_ns = avg_cycles / 3.0;
        
        std::cout << "FIX Parser - " << tc.name << ": " 
                  << avg_ns << " ns per message\n";
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
