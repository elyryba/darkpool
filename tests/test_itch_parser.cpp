#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <thread>
#include <vector>
#include <cstring>
#include <arpa/inet.h>
#include "darkpool/protocols/itch_parser.hpp"
#include "darkpool/utils/cpu_affinity.hpp"

using namespace darkpool;
using namespace darkpool::protocols;

class ITCHParserTest : public ::testing::Test {
protected:
    void SetUp() override {
        parser_ = std::make_unique<ITCHParser>();
        
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
    
    // Helper to create ITCH messages
    std::vector<uint8_t> create_system_event(char event_code) {
        std::vector<uint8_t> msg(12);
        msg[0] = 0;  // Length MSB
        msg[1] = 11; // Length LSB
        msg[2] = 'S'; // Message type
        
        // Stock locate
        *reinterpret_cast<uint16_t*>(&msg[3]) = htons(1);
        
        // Tracking number
        *reinterpret_cast<uint16_t*>(&msg[5]) = htons(1);
        
        // Timestamp (6 bytes)
        uint64_t timestamp = 34200000000000; // 9:30 AM in nanoseconds
        msg[7] = (timestamp >> 40) & 0xFF;
        msg[8] = (timestamp >> 32) & 0xFF;
        msg[9] = (timestamp >> 24) & 0xFF;
        msg[10] = (timestamp >> 16) & 0xFF;
        msg[11] = (timestamp >> 8) & 0xFF;
        msg[12] = timestamp & 0xFF;
        
        msg.push_back(event_code);
        return msg;
    }
    
    std::vector<uint8_t> create_add_order(uint64_t order_ref, bool buy_side,
                                         uint32_t shares, const std::string& symbol,
                                         uint32_t price) {
        std::vector<uint8_t> msg(36);
        msg[0] = 0;
        msg[1] = 35; // Length
        msg[2] = 'A'; // Add Order
        
        // Stock locate
        *reinterpret_cast<uint16_t*>(&msg[3]) = htons(1);
        
        // Tracking number
        *reinterpret_cast<uint16_t*>(&msg[5]) = htons(1);
        
        // Timestamp
        uint64_t timestamp = 34200000000000;
        msg[7] = (timestamp >> 40) & 0xFF;
        msg[8] = (timestamp >> 32) & 0xFF;
        msg[9] = (timestamp >> 24) & 0xFF;
        msg[10] = (timestamp >> 16) & 0xFF;
        msg[11] = (timestamp >> 8) & 0xFF;
        msg[12] = timestamp & 0xFF;
        
        // Order reference number (8 bytes)
        *reinterpret_cast<uint64_t*>(&msg[13]) = htobe64(order_ref);
        
        // Buy/Sell indicator
        msg[21] = buy_side ? 'B' : 'S';
        
        // Shares
        *reinterpret_cast<uint32_t*>(&msg[22]) = htonl(shares);
        
        // Stock symbol (8 bytes, padded with spaces)
        std::string padded_symbol = symbol;
        padded_symbol.resize(8, ' ');
        std::memcpy(&msg[26], padded_symbol.c_str(), 8);
        
        // Price
        *reinterpret_cast<uint32_t*>(&msg[34]) = htonl(price);
        
        return msg;
    }
    
    std::vector<uint8_t> create_order_executed(uint64_t order_ref,
                                              uint32_t executed_shares,
                                              uint64_t match_number) {
        std::vector<uint8_t> msg(31);
        msg[0] = 0;
        msg[1] = 30; // Length
        msg[2] = 'E'; // Order Executed
        
        // Stock locate
        *reinterpret_cast<uint16_t*>(&msg[3]) = htons(1);
        
        // Tracking number
        *reinterpret_cast<uint16_t*>(&msg[5]) = htons(1);
        
        // Timestamp
        uint64_t timestamp = 34200000000000;
        msg[7] = (timestamp >> 40) & 0xFF;
        msg[8] = (timestamp >> 32) & 0xFF;
        msg[9] = (timestamp >> 24) & 0xFF;
        msg[10] = (timestamp >> 16) & 0xFF;
        msg[11] = (timestamp >> 8) & 0xFF;
        msg[12] = timestamp & 0xFF;
        
        // Order reference number
        *reinterpret_cast<uint64_t*>(&msg[13]) = htobe64(order_ref);
        
        // Executed shares
        *reinterpret_cast<uint32_t*>(&msg[21]) = htonl(executed_shares);
        
        // Match number
        *reinterpret_cast<uint64_t*>(&msg[25]) = htobe64(match_number);
        
        return msg;
    }
    
    std::vector<uint8_t> create_order_cancel(uint64_t order_ref,
                                            uint32_t canceled_shares) {
        std::vector<uint8_t> msg(23);
        msg[0] = 0;
        msg[1] = 22; // Length
        msg[2] = 'X'; // Order Cancel
        
        // Stock locate
        *reinterpret_cast<uint16_t*>(&msg[3]) = htons(1);
        
        // Tracking number
        *reinterpret_cast<uint16_t*>(&msg[5]) = htons(1);
        
        // Timestamp
        uint64_t timestamp = 34200000000000;
        msg[7] = (timestamp >> 40) & 0xFF;
        msg[8] = (timestamp >> 32) & 0xFF;
        msg[9] = (timestamp >> 24) & 0xFF;
        msg[10] = (timestamp >> 16) & 0xFF;
        msg[11] = (timestamp >> 8) & 0xFF;
        msg[12] = timestamp & 0xFF;
        
        // Order reference number
        *reinterpret_cast<uint64_t*>(&msg[13]) = htobe64(order_ref);
        
        // Canceled shares
        *reinterpret_cast<uint32_t*>(&msg[21]) = htonl(canceled_shares);
        
        return msg;
    }
    
    std::vector<uint8_t> create_order_replace(uint64_t original_ref,
                                             uint64_t new_ref,
                                             uint32_t shares,
                                             uint32_t price) {
        std::vector<uint8_t> msg(35);
        msg[0] = 0;
        msg[1] = 34; // Length
        msg[2] = 'U'; // Order Replace
        
        // Stock locate
        *reinterpret_cast<uint16_t*>(&msg[3]) = htons(1);
        
        // Tracking number
        *reinterpret_cast<uint16_t*>(&msg[5]) = htons(1);
        
        // Timestamp
        uint64_t timestamp = 34200000000000;
        msg[7] = (timestamp >> 40) & 0xFF;
        msg[8] = (timestamp >> 32) & 0xFF;
        msg[9] = (timestamp >> 24) & 0xFF;
        msg[10] = (timestamp >> 16) & 0xFF;
        msg[11] = (timestamp >> 8) & 0xFF;
        msg[12] = timestamp & 0xFF;
        
        // Original order reference
        *reinterpret_cast<uint64_t*>(&msg[13]) = htobe64(original_ref);
        
        // New order reference
        *reinterpret_cast<uint64_t*>(&msg[21]) = htobe64(new_ref);
        
        // Shares
        *reinterpret_cast<uint32_t*>(&msg[29]) = htonl(shares);
        
        // Price
        *reinterpret_cast<uint32_t*>(&msg[33]) = htonl(price);
        
        return msg;
    }
    
    // CPU cycle counter
    inline uint64_t rdtsc() {
        uint32_t lo, hi;
        __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
        return ((uint64_t)hi << 32) | lo;
    }
    
    std::unique_ptr<ITCHParser> parser_;
};

// Test basic message parsing
TEST_F(ITCHParserTest, ParseSystemEvent) {
    auto msg = create_system_event('O'); // Start of messages
    
    ITCHMessage parsed;
    ASSERT_TRUE(parser_->parse(msg.data(), msg.size(), parsed));
    
    EXPECT_EQ(parsed.type, ITCHMessageType::SYSTEM_EVENT);
    EXPECT_EQ(parsed.system_event.event_code, 'O');
    EXPECT_EQ(parsed.timestamp, 34200000000000ULL);
}

TEST_F(ITCHParserTest, ParseAddOrder) {
    auto msg = create_add_order(123456789, true, 100, "AAPL", 15050); // $150.50
    
    ITCHMessage parsed;
    ASSERT_TRUE(parser_->parse(msg.data(), msg.size(), parsed));
    
    EXPECT_EQ(parsed.type, ITCHMessageType::ADD_ORDER);
    EXPECT_EQ(parsed.add_order.order_reference_number, 123456789ULL);
    EXPECT_EQ(parsed.add_order.buy_sell_indicator, 'B');
    EXPECT_EQ(parsed.add_order.shares, 100U);
    EXPECT_EQ(std::string(parsed.add_order.stock, 8), "AAPL    ");
    EXPECT_EQ(parsed.add_order.price, 15050U);
}

TEST_F(ITCHParserTest, ParseOrderExecuted) {
    auto msg = create_order_executed(123456789, 50, 987654321);
    
    ITCHMessage parsed;
    ASSERT_TRUE(parser_->parse(msg.data(), msg.size(), parsed));
    
    EXPECT_EQ(parsed.type, ITCHMessageType::ORDER_EXECUTED);
    EXPECT_EQ(parsed.order_executed.order_reference_number, 123456789ULL);
    EXPECT_EQ(parsed.order_executed.executed_shares, 50U);
    EXPECT_EQ(parsed.order_executed.match_number, 987654321ULL);
}

TEST_F(ITCHParserTest, ParseOrderCancel) {
    auto msg = create_order_cancel(123456789, 100);
    
    ITCHMessage parsed;
    ASSERT_TRUE(parser_->parse(msg.data(), msg.size(), parsed));
    
    EXPECT_EQ(parsed.type, ITCHMessageType::ORDER_CANCEL);
    EXPECT_EQ(parsed.order_cancel.order_reference_number, 123456789ULL);
    EXPECT_EQ(parsed.order_cancel.canceled_shares, 100U);
}

TEST_F(ITCHParserTest, ParseOrderReplace) {
    auto msg = create_order_replace(123456789, 987654321, 200, 15100);
    
    ITCHMessage parsed;
    ASSERT_TRUE(parser_->parse(msg.data(), msg.size(), parsed));
    
    EXPECT_EQ(parsed.type, ITCHMessageType::ORDER_REPLACE);
    EXPECT_EQ(parsed.order_replace.original_order_reference_number, 123456789ULL);
    EXPECT_EQ(parsed.order_replace.new_order_reference_number, 987654321ULL);
    EXPECT_EQ(parsed.order_replace.shares, 200U);
    EXPECT_EQ(parsed.order_replace.price, 15100U);
}

// Test endianness handling
TEST_F(ITCHParserTest, EndiannessConversion) {
    // Create message with known values
    uint64_t order_ref = 0x123456789ABCDEF0ULL;
    uint32_t shares = 0x12345678U;
    uint32_t price = 0xABCDEF01U;
    
    auto msg = create_add_order(order_ref, true, shares, "TEST", price);
    
    ITCHMessage parsed;
    ASSERT_TRUE(parser_->parse(msg.data(), msg.size(), parsed));
    
    // Verify values were correctly converted from network byte order
    EXPECT_EQ(parsed.add_order.order_reference_number, order_ref);
    EXPECT_EQ(parsed.add_order.shares, shares);
    EXPECT_EQ(parsed.add_order.price, price);
}

// Test message boundaries
TEST_F(ITCHParserTest, MessageBoundaries) {
    // Test exact size
    auto msg = create_add_order(1, true, 100, "AAPL", 15000);
    ITCHMessage parsed;
    EXPECT_TRUE(parser_->parse(msg.data(), msg.size(), parsed));
    
    // Test truncated message
    EXPECT_FALSE(parser_->parse(msg.data(), msg.size() - 1, parsed));
    
    // Test extra bytes (should still parse)
    msg.push_back(0xFF);
    EXPECT_TRUE(parser_->parse(msg.data(), msg.size() - 1, parsed));
}

// Test invalid messages
TEST_F(ITCHParserTest, InvalidMessageType) {
    std::vector<uint8_t> msg(10);
    msg[0] = 0;
    msg[1] = 9;
    msg[2] = 'Z'; // Invalid type
    
    ITCHMessage parsed;
    EXPECT_FALSE(parser_->parse(msg.data(), msg.size(), parsed));
}

TEST_F(ITCHParserTest, InvalidMessageLength) {
    std::vector<uint8_t> msg(5);
    msg[0] = 0xFF; // Invalid length
    msg[1] = 0xFF;
    msg[2] = 'S';
    
    ITCHMessage parsed;
    EXPECT_FALSE(parser_->parse(msg.data(), msg.size(), parsed));
}

TEST_F(ITCHParserTest, EmptyMessage) {
    ITCHMessage parsed;
    EXPECT_FALSE(parser_->parse(nullptr, 0, parsed));
}

// Test timestamp precision
TEST_F(ITCHParserTest, TimestampPrecision) {
    // Create message with precise timestamp (nanoseconds since midnight)
    std::vector<uint8_t> msg = create_system_event('O');
    
    // Set specific timestamp: 09:30:15.123456789
    uint64_t timestamp = (9 * 3600 + 30 * 60 + 15) * 1000000000ULL + 123456789ULL;
    msg[7] = (timestamp >> 40) & 0xFF;
    msg[8] = (timestamp >> 32) & 0xFF;
    msg[9] = (timestamp >> 24) & 0xFF;
    msg[10] = (timestamp >> 16) & 0xFF;
    msg[11] = (timestamp >> 8) & 0xFF;
    msg[12] = timestamp & 0xFF;
    
    ITCHMessage parsed;
    ASSERT_TRUE(parser_->parse(msg.data(), msg.size(), parsed));
    EXPECT_EQ(parsed.timestamp, timestamp);
}

// Performance tests
TEST_F(ITCHParserTest, PerformanceSingleMessage) {
    auto msg = create_add_order(123456789, true, 100, "AAPL", 15050);
    ITCHMessage parsed;
    
    // Warm up
    for (int i = 0; i < 100000; ++i) {
        parser_->parse(msg.data(), msg.size(), parsed);
    }
    
    // Measure
    const int iterations = 1000000;
    std::vector<uint64_t> cycles;
    cycles.reserve(iterations);
    
    for (int i = 0; i < iterations; ++i) {
        uint64_t start = rdtsc();
        parser_->parse(msg.data(), msg.size(), parsed);
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
    
    std::cout << "ITCH Parser Performance (single message):\n"
              << "  P50:  " << p50 / cycles_per_ns << " ns\n"
              << "  P95:  " << p95 / cycles_per_ns << " ns\n"
              << "  P99:  " << p99 / cycles_per_ns << " ns\n"
              << "  P99.9: " << p999 / cycles_per_ns << " ns\n";
    
    // Verify <50ns target at P99
    EXPECT_LT(p99 / cycles_per_ns, 50.0);
}

TEST_F(ITCHParserTest, PerformanceMixedMessages) {
    // Create different message types
    std::vector<std::vector<uint8_t>> messages;
    messages.push_back(create_system_event('O'));
    messages.push_back(create_add_order(1, true, 100, "AAPL", 15000));
    messages.push_back(create_order_executed(1, 50, 1000));
    messages.push_back(create_order_cancel(1, 50));
    messages.push_back(create_order_replace(1, 2, 200, 15100));
    
    ITCHMessage parsed;
    
    // Warm up
    for (int i = 0; i < 100000; ++i) {
        const auto& msg = messages[i % messages.size()];
        parser_->parse(msg.data(), msg.size(), parsed);
    }
    
    // Measure
    const int iterations = 1000000;
    uint64_t start = rdtsc();
    for (int i = 0; i < iterations; ++i) {
        const auto& msg = messages[i % messages.size()];
        parser_->parse(msg.data(), msg.size(), parsed);
    }
    uint64_t end = rdtsc();
    
    double avg_cycles = static_cast<double>(end - start) / iterations;
    double avg_ns = avg_cycles / 3.0; // Assuming 3GHz
    
    std::cout << "ITCH Parser Performance (mixed messages):\n"
              << "  Average: " << avg_ns << " ns per message\n";
    
    EXPECT_LT(avg_ns, 50.0);
}

// Test order book update sequence
TEST_F(ITCHParserTest, OrderBookUpdateSequence) {
    std::vector<ITCHMessage> messages;
    ITCHMessage parsed;
    
    // Add order
    auto add_msg = create_add_order(1, true, 1000, "AAPL", 15000);
    ASSERT_TRUE(parser_->parse(add_msg.data(), add_msg.size(), parsed));
    messages.push_back(parsed);
    
    // Partial execution
    auto exec_msg = create_order_executed(1, 300, 1001);
    ASSERT_TRUE(parser_->parse(exec_msg.data(), exec_msg.size(), parsed));
    messages.push_back(parsed);
    
    // Replace remaining
    auto replace_msg = create_order_replace(1, 2, 700, 15005);
    ASSERT_TRUE(parser_->parse(replace_msg.data(), replace_msg.size(), parsed));
    messages.push_back(parsed);
    
    // Cancel remaining
    auto cancel_msg = create_order_cancel(2, 700);
    ASSERT_TRUE(parser_->parse(cancel_msg.data(), cancel_msg.size(), parsed));
    messages.push_back(parsed);
    
    // Verify sequence makes sense
    EXPECT_EQ(messages[0].type, ITCHMessageType::ADD_ORDER);
    EXPECT_EQ(messages[1].type, ITCHMessageType::ORDER_EXECUTED);
    EXPECT_EQ(messages[2].type, ITCHMessageType::ORDER_REPLACE);
    EXPECT_EQ(messages[3].type, ITCHMessageType::ORDER_CANCEL);
}

// Test trading status messages
TEST_F(ITCHParserTest, TradingStatusMessages) {
    // Stock Trading Action
    std::vector<uint8_t> msg(25);
    msg[0] = 0;
    msg[1] = 24;
    msg[2] = 'H'; // Stock Trading Action
    
    // Fill in required fields
    *reinterpret_cast<uint16_t*>(&msg[3]) = htons(1); // Stock locate
    *reinterpret_cast<uint16_t*>(&msg[5]) = htons(1); // Tracking
    
    // Timestamp
    uint64_t timestamp = 34200000000000ULL;
    msg[7] = (timestamp >> 40) & 0xFF;
    msg[8] = (timestamp >> 32) & 0xFF;
    msg[9] = (timestamp >> 24) & 0xFF;
    msg[10] = (timestamp >> 16) & 0xFF;
    msg[11] = (timestamp >> 8) & 0xFF;
    msg[12] = timestamp & 0xFF;
    
    // Stock symbol
    std::memcpy(&msg[13], "AAPL    ", 8);
    
    // Trading state
    msg[21] = 'T'; // Trading
    msg[22] = ' '; // Reserved
    msg[23] = 'N'; // Reason: Normal
    
    ITCHMessage parsed;
    ASSERT_TRUE(parser_->parse(msg.data(), msg.size(), parsed));
    EXPECT_EQ(parsed.type, ITCHMessageType::STOCK_TRADING_ACTION);
}

// Benchmark message size impact
TEST_F(ITCHParserTest, BenchmarkMessageSizes) {
    struct TestCase {
        std::string name;
        std::function<std::vector<uint8_t>()> creator;
    };
    
    TestCase cases[] = {
        {"System Event (small)", [this]() { return create_system_event('O'); }},
        {"Add Order (medium)", [this]() { return create_add_order(1, true, 100, "AAPL", 15000); }},
        {"Order Replace (large)", [this]() { return create_order_replace(1, 2, 100, 15000); }}
    };
    
    for (const auto& tc : cases) {
        auto msg = tc.creator();
        ITCHMessage parsed;
        
        // Warm up
        for (int i = 0; i < 100000; ++i) {
            parser_->parse(msg.data(), msg.size(), parsed);
        }
        
        // Measure
        const int iterations = 1000000;
        uint64_t start = rdtsc();
        for (int i = 0; i < iterations; ++i) {
            parser_->parse(msg.data(), msg.size(), parsed);
        }
        uint64_t end = rdtsc();
        
        double avg_cycles = static_cast<double>(end - start) / iterations;
        double avg_ns = avg_cycles / 3.0;
        
        std::cout << "ITCH Parser - " << tc.name << ": " 
                  << avg_ns << " ns per message (size: " << msg.size() << " bytes)\n";
    }
}

// Test concurrent parsing
TEST_F(ITCHParserTest, ThreadSafety) {
    const int num_threads = 4;
    const int messages_per_thread = 100000;
    
    std::vector<std::thread> threads;
    std::atomic<int> errors{0};
    
    auto worker = [&](int thread_id) {
        utils::set_cpu_affinity(thread_id);
        ITCHParser local_parser;
        
        for (int i = 0; i < messages_per_thread; ++i) {
            auto msg = create_add_order(thread_id * 1000000 + i, 
                                       i % 2 == 0, 
                                       100 + i, 
                                       "TEST", 
                                       10000 + i);
            
            ITCHMessage parsed;
            if (!local_parser.parse(msg.data(), msg.size(), parsed)) {
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

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
