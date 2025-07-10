#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <thread>
#include <vector>
#include "darkpool/core/trade_to_quote_ratio.hpp"
#include "darkpool/utils/cpu_affinity.hpp"

using namespace darkpool;
using namespace darkpool::core;

class TQRCalculatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        calculator_ = std::make_unique<TradeToQuoteRatioCalculator>();
        
        // Pin to CPU for consistent timing
        utils::set_cpu_affinity(0);
        
        // Warm up
        volatile uint64_t dummy = 0;
        for (int i = 0; i < 1000000; ++i) {
            dummy += i;
        }
    }
    
    void TearDown() override {
        calculator_.reset();
    }
    
    // Create test trade
    Trade create_trade(const std::string& symbol, double price, uint32_t size) {
        Trade trade;
        trade.symbol = symbol;
        trade.price = price;
        trade.size = size;
        trade.timestamp = std::chrono::system_clock::now();
        trade.aggressor_side = (rand() % 2) ? Side::BUY : Side::SELL;
        
        return trade;
    }
    
    // Create test quote
    Quote create_quote(const std::string& symbol, double bid, double ask, 
                      uint32_t bid_size, uint32_t ask_size) {
        Quote quote;
        quote.symbol = symbol;
        quote.bid_price = bid;
        quote.ask_price = ask;
        quote.bid_size = bid_size;
        quote.ask_size = ask_size;
        quote.timestamp = std::chrono::system_clock::now();
        
        return quote;
    }
    
    // Generate realistic quote stream
    std::vector<Quote> generate_quote_stream(const std::string& symbol,
                                           double base_price,
                                           int count,
                                           double volatility = 0.001) {
        std::vector<Quote> quotes;
        std::mt19937 gen(42);
        std::normal_distribution<double> dist(0.0, volatility);
        
        double mid = base_price;
        
        for (int i = 0; i < count; ++i) {
            mid += dist(gen);
            double spread = 0.01 + std::abs(dist(gen)) * 10; // 1-11 cents
            
            quotes.push_back(create_quote(
                symbol,
                mid - spread/2,
                mid + spread/2,
                1000 + (gen() % 9000),
                1000 + (gen() % 9000)
            ));
        }
        
        return quotes;
    }
    
    // Generate trade stream
    std::vector<Trade> generate_trade_stream(const std::string& symbol,
                                           double base_price,
                                           int count,
                                           double avg_size = 100) {
        std::vector<Trade> trades;
        std::mt19937 gen(42);
        std::normal_distribution<double> price_dist(base_price, base_price * 0.001);
        std::exponential_distribution<double> size_dist(1.0 / avg_size);
        
        for (int i = 0; i < count; ++i) {
            trades.push_back(create_trade(
                symbol,
                price_dist(gen),
                std::max(1u, static_cast<uint32_t>(size_dist(gen)))
            ));
        }
        
        return trades;
    }
    
    inline uint64_t rdtsc() {
        uint32_t lo, hi;
        __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
        return ((uint64_t)hi << 32) | lo;
    }
    
    std::unique_ptr<TradeToQuoteRatioCalculator> calculator_;
};

// Test basic TQR calculation
TEST_F(TQRCalculatorTest, BasicTQRCalculation) {
    std::string symbol = "AAPL";
    
    // Add quotes
    for (int i = 0; i < 100; ++i) {
        auto quote = create_quote(symbol, 150.00 - 0.01, 150.00 + 0.01, 1000, 1000);
        calculator_->add_quote(quote);
    }
    
    // Add trades (10 trades for 100 quotes = 0.1 TQR)
    for (int i = 0; i < 10; ++i) {
        auto trade = create_trade(symbol, 150.00, 100);
        calculator_->add_trade(trade);
    }
    
    auto result = calculator_->get_tqr(symbol);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->ratio, 0.1, 0.01);
    EXPECT_EQ(result->trade_count, 10);
    EXPECT_EQ(result->quote_count, 100);
}

// Test rolling window accuracy
TEST_F(TQRCalculatorTest, RollingWindowAccuracy) {
    std::string symbol = "MSFT";
    auto base_time = std::chrono::system_clock::now();
    
    // Fill window with old data
    for (int i = 0; i < 1000; ++i) {
        Quote q = create_quote(symbol, 349.99, 350.01, 1000, 1000);
        q.timestamp = base_time - std::chrono::seconds(300); // 5 min old
        calculator_->add_quote(q);
    }
    
    // Add recent data
    for (int i = 0; i < 100; ++i) {
        Quote q = create_quote(symbol, 349.99, 350.01, 1000, 1000);
        q.timestamp = base_time - std::chrono::seconds(30); // 30 sec old
        calculator_->add_quote(q);
        
        if (i % 10 == 0) {
            Trade t = create_trade(symbol, 350.00, 100);
            t.timestamp = base_time - std::chrono::seconds(30);
            calculator_->add_trade(t);
        }
    }
    
    // Check window contains only recent data
    auto result = calculator_->get_tqr(symbol, std::chrono::seconds(60));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->quote_count, 100); // Only recent quotes
    EXPECT_EQ(result->trade_count, 10);  // Only recent trades
    EXPECT_NEAR(result->ratio, 0.1, 0.01);
}

// Test anomaly threshold calibration
TEST_F(TQRCalculatorTest, AnomalyThresholdCalibration) {
    std::string symbol = "GOOGL";
    
    // Normal market - high quote activity
    for (int i = 0; i < 1000; ++i) {
        calculator_->add_quote(create_quote(symbol, 2749.99, 2750.01, 500, 500));
        
        // Normal TQR ~0.05
        if (i % 20 == 0) {
            calculator_->add_trade(create_trade(symbol, 2750.00, 100));
        }
    }
    
    // Check normal condition
    auto normal_result = calculator_->calculate_anomaly(symbol);
    ASSERT_TRUE(normal_result.has_value());
    EXPECT_FALSE(normal_result->is_anomaly);
    EXPECT_NEAR(normal_result->tqr, 0.05, 0.01);
    
    // Sudden spike in trades (potential sweep)
    for (int i = 0; i < 50; ++i) {
        calculator_->add_trade(create_trade(symbol, 2750.00 + i * 0.01, 1000));
    }
    
    // Should detect anomaly
    auto anomaly_result = calculator_->calculate_anomaly(symbol);
    ASSERT_TRUE(anomaly_result.has_value());
    EXPECT_TRUE(anomaly_result->is_anomaly);
    EXPECT_GT(anomaly_result->tqr, 0.5); // High TQR
    EXPECT_EQ(anomaly_result->anomaly_type, AnomalyType::HIGH_TQR);
}

// Test performance under high message rates
TEST_F(TQRCalculatorTest, HighMessageRatePerformance) {
    std::string symbol = "SPY";
    
    // Generate high-frequency data
    auto quotes = generate_quote_stream(symbol, 440.00, 100000);
    auto trades = generate_trade_stream(symbol, 440.00, 10000);
    
    // Warm up
    for (int i = 0; i < 1000; ++i) {
        calculator_->add_quote(quotes[i % quotes.size()]);
        if (i % 10 == 0) {
            calculator_->add_trade(trades[i / 10 % trades.size()]);
        }
    }
    
    // Measure quote processing
    std::vector<uint64_t> quote_cycles;
    quote_cycles.reserve(quotes.size());
    
    for (const auto& quote : quotes) {
        uint64_t start = rdtsc();
        calculator_->add_quote(quote);
        uint64_t end = rdtsc();
        quote_cycles.push_back(end - start);
    }
    
    // Measure trade processing
    std::vector<uint64_t> trade_cycles;
    trade_cycles.reserve(trades.size());
    
    for (const auto& trade : trades) {
        uint64_t start = rdtsc();
        calculator_->add_trade(trade);
        uint64_t end = rdtsc();
        trade_cycles.push_back(end - start);
    }
    
    // Calculate percentiles
    std::sort(quote_cycles.begin(), quote_cycles.end());
    std::sort(trade_cycles.begin(), trade_cycles.end());
    
    const double cycles_per_ns = 3.0;
    
    std::cout << "TQR Calculator Performance:\n"
              << "  Quote P99: " << quote_cycles[quotes.size() * 0.99] / cycles_per_ns << " ns\n"
              << "  Trade P99: " << trade_cycles[trades.size() * 0.99] / cycles_per_ns << " ns\n";
    
    // Should process each message in <200ns
    EXPECT_LT(quote_cycles[quotes.size() * 0.99] / cycles_per_ns, 200.0);
    EXPECT_LT(trade_cycles[trades.size() * 0.99] / cycles_per_ns, 200.0);
}

// Test memory stability
TEST_F(TQRCalculatorTest, MemoryStability) {
    std::string symbol = "AMZN";
    
    // Get initial memory
    size_t initial_memory = calculator_->get_memory_usage();
    
    // Process millions of messages
    for (int i = 0; i < 1000000; ++i) {
        if (i % 10 == 0) {
            calculator_->add_trade(create_trade(symbol, 3300.00 + (i % 100) * 0.01, 100));
        } else {
            calculator_->add_quote(create_quote(symbol, 3299.99, 3300.01, 1000, 1000));
        }
        
        // Check memory periodically
        if (i % 100000 == 0 && i > 0) {
            size_t current_memory = calculator_->get_memory_usage();
            size_t growth = current_memory - initial_memory;
            
            // Memory should stabilize (circular buffer)
            EXPECT_LT(growth, 10 * 1024 * 1024) << "Memory leak at iteration " << i;
        }
    }
}

// Test window edge cases
TEST_F(TQRCalculatorTest, WindowEdgeCases) {
    std::string symbol = "EDGE";
    
    // Empty window
    auto result = calculator_->get_tqr(symbol);
    EXPECT_FALSE(result.has_value());
    
    // Only quotes, no trades
    for (int i = 0; i < 100; ++i) {
        calculator_->add_quote(create_quote(symbol, 99.99, 100.01, 1000, 1000));
    }
    
    result = calculator_->get_tqr(symbol);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->ratio, 0.0);
    EXPECT_EQ(result->trade_count, 0);
    EXPECT_EQ(result->quote_count, 100);
    
    // Only trades, no quotes (unusual but possible)
    calculator_->reset();
    for (int i = 0; i < 10; ++i) {
        calculator_->add_trade(create_trade(symbol, 100.00, 100));
    }
    
    result = calculator_->get_tqr(symbol);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(std::isinf(result->ratio)); // Infinity
    EXPECT_EQ(result->trade_count, 10);
    EXPECT_EQ(result->quote_count, 0);
}

// Test multiple symbols
TEST_F(TQRCalculatorTest, MultipleSymbols) {
    std::vector<std::string> symbols = {"AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"};
    std::map<std::string, double> expected_tqrs;
    
    // Create different TQR patterns for each symbol
    for (size_t i = 0; i < symbols.size(); ++i) {
        const auto& symbol = symbols[i];
        double tqr = 0.05 * (i + 1); // 0.05, 0.10, 0.15, 0.20, 0.25
        expected_tqrs[symbol] = tqr;
        
        // Add quotes
        for (int j = 0; j < 1000; ++j) {
            calculator_->add_quote(create_quote(symbol, 100.0 + i, 100.0 + i + 0.02, 
                                               1000, 1000));
        }
        
        // Add trades to achieve target TQR
        int trade_count = static_cast<int>(1000 * tqr);
        for (int j = 0; j < trade_count; ++j) {
            calculator_->add_trade(create_trade(symbol, 100.0 + i + 0.01, 100));
        }
    }
    
    // Verify each symbol maintains correct TQR
    for (const auto& symbol : symbols) {
        auto result = calculator_->get_tqr(symbol);
        ASSERT_TRUE(result.has_value());
        EXPECT_NEAR(result->ratio, expected_tqrs[symbol], 0.01)
            << "Failed for symbol: " << symbol;
    }
}

// Test pattern detection
TEST_F(TQRCalculatorTest, PatternDetection) {
    std::string symbol = "PATTERN";
    
    // Create spoofing pattern: high quotes, low trades
    for (int cycle = 0; cycle < 10; ++cycle) {
        // Burst of quotes
        for (int i = 0; i < 100; ++i) {
            calculator_->add_quote(create_quote(symbol, 
                                               100.00 - 0.01 * (i % 5),
                                               100.00 + 0.01 * (i % 5),
                                               10000, 10000));
        }
        
        // Very few trades
        calculator_->add_trade(create_trade(symbol, 100.00, 10));
    }
    
    auto pattern = calculator_->detect_pattern(symbol);
    ASSERT_TRUE(pattern.has_value());
    EXPECT_EQ(pattern->pattern_type, TQRPattern::POTENTIAL_SPOOFING);
    EXPECT_GT(pattern->confidence, 0.7);
    
    // Create sweep pattern: burst of trades
    calculator_->reset();
    
    // Normal activity
    for (int i = 0; i < 100; ++i) {
        calculator_->add_quote(create_quote(symbol, 99.99, 100.01, 1000, 1000));
        if (i % 20 == 0) {
            calculator_->add_trade(create_trade(symbol, 100.00, 100));
        }
    }
    
    // Sudden sweep
    for (int i = 0; i < 50; ++i) {
        calculator_->add_trade(create_trade(symbol, 100.00 + i * 0.01, 1000));
    }
    
    pattern = calculator_->detect_pattern(symbol);
    ASSERT_TRUE(pattern.has_value());
    EXPECT_EQ(pattern->pattern_type, TQRPattern::LIQUIDITY_SWEEP);
}

// Test real-time updates
TEST_F(TQRCalculatorTest, RealTimeUpdates) {
    std::string symbol = "REAL";
    
    // Simulate real-time feed
    auto start_time = std::chrono::steady_clock::now();
    int quote_count = 0;
    int trade_count = 0;
    
    while (std::chrono::steady_clock::now() - start_time < std::chrono::milliseconds(100)) {
        // High frequency quotes
        calculator_->add_quote(create_quote(symbol, 99.99, 100.01, 1000, 1000));
        quote_count++;
        
        // Occasional trades
        if (quote_count % 50 == 0) {
            calculator_->add_trade(create_trade(symbol, 100.00, 100));
            trade_count++;
        }
        
        // Check TQR periodically
        if (quote_count % 100 == 0) {
            auto result = calculator_->get_tqr(symbol);
            ASSERT_TRUE(result.has_value());
            EXPECT_GT(result->quote_count, 0);
        }
    }
    
    std::cout << "Real-time test: " << quote_count << " quotes, " 
              << trade_count << " trades processed\n";
}

// Test concurrent access
TEST_F(TQRCalculatorTest, ConcurrentAccess) {
    const int num_threads = 4;
    const int messages_per_thread = 10000;
    
    std::vector<std::thread> threads;
    std::atomic<int> total_quotes{0};
    std::atomic<int> total_trades{0};
    
    auto worker = [&](int thread_id) {
        utils::set_cpu_affinity(thread_id);
        std::string symbol = "SYM" + std::to_string(thread_id);
        
        for (int i = 0; i < messages_per_thread; ++i) {
            if (i % 10 == 0) {
                calculator_->add_trade(create_trade(symbol, 100.00 + thread_id, 100));
                total_trades++;
            } else {
                calculator_->add_quote(create_quote(symbol, 
                                                   100.00 + thread_id - 0.01,
                                                   100.00 + thread_id + 0.01,
                                                   1000, 1000));
                total_quotes++;
            }
        }
    };
    
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker, i);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    // Verify all messages processed
    EXPECT_EQ(total_quotes.load() + total_trades.load(), 
              num_threads * messages_per_thread);
    
    // Check each symbol has correct data
    for (int i = 0; i < num_threads; ++i) {
        auto result = calculator_->get_tqr("SYM" + std::to_string(i));
        ASSERT_TRUE(result.has_value());
        EXPECT_GT(result->quote_count, 0);
        EXPECT_GT(result->trade_count, 0);
    }
}

// Benchmark different window sizes
TEST_F(TQRCalculatorTest, BenchmarkWindowSizes) {
    std::string symbol = "BENCH";
    
    // Pre-populate with data
    for (int i = 0; i < 100000; ++i) {
        calculator_->add_quote(create_quote(symbol, 99.99, 100.01, 1000, 1000));
        if (i % 20 == 0) {
            calculator_->add_trade(create_trade(symbol, 100.00, 100));
        }
    }
    
    std::vector<std::chrono::seconds> windows = {
        std::chrono::seconds(10),
        std::chrono::seconds(30),
        std::chrono::seconds(60),
        std::chrono::seconds(300)
    };
    
    for (auto window : windows) {
        // Warm up
        for (int i = 0; i < 1000; ++i) {
            calculator_->get_tqr(symbol, window);
        }
        
        // Measure
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10000; ++i) {
            calculator_->get_tqr(symbol, window);
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        double avg_ns = duration.count() / 10000.0;
        
        std::cout << "TQR calculation for " << window.count() 
                  << "s window: " << avg_ns << " ns\n";
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
