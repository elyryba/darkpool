#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <thread>
#include <vector>
#include "darkpool/strategies/dark_pool_strategy.hpp"
#include "darkpool/strategies/execution_optimizer.hpp"
#include "darkpool/strategies/cross_venue_optimizer.hpp"
#include "darkpool/utils/cpu_affinity.hpp"

using namespace darkpool;
using namespace darkpool::strategies;

class StrategyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize strategies
        dark_pool_strategy_ = std::make_unique<DarkPoolStrategy>();
        execution_optimizer_ = std::make_unique<ExecutionOptimizer>();
        cross_venue_optimizer_ = std::make_unique<CrossVenueOptimizer>();
        
        // Pin to CPU
        utils::set_cpu_affinity(0);
        
        // Warm up
        volatile uint64_t dummy = 0;
        for (int i = 0; i < 1000000; ++i) {
            dummy += i;
        }
    }
    
    void TearDown() override {
        dark_pool_strategy_.reset();
        execution_optimizer_.reset();
        cross_venue_optimizer_.reset();
    }
    
    // Create test market context
    MarketContext create_market_context(const std::string& symbol,
                                       double mid_price,
                                       double spread,
                                       uint64_t volume,
                                       double volatility) {
        MarketContext ctx;
        ctx.symbol = symbol;
        ctx.timestamp = std::chrono::system_clock::now();
        ctx.mid_price = mid_price;
        ctx.bid_price = mid_price - spread / 2;
        ctx.ask_price = mid_price + spread / 2;
        ctx.bid_size = 1000;
        ctx.ask_size = 1000;
        ctx.volume = volume;
        ctx.volatility = volatility;
        ctx.vwap = mid_price;
        ctx.market_phase = MarketPhase::CONTINUOUS_TRADING;
        
        // Add order book depth
        for (int i = 0; i < 10; ++i) {
            ctx.bid_levels[i] = {mid_price - spread/2 - i*0.01, 1000*(i+1)};
            ctx.ask_levels[i] = {mid_price + spread/2 + i*0.01, 1000*(i+1)};
        }
        
        return ctx;
    }
    
    // Create test order
    Order create_order(const std::string& id,
                      const std::string& symbol,
                      Side side,
                      uint32_t quantity,
                      OrderType type = OrderType::LIMIT,
                      double price = 0.0) {
        Order order;
        order.order_id = id;
        order.symbol = symbol;
        order.side = side;
        order.quantity = quantity;
        order.order_type = type;
        order.limit_price = price;
        order.time_in_force = TimeInForce::DAY;
        order.creation_time = std::chrono::system_clock::now();
        
        return order;
    }
    
    inline uint64_t rdtsc() {
        uint32_t lo, hi;
        __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
        return ((uint64_t)hi << 32) | lo;
    }
    
    std::unique_ptr<DarkPoolStrategy> dark_pool_strategy_;
    std::unique_ptr<ExecutionOptimizer> execution_optimizer_;
    std::unique_ptr<CrossVenueOptimizer> cross_venue_optimizer_;
};

// Test strategy state transitions
TEST_F(StrategyTest, StateTransitions) {
    auto order = create_order("STATE1", "AAPL", Side::BUY, 1000, OrderType::LIMIT, 150.00);
    auto context = create_market_context("AAPL", 150.00, 0.02, 1000000, 0.15);
    
    // Initial state
    EXPECT_EQ(dark_pool_strategy_->get_state(), StrategyState::IDLE);
    
    // Submit order
    auto decision = dark_pool_strategy_->on_order_request(order, context);
    ASSERT_TRUE(decision.has_value());
    EXPECT_EQ(dark_pool_strategy_->get_state(), StrategyState::WORKING);
    
    // Partial fill
    Execution exec;
    exec.order_id = "STATE1";
    exec.quantity = 300;
    exec.price = 150.00;
    dark_pool_strategy_->on_execution(exec, context);
    EXPECT_EQ(dark_pool_strategy_->get_state(), StrategyState::WORKING);
    
    // Complete fill
    exec.quantity = 700;
    dark_pool_strategy_->on_execution(exec, context);
    EXPECT_EQ(dark_pool_strategy_->get_state(), StrategyState::COMPLETED);
    
    // Reset
    dark_pool_strategy_->reset();
    EXPECT_EQ(dark_pool_strategy_->get_state(), StrategyState::IDLE);
}

// Test order placement logic
TEST_F(StrategyTest, OrderPlacementLogic) {
    auto order = create_order("PLACE1", "MSFT", Side::BUY, 5000, OrderType::LIMIT, 350.00);
    auto context = create_market_context("MSFT", 350.00, 0.02, 5000000, 0.20);
    
    // Dark pool strategy should split large orders
    auto dp_decision = dark_pool_strategy_->on_order_request(order, context);
    ASSERT_TRUE(dp_decision.has_value());
    EXPECT_EQ(dp_decision->action, StrategyAction::SPLIT_ORDER);
    EXPECT_GT(dp_decision->child_orders.size(), 1);
    
    // Verify child orders sum to parent
    uint32_t total_qty = 0;
    for (const auto& child : dp_decision->child_orders) {
        total_qty += child.quantity;
        EXPECT_LE(child.quantity, 1000); // Max child size
    }
    EXPECT_EQ(total_qty, 5000);
    
    // Execution optimizer should use algorithms
    auto eo_decision = execution_optimizer_->on_order_request(order, context);
    ASSERT_TRUE(eo_decision.has_value());
    EXPECT_EQ(eo_decision->execution_algo, ExecutionAlgo::VWAP);
    EXPECT_GT(eo_decision->schedule.size(), 0);
}

// Test risk limit enforcement
TEST_F(StrategyTest, RiskLimitEnforcement) {
    auto context = create_market_context("RISK", 100.00, 0.05, 1000000, 0.30);
    
    // Set risk limits
    RiskLimits limits;
    limits.max_order_value = 100000.0;
    limits.max_position_value = 500000.0;
    limits.max_daily_volume = 1000000;
    
    dark_pool_strategy_->set_risk_limits(limits);
    execution_optimizer_->set_risk_limits(limits);
    
    // Order within limits
    auto small_order = create_order("RISK1", "RISK", Side::BUY, 500, OrderType::LIMIT, 100.00);
    auto decision = dark_pool_strategy_->on_order_request(small_order, context);
    ASSERT_TRUE(decision.has_value());
    EXPECT_NE(decision->action, StrategyAction::REJECT);
    
    // Order exceeding limits
    auto large_order = create_order("RISK2", "RISK", Side::BUY, 2000, OrderType::LIMIT, 100.00);
    decision = dark_pool_strategy_->on_order_request(large_order, context);
    ASSERT_TRUE(decision.has_value());
    EXPECT_EQ(decision->action, StrategyAction::REJECT);
    EXPECT_EQ(decision->reject_reason, "Order exceeds risk limits");
}

// Test multi-venue routing
TEST_F(StrategyTest, MultiVenueRouting) {
    auto order = create_order("VENUE1", "GOOGL", Side::BUY, 1000, OrderType::LIMIT, 2750.00);
    auto context = create_market_context("GOOGL", 2750.00, 0.05, 500000, 0.25);
    
    // Add venue information
    std::vector<VenueQuote> venue_quotes = {
        {"NASDAQ", 2749.98, 2750.02, 500, 600, 0.0025, 1000000},
        {"NYSE", 2749.99, 2750.01, 400, 500, 0.0020, 800000},
        {"BATS", 2749.97, 2750.03, 600, 700, 0.0030, 1200000},
        {"DARKPOOL1", 2750.00, 2750.00, 1000, 1000, 0.0010, 0}
    };
    
    auto routing = cross_venue_optimizer_->optimize_routing(order, venue_quotes);
    ASSERT_FALSE(routing.empty());
    
    // Should prefer venues with better prices and lower fees
    uint32_t total_routed = 0;
    for (const auto& [venue, qty] : routing) {
        total_routed += qty;
        std::cout << "Route " << qty << " to " << venue << "\n";
    }
    EXPECT_EQ(total_routed, 1000);
    
    // Dark pool should get some allocation
    EXPECT_GT(routing["DARKPOOL1"], 0);
}

// Test anti-gaming mechanisms
TEST_F(StrategyTest, AntiGamingMechanisms) {
    auto order = create_order("GAME1", "SPY", Side::BUY, 10000, OrderType::LIMIT, 440.00);
    auto context = create_market_context("SPY", 440.00, 0.01, 50000000, 0.12);
    
    // Detect potential gaming patterns
    context.recent_trades = {
        {439.99, 100, context.timestamp - std::chrono::seconds(1)},
        {440.01, 100, context.timestamp - std::chrono::seconds(2)},
        {439.99, 100, context.timestamp - std::chrono::seconds(3)},
        {440.01, 100, context.timestamp - std::chrono::seconds(4)}
    };
    
    auto decision = dark_pool_strategy_->on_order_request(order, context);
    ASSERT_TRUE(decision.has_value());
    
    // Should detect ping-pong pattern and adjust strategy
    EXPECT_TRUE(decision->anti_gaming_enabled);
    EXPECT_GT(decision->randomization_factor, 0.0);
    EXPECT_TRUE(decision->use_hidden_orders);
}

// Test execution algorithms
TEST_F(StrategyTest, ExecutionAlgorithms) {
    auto order = create_order("ALGO1", "AAPL", Side::BUY, 50000, OrderType::LIMIT, 150.00);
    auto context = create_market_context("AAPL", 150.00, 0.02, 10000000, 0.18);
    
    // Test VWAP
    execution_optimizer_->set_algorithm(ExecutionAlgo::VWAP);
    auto vwap_decision = execution_optimizer_->on_order_request(order, context);
    ASSERT_TRUE(vwap_decision.has_value());
    EXPECT_EQ(vwap_decision->execution_algo, ExecutionAlgo::VWAP);
    EXPECT_GT(vwap_decision->schedule.size(), 10); // Multiple time slices
    
    // Verify VWAP schedule follows volume curve
    uint32_t total_scheduled = 0;
    for (const auto& slice : vwap_decision->schedule) {
        total_scheduled += slice.quantity;
        EXPECT_GT(slice.participation_rate, 0.0);
        EXPECT_LE(slice.participation_rate, 0.20); // Max 20% participation
    }
    EXPECT_EQ(total_scheduled, 50000);
    
    // Test TWAP
    execution_optimizer_->set_algorithm(ExecutionAlgo::TWAP);
    auto twap_decision = execution_optimizer_->on_order_request(order, context);
    ASSERT_TRUE(twap_decision.has_value());
    EXPECT_EQ(twap_decision->execution_algo, ExecutionAlgo::TWAP);
    
    // TWAP should have uniform slices
    uint32_t expected_slice = 50000 / twap_decision->schedule.size();
    for (const auto& slice : twap_decision->schedule) {
        EXPECT_NEAR(slice.quantity, expected_slice, expected_slice * 0.1);
    }
    
    // Test Implementation Shortfall
    execution_optimizer_->set_algorithm(ExecutionAlgo::IMPLEMENTATION_SHORTFALL);
    auto is_decision = execution_optimizer_->on_order_request(order, context);
    ASSERT_TRUE(is_decision.has_value());
    EXPECT_EQ(is_decision->execution_algo, ExecutionAlgo::IMPLEMENTATION_SHORTFALL);
    
    // IS should front-load execution
    EXPECT_GT(is_decision->schedule[0].quantity, is_decision->schedule.back().quantity);
}

// Test position tracking
TEST_F(StrategyTest, PositionTracking) {
    std::string symbol = "TRACK";
    auto context = create_market_context(symbol, 100.00, 0.02, 1000000, 0.20);
    
    // Initial position
    EXPECT_EQ(dark_pool_strategy_->get_position(symbol), 0);
    
    // Buy order
    auto buy_order = create_order("POS1", symbol, Side::BUY, 1000, OrderType::LIMIT, 100.00);
    dark_pool_strategy_->on_order_request(buy_order, context);
    
    // Execute
    Execution buy_exec;
    buy_exec.order_id = "POS1";
    buy_exec.quantity = 1000;
    buy_exec.price = 100.00;
    dark_pool_strategy_->on_execution(buy_exec, context);
    
    EXPECT_EQ(dark_pool_strategy_->get_position(symbol), 1000);
    
    // Sell order
    auto sell_order = create_order("POS2", symbol, Side::SELL, 300, OrderType::LIMIT, 100.50);
    dark_pool_strategy_->on_order_request(sell_order, context);
    
    // Execute
    Execution sell_exec;
    sell_exec.order_id = "POS2";
    sell_exec.quantity = 300;
    sell_exec.price = 100.50;
    dark_pool_strategy_->on_execution(sell_exec, context);
    
    EXPECT_EQ(dark_pool_strategy_->get_position(symbol), 700);
}

// Test performance requirements
TEST_F(StrategyTest, PerformanceRequirements) {
    auto order = create_order("PERF1", "AAPL", Side::BUY, 1000, OrderType::LIMIT, 150.00);
    auto context = create_market_context("AAPL", 150.00, 0.02, 5000000, 0.15);
    
    // Warm up
    for (int i = 0; i < 10000; ++i) {
        dark_pool_strategy_->on_order_request(order, context);
        dark_pool_strategy_->reset();
    }
    
    // Measure dark pool strategy
    const int iterations = 100000;
    std::vector<uint64_t> cycles;
    cycles.reserve(iterations);
    
    for (int i = 0; i < iterations; ++i) {
        uint64_t start = rdtsc();
        auto decision = dark_pool_strategy_->on_order_request(order, context);
        uint64_t end = rdtsc();
        cycles.push_back(end - start);
        dark_pool_strategy_->reset();
    }
    
    std::sort(cycles.begin(), cycles.end());
    uint64_t p50 = cycles[iterations * 0.50];
    uint64_t p95 = cycles[iterations * 0.95];
    uint64_t p99 = cycles[iterations * 0.99];
    
    const double cycles_per_ns = 3.0;
    
    std::cout << "Strategy Performance:\n"
              << "  P50: " << p50 / cycles_per_ns << " ns\n"
              << "  P95: " << p95 / cycles_per_ns << " ns\n"
              << "  P99: " << p99 / cycles_per_ns << " ns\n";
    
    // Should make decisions in <1ms (1,000,000 ns)
    EXPECT_LT(p99 / cycles_per_ns, 1000000.0);
}

// Test market phase handling
TEST_F(StrategyTest, MarketPhaseHandling) {
    auto order = create_order("PHASE1", "AAPL", Side::BUY, 1000, OrderType::LIMIT, 150.00);
    
    // Pre-market
    auto context = create_market_context("AAPL", 150.00, 0.05, 100000, 0.25);
    context.market_phase = MarketPhase::PRE_MARKET;
    
    auto decision = execution_optimizer_->on_order_request(order, context);
    ASSERT_TRUE(decision.has_value());
    EXPECT_TRUE(decision->delay_until_open);
    
    // Opening auction
    context.market_phase = MarketPhase::OPENING_AUCTION;
    decision = execution_optimizer_->on_order_request(order, context);
    ASSERT_TRUE(decision.has_value());
    EXPECT_TRUE(decision->participate_in_auction);
    
    // Continuous trading
    context.market_phase = MarketPhase::CONTINUOUS_TRADING;
    decision = execution_optimizer_->on_order_request(order, context);
    ASSERT_TRUE(decision.has_value());
    EXPECT_FALSE(decision->delay_until_open);
    
    // Closing auction
    context.market_phase = MarketPhase::CLOSING_AUCTION;
    decision = execution_optimizer_->on_order_request(order, context);
    ASSERT_TRUE(decision.has_value());
    EXPECT_TRUE(decision->participate_in_close);
}

// Test adaptive behavior
TEST_F(StrategyTest, AdaptiveBehavior) {
    auto order = create_order("ADAPT1", "TSLA", Side::BUY, 5000, OrderType::LIMIT, 900.00);
    auto context = create_market_context("TSLA", 900.00, 0.10, 2000000, 0.35);
    
    // Simulate changing market conditions
    for (int i = 0; i < 10; ++i) {
        // Increase volatility
        context.volatility = 0.35 + i * 0.05;
        context.timestamp += std::chrono::seconds(60);
        
        auto decision = execution_optimizer_->on_order_request(order, context);
        ASSERT_TRUE(decision.has_value());
        
        // Strategy should adapt to higher volatility
        if (i > 5) {
            EXPECT_GT(decision->urgency_score, 0.7);
            EXPECT_LT(decision->max_participation_rate, 0.15); // More passive
        }
    }
}

// Test order modification handling
TEST_F(StrategyTest, OrderModificationHandling) {
    auto original = create_order("MOD1", "AAPL", Side::BUY, 1000, OrderType::LIMIT, 150.00);
    auto context = create_market_context("AAPL", 150.00, 0.02, 5000000, 0.15);
    
    // Submit original
    auto decision = dark_pool_strategy_->on_order_request(original, context);
    ASSERT_TRUE(decision.has_value());
    
    // Modify quantity
    OrderModification mod;
    mod.order_id = "MOD1";
    mod.new_quantity = 2000;
    mod.new_price = 150.50;
    
    auto mod_decision = dark_pool_strategy_->on_order_modify(mod, context);
    ASSERT_TRUE(mod_decision.has_value());
    EXPECT_EQ(mod_decision->action, StrategyAction::MODIFY_ORDER);
    
    // Cancel
    auto cancel_decision = dark_pool_strategy_->on_order_cancel("MOD1", context);
    ASSERT_TRUE(cancel_decision.has_value());
    EXPECT_EQ(cancel_decision->action, StrategyAction::CANCEL_ORDER);
}

// Test concurrent strategy execution
TEST_F(StrategyTest, ConcurrentExecution) {
    const int num_threads = 4;
    const int orders_per_thread = 1000;
    
    std::vector<std::thread> threads;
    std::atomic<int> decisions_made{0};
    std::atomic<int> rejections{0};
    
    auto worker = [&](int thread_id) {
        utils::set_cpu_affinity(thread_id);
        
        for (int i = 0; i < orders_per_thread; ++i) {
            auto order = create_order(
                "T" + std::to_string(thread_id) + "_" + std::to_string(i),
                "CONC", Side::BUY, 100 + i % 900, OrderType::LIMIT, 100.00
            );
            
            auto context = create_market_context("CONC", 100.00, 0.02, 1000000, 0.20);
            
            auto decision = dark_pool_strategy_->on_order_request(order, context);
            if (decision.has_value()) {
                decisions_made++;
                if (decision->action == StrategyAction::REJECT) {
                    rejections++;
                }
            }
        }
    };
    
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker, i);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    EXPECT_EQ(decisions_made.load(), num_threads * orders_per_thread);
    std::cout << "Concurrent execution: " << decisions_made << " decisions, "
              << rejections << " rejections\n";
}

// Benchmark different order sizes
TEST_F(StrategyTest, BenchmarkOrderSizes) {
    struct TestCase {
        const char* name;
        uint32_t quantity;
        double expected_latency_us;
    };
    
    TestCase cases[] = {
        {"Small order (100)", 100, 100},
        {"Medium order (1K)", 1000, 200},
        {"Large order (10K)", 10000, 500},
        {"Huge order (100K)", 100000, 1000}
    };
    
    auto context = create_market_context("BENCH", 100.00, 0.02, 10000000, 0.15);
    
    for (const auto& tc : cases) {
        auto order = create_order("BENCH1", "BENCH", Side::BUY, tc.quantity, 
                                 OrderType::LIMIT, 100.00);
        
        // Warm up
        for (int i = 0; i < 1000; ++i) {
            execution_optimizer_->on_order_request(order, context);
            execution_optimizer_->reset();
        }
        
        // Measure
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10000; ++i) {
            execution_optimizer_->on_order_request(order, context);
            execution_optimizer_->reset();
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        double avg_us = duration.count() / 10000.0 / 1000.0;
        
        std::cout << "Strategy benchmark - " << tc.name << ": " << avg_us << " μs\n";
        EXPECT_LT(avg_us, tc.expected_latency_us);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
