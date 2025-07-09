#pragma once

#include "darkpool/strategies/strategy_base.hpp"
#include <set>
#include <chrono>

namespace darkpool::strategies {

class CrossVenueOptimizer : public StrategyBase {
public:
    struct VenueCharacteristics {
        double avg_spread_bps;
        double fill_probability;
        double avg_fill_size;
        double market_impact_bps;
        double fee_bps;
        double rebate_bps;
        int64_t avg_latency_us;
        double hidden_liquidity_ratio;
        bool supports_hidden_orders;
        bool supports_pegging;
        std::set<OrderType> supported_order_types;
    };
    
    struct CrossVenueConfig : Config {
        // Venue optimization
        bool enable_smart_routing = true;
        bool enable_spray_orders = true;
        bool enable_liquidity_aggregation = true;
        double min_venue_score = 0.5;
        
        // Cost optimization
        double max_total_cost_bps = 3.0;
        bool optimize_for_rebates = true;
        bool consider_opportunity_cost = true;
        
        // Latency management
        int64_t max_venue_latency_us = 1000;
        bool enable_latency_arbitrage = true;
        int64_t cancel_timeout_ms = 100;
        
        // Risk controls
        double max_venue_concentration = 0.4;
        size_t max_simultaneous_venues = 5;
        double min_fill_probability = 0.6;
        
        // Learning parameters
        bool enable_venue_learning = true;
        size_t learning_window = 1000;
        double learning_rate = 0.1;
        double explore_probability = 0.1;
    };
    
    explicit CrossVenueOptimizer(const CrossVenueConfig& config);
    ~CrossVenueOptimizer() override;
    
    void on_market_data(const MarketMessage& msg) noexcept override;
    void on_anomaly(const Anomaly& anomaly) noexcept override;
    void on_fill(const Trade& trade, uint64_t order_id) noexcept override;
    void on_order_update(const ExecutionOrder& order) noexcept override;
    
    // Venue management
    void register_venue(const std::string& venue_id,
                       const VenueCharacteristics& chars) noexcept;
    void update_venue_characteristics(const std::string& venue_id,
                                    const VenueCharacteristics& chars) noexcept;
    void enable_venue(const std::string& venue_id) noexcept;
    void disable_venue(const std::string& venue_id) noexcept;
    
    // Cross-venue execution
    struct CrossVenueOrder {
        uint32_t total_quantity;
        Side side;
        std::unordered_map<std::string, uint32_t> venue_allocations;
        OrderType order_type;
        double urgency;
        int64_t timeout_ms;
    };
    
    void execute_cross_venue(const CrossVenueOrder& order) noexcept;
    
    // Analytics
    struct VenueAnalytics {
        std::string venue_id;
        double realized_spread_bps;
        double effective_fill_rate;
        double avg_queue_position;
        double price_improvement_bps;
        double total_cost_bps;
        double information_leakage_score;
        uint64_t total_volume;
        std::chrono::milliseconds avg_time_to_fill;
    };
    
    std::vector<VenueAnalytics> get_venue_analytics() const noexcept;
    
private:
    struct VenueState {
        VenueCharacteristics characteristics;
        std::atomic<bool> enabled{true};
        
        // Performance tracking
        std::atomic<uint64_t> orders_sent{0};
        std::atomic<uint64_t> orders_filled{0};
        std::atomic<uint64_t> orders_cancelled{0};
        std::atomic<uint64_t> total_filled_quantity{0};
        std::atomic<double> total_cost{0.0};
        std::atomic<int64_t> total_latency_us{0};
        
        // Learning state
        std::deque<double> recent_spreads;
        std::deque<double> recent_fill_rates;
        std::deque<double> recent_impacts;
        double score{0.5};
        
        // Queue position modeling
        std::atomic<double> avg_queue_position{0.5};
        std::atomic<uint32_t> queue_model_updates{0};
    };
    
    struct SplitOrder {
        uint64_t parent_id;
        CrossVenueOrder original;
        std::unordered_map<uint64_t, std::string> child_orders; // order_id -> venue
        std::unordered_map<std::string, uint32_t> filled_by_venue;
        int64_t start_time;
        std::atomic<uint32_t> total_filled{0};
        std::atomic<bool> completed{false};
    };
    
    CrossVenueConfig config_;
    
    // Venue management
    std::unordered_map<std::string, std::unique_ptr<VenueState>> venues_;
    mutable std::mutex venues_mutex_;
    
    // Order tracking
    std::unordered_map<uint64_t, SplitOrder> split_orders_;
    mutable std::mutex orders_mutex_;
    
    // Optimization engine
    std::unordered_map<std::string, uint32_t> optimize_allocation(
        uint32_t quantity, Side side) noexcept;
    
    double calculate_venue_score(const VenueState& venue,
                               uint32_t quantity, Side side) noexcept;
    
    // Smart order routing
    void route_order_slice(const SplitOrder& parent,
                          const std::string& venue_id,
                          uint32_t quantity) noexcept;
    
    void rebalance_unfilled_quantity(uint64_t parent_id) noexcept;
    
    // Latency arbitrage
    bool detect_latency_opportunity(const std::string& fast_venue,
                                   const std::string& slow_venue) noexcept;
    
    void execute_latency_arbitrage(const std::string& fast_venue,
                                  const std::string& slow_venue,
                                  uint32_t quantity, Side side) noexcept;
    
    // Learning and adaptation
    void update_venue_model(const std::string& venue_id,
                           const Trade& fill,
                           const ExecutionOrder& order) noexcept;
    
    void update_queue_position_model(const std::string& venue_id,
                                   const Trade& fill) noexcept;
    
    // Cost analysis
    double calculate_total_cost(const std::unordered_map<std::string, uint32_t>& allocation,
                               Side side) noexcept;
    
    double estimate_information_leakage(const std::string& venue_id,
                                      uint32_t quantity) noexcept;
};

// Multi-venue market maker
class CrossVenueMarketMaker : public CrossVenueOptimizer {
public:
    struct MMConfig : CrossVenueConfig {
        double target_spread_bps = 2.0;
        double inventory_limit = 100000;
        double skew_multiplier = 0.5;
        bool enable_auto_hedging = true;
        double min_edge_bps = 0.5;
        std::unordered_map<std::string, double> venue_priorities;
    };
    
    explicit CrossVenueMarketMaker(const MMConfig& config);
    
    void update_quotes() noexcept;
    void manage_inventory() noexcept;
    
private:
    void calculate_optimal_quotes(std::unordered_map<std::string, Quote>& quotes) noexcept;
    void hedge_inventory(int32_t position) noexcept;
};

} 
