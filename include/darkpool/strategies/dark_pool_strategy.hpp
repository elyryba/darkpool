#pragma once

#include "darkpool/strategies/strategy_base.hpp"
#include "darkpool/ml/inference_engine.hpp"
#include <queue>
#include <random>

namespace darkpool::strategies {

class DarkPoolStrategy : public StrategyBase {
public:
    struct DarkPoolConfig : Config {
        // Dark pool specific settings
        size_t num_venues = 10;
        double min_dark_fill_size = 1000;
        double max_dark_participation = 0.25;  // Max % of dark volume
        double iceberg_ratio = 0.1;           // Show only 10% of order
        double minimum_spread_bps = 1.0;      // Min spread to use dark
        
        // Anti-gaming parameters
        int64_t min_order_interval_ms = 100;
        int64_t randomization_ms = 50;
        double size_randomization = 0.1;      // +/- 10% size variation
        bool enable_anti_gaming = true;
        
        // Venue selection
        double venue_fill_rate_threshold = 0.7;
        double venue_cost_threshold_bps = 0.5;
        size_t venue_history_window = 1000;
        
        // ML integration
        bool enable_ml_routing = true;
        double ml_confidence_threshold = 0.8;
        
        // Information leakage prevention
        size_t max_venues_per_order = 3;
        double max_show_ratio = 0.05;         // Max to show vs total
        bool enable_time_slicing = true;
        size_t slice_count = 10;
    };
    
    explicit DarkPoolStrategy(const DarkPoolConfig& config);
    ~DarkPoolStrategy() override;
    
    // Strategy interface
    void on_market_data(const MarketMessage& msg) noexcept override;
    void on_anomaly(const Anomaly& anomaly) noexcept override;
    void on_fill(const Trade& trade, uint64_t order_id) noexcept override;
    void on_order_update(const ExecutionOrder& order) noexcept override;
    
    // Dark pool specific methods
    void set_target_quantity(uint32_t quantity, Side side) noexcept;
    void set_urgency(double urgency) noexcept; // 0.0 = patient, 1.0 = urgent
    
    // Venue management
    void add_dark_venue(const std::string& venue_id, 
                       VenueType type = VenueType::DARK_POOL) noexcept;
    void remove_venue(const std::string& venue_id) noexcept;
    void update_venue_metrics(const std::string& venue_id,
                            double fill_rate, double cost_bps) noexcept;
    
private:
    struct VenueInfo {
        std::string id;
        VenueType type;
        std::atomic<uint64_t> orders_sent{0};
        std::atomic<uint64_t> orders_filled{0};
        std::atomic<uint64_t> total_filled_quantity{0};
        std::atomic<double> total_cost_bps{0.0};
        std::atomic<int64_t> avg_fill_time_ms{0};
        std::atomic<int64_t> last_order_time{0};
        std::atomic<double> fill_rate{0.0};
        std::atomic<double> avg_fill_size{0.0};
        
        // Anti-gaming tracking
        std::deque<int64_t> recent_order_times;
        std::atomic<uint32_t> consecutive_rejects{0};
        
        double get_score() const noexcept;
        bool is_healthy() const noexcept;
    };
    
    struct SliceOrder {
        uint32_t total_quantity;
        uint32_t remaining_quantity;
        uint32_t slice_size;
        Side side;
        int64_t start_time;
        int64_t next_slice_time;
        double urgency;
        std::vector<uint64_t> child_orders;
    };
    
    // Configuration
    DarkPoolConfig config_;
    
    // Venue management
    std::unordered_map<std::string, std::unique_ptr<VenueInfo>> venues_;
    mutable std::mutex venue_mutex_;
    
    // Order management
    std::unordered_map<uint64_t, SliceOrder> parent_orders_;
    std::atomic<uint32_t> target_quantity_{0};
    std::atomic<Side> target_side_{Side::UNKNOWN};
    std::atomic<double> urgency_{0.5};
    
    // ML integration
    ml::InferenceEngine* ml_engine_ = nullptr;
    
    // Anti-gaming
    std::mt19937 rng_;
    std::uniform_real_distribution<> size_dist_;
    std::uniform_int_distribution<> time_dist_;
    
    // Internal methods
    std::vector<std::string> select_venues(uint32_t quantity) noexcept;
    uint32_t calculate_slice_size(const SliceOrder& parent) noexcept;
    void send_dark_orders(const SliceOrder& parent) noexcept;
    ExecutionOrder create_dark_order(const std::string& venue_id,
                                    uint32_t quantity,
                                    Side side) noexcept;
    
    // Anti-gaming logic
    uint32_t randomize_size(uint32_t base_size) noexcept;
    int64_t randomize_timing() noexcept;
    bool detect_gaming(const VenueInfo& venue) noexcept;
    void handle_suspected_gaming(const std::string& venue_id) noexcept;
    
    // Venue scoring and selection
    double score_venue(const VenueInfo& venue, uint32_t quantity) noexcept;
    void update_venue_learning(const std::string& venue_id,
                              const Trade& fill) noexcept;
    
    // Risk checks
    bool check_dark_pool_risk(const ExecutionOrder& order) noexcept;
    bool should_use_dark_pool() noexcept;
};

// Dark pool aggregator for multi-venue optimization
class DarkPoolAggregator {
public:
    struct AggregatorConfig {
        size_t max_venues = 20;
        double min_aggregate_size = 10000;
        double rebalance_threshold = 0.1;
        int64_t rebalance_interval_ms = 60000;
        bool enable_cross_venue_netting = true;
    };
    
    explicit DarkPoolAggregator(const AggregatorConfig& config);
    
    // Aggregate liquidity discovery
    struct LiquiditySnapshot {
        std::unordered_map<std::string, double> venue_liquidity;
        std::unordered_map<std::string, double> venue_spreads;
        double total_bid_liquidity;
        double total_ask_liquidity;
        double weighted_spread;
        int64_t timestamp;
    };
    
    LiquiditySnapshot get_aggregated_liquidity(uint32_t symbol) noexcept;
    
    // Optimal routing
    struct RoutingDecision {
        std::vector<std::pair<std::string, uint32_t>> venue_allocations;
        double expected_cost_bps;
        double expected_fill_rate;
        int64_t expected_fill_time_ms;
    };
    
    RoutingDecision optimize_routing(uint32_t symbol, uint32_t quantity,
                                   Side side) noexcept;
    
private:
    AggregatorConfig config_;
    std::unordered_map<uint32_t, LiquiditySnapshot> liquidity_cache_;
    mutable std::mutex cache_mutex_;
};

}
