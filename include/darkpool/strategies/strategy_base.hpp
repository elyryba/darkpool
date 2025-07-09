#pragma once

#include "darkpool/types.hpp"
#include <memory>
#include <vector>
#include <unordered_map>
#include <atomic>

namespace darkpool::strategies {

// Order types for execution
enum class OrderType {
    MARKET = 0,
    LIMIT,
    HIDDEN,          // Dark pool order
    ICEBERG,         // Hidden size order
    PEG,             // Pegged to market
    VWAP,            // Volume-weighted average price
    TWAP,            // Time-weighted average price
    CLOSE,           // Market-on-close
    CONDITIONAL      // Triggered by conditions
};

// Venue types
enum class VenueType {
    LIT_EXCHANGE = 0,
    DARK_POOL,
    ECN,
    ATS,             // Alternative Trading System
    SDP,             // Single Dealer Platform
    INTERNALIZER,
    RFQ              // Request for Quote
};

// Strategy execution state
enum class StrategyState {
    IDLE = 0,
    ACTIVE,
    PAUSED,
    COMPLETED,
    CANCELLED,
    ERROR
};

struct ExecutionOrder {
    uint64_t order_id;
    uint32_t symbol;
    OrderType type;
    Side side;
    int64_t price;      // Limit price (0 for market)
    uint32_t quantity;
    uint32_t filled_quantity;
    uint32_t hidden_quantity;  // For iceberg orders
    VenueType venue;
    std::string venue_id;
    int64_t timestamp;
    int64_t expire_time;
    
    // Execution constraints
    uint32_t min_quantity;
    uint32_t max_floor;        // Max to show
    double participation_rate; // % of volume
    int64_t price_limit;       // Don't cross this price
    
    // Tracking
    std::vector<Trade> fills;
    double avg_fill_price;
    int64_t last_update;
};

struct StrategyMetrics {
    // Performance metrics
    double pnl;
    double realized_pnl;
    double unrealized_pnl;
    double sharpe_ratio;
    double max_drawdown;
    
    // Execution metrics
    uint64_t orders_sent;
    uint64_t orders_filled;
    uint64_t orders_cancelled;
    double fill_rate;
    double avg_fill_time_ms;
    
    // Cost analysis
    double total_commission;
    double total_spread_cost;
    double total_market_impact;
    double implementation_shortfall;
    
    // Risk metrics
    double position_value;
    double var_95;           // Value at Risk
    double expected_shortfall;
    double beta_to_market;
    
    // Venue analysis
    std::unordered_map<VenueType, double> venue_fill_rates;
    std::unordered_map<VenueType, double> venue_costs;
};

class StrategyBase {
public:
    struct Config {
        uint32_t symbol;
        double max_position_size;
        double max_order_size;
        double risk_limit;
        double participation_rate_limit;
        bool enable_dark_pools = true;
        bool enable_hidden_orders = true;
        bool enable_smart_routing = true;
        size_t max_open_orders = 100;
        int64_t order_timeout_ms = 30000;
        double min_fill_size = 100;
    };
    
    explicit StrategyBase(const Config& config);
    virtual ~StrategyBase() = default;
    
    // Core strategy interface
    virtual void on_market_data(const MarketMessage& msg) noexcept = 0;
    virtual void on_anomaly(const Anomaly& anomaly) noexcept = 0;
    virtual void on_fill(const Trade& trade, uint64_t order_id) noexcept = 0;
    virtual void on_order_update(const ExecutionOrder& order) noexcept = 0;
    
    // Strategy control
    virtual bool start() noexcept;
    virtual void stop() noexcept;
    virtual void pause() noexcept;
    virtual void resume() noexcept;
    
    // Order management
    uint64_t send_order(const ExecutionOrder& order) noexcept;
    bool cancel_order(uint64_t order_id) noexcept;
    bool modify_order(uint64_t order_id, int64_t new_price, 
                     uint32_t new_quantity) noexcept;
    void cancel_all_orders() noexcept;
    
    // Position and risk
    int32_t get_position() const noexcept;
    double get_pnl() const noexcept;
    bool check_risk_limits() const noexcept;
    
    // Metrics
    StrategyMetrics get_metrics() const noexcept;
    void reset_metrics() noexcept;
    
    // State
    StrategyState get_state() const noexcept;
    
protected:
    // Override these for strategy-specific logic
    virtual bool should_send_order(const ExecutionOrder& order) noexcept;
    virtual void on_risk_limit_breach() noexcept;
    virtual VenueType select_venue(const ExecutionOrder& order) noexcept;
    
    // Utilities for derived classes
    double calculate_vwap(size_t lookback_ms) const noexcept;
    double calculate_spread() const noexcept;
    double estimate_market_impact(uint32_t quantity) const noexcept;
    
    // Order tracking
    std::unordered_map<uint64_t, ExecutionOrder> active_orders_;
    std::vector<ExecutionOrder> completed_orders_;
    
    // Position tracking
    std::atomic<int32_t> position_{0};
    std::atomic<double> avg_entry_price_{0.0};
    
    // Market data cache
    struct MarketCache {
        Quote last_quote;
        Trade last_trade;
        std::deque<Trade> recent_trades;
        std::deque<Quote> recent_quotes;
        int64_t last_update;
    };
    MarketCache market_cache_;
    
    Config config_;
    std::atomic<StrategyState> state_{StrategyState::IDLE};
    
private:
    mutable StrategyMetrics metrics_;
    std::atomic<uint64_t> next_order_id_{1};
    mutable std::mutex order_mutex_;
    mutable std::mutex metrics_mutex_;
    
    void update_metrics(const ExecutionOrder& order) noexcept;
    void update_position(const Trade& fill) noexcept;
};

// Risk management component
class RiskManager {
public:
    struct RiskLimits {
        double max_position_value;
        double max_loss;
        double max_daily_loss;
        double position_limit;
        double order_size_limit;
        double var_limit;
        double concentration_limit;
        double leverage_limit;
    };
    
    explicit RiskManager(const RiskLimits& limits);
    
    bool check_order(const ExecutionOrder& order, 
                    double current_position,
                    double current_pnl) const noexcept;
    
    bool check_position(double position_value,
                       double unrealized_pnl) const noexcept;
    
    void update_risk_metrics(const Trade& trade) noexcept;
    
    double calculate_var(const std::vector<double>& returns,
                        double confidence = 0.95) const noexcept;
    
private:
    RiskLimits limits_;
    std::atomic<double> daily_loss_{0.0};
    std::atomic<double> max_position_today_{0.0};
    int64_t last_reset_time_{0};
};

} 
