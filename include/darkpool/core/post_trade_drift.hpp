#pragma once

#include <vector>
#include <deque>
#include <unordered_map>
#include <shared_mutex>
#include "darkpool/types.hpp"

namespace darkpool {

class PostTradeDrift {
public:
    struct Config {
        Quantity large_trade_threshold = 10000;   // Size threshold for large trades
        size_t drift_windows[] = {10, 30, 60, 300, 600}; // Seconds to measure drift
        size_t num_windows = 5;
        double significant_drift = 0.002;         // 20 bps drift is significant
        size_t min_reference_trades = 20;         // Min trades for reference price
        double volume_percentile = 0.95;          // Large trade percentile
        bool track_all_trades = false;            // Track all or just large trades
        size_t max_tracked_trades = 1000;         // Memory limit per symbol
    };
    
    explicit PostTradeDrift(const Config& config = Config{});
    
    // Process market events
    void on_trade(const Trade& trade);
    void on_quote(const Quote& quote);
    
    // Drift analysis results
    struct DriftResult {
        double drift_10s;      // Price drift after 10 seconds
        double drift_30s;      // Price drift after 30 seconds
        double drift_60s;      // Price drift after 1 minute
        double drift_300s;     // Price drift after 5 minutes
        double drift_600s;     // Price drift after 10 minutes
        double max_drift;      // Maximum drift observed
        Timestamp max_drift_time;
        double reversion_ratio; // How much price reverted
        bool is_permanent;     // Permanent vs temporary impact
    };
    
    // Analyze drift for a specific trade
    DriftResult analyze_drift(Symbol symbol, OrderId trade_id) const;
    
    // Get aggregate drift statistics
    struct DriftStats {
        double avg_drift_60s;          // Average 1-minute drift
        double drift_std_dev;          // Drift volatility
        double information_ratio;      // Drift significance
        size_t large_trades_count;     // Number of large trades
        double avg_large_trade_size;   // Average size
        double permanent_impact_ratio; // Ratio of permanent impacts
    };
    
    DriftStats get_drift_stats(Symbol symbol) const;
    
    // Detect information leakage
    std::optional<Anomaly> check_anomaly(Symbol symbol) const;
    
    // Get recent large trades with drift
    struct LargeTradeInfo {
        OrderId trade_id;
        Timestamp timestamp;
        Price execution_price;
        Quantity size;
        Side side;
        DriftResult drift;
        double z_score;  // Drift significance
    };
    
    std::vector<LargeTradeInfo> get_large_trades(Symbol symbol, size_t count = 10) const;
    
private:
    struct TrackedTrade {
        OrderId id;
        Symbol symbol;
        Timestamp timestamp;
        Price execution_price;
        Quantity size;
        Side side;
        
        // Reference prices
        Price pre_trade_mid;      // Mid quote before trade
        Price post_trade_prices[5]; // Prices at each window
        bool window_complete[5];    // Whether window has passed
        
        // Calculated drift
        mutable DriftResult cached_drift;
        mutable bool drift_calculated = false;
    };
    
    struct PricePoint {
        Timestamp timestamp;
        Price mid_price;
        Price last_trade;
    };
    
    struct SymbolData {
        std::deque<TrackedTrade> tracked_trades;
        std::deque<Trade> recent_trades;      // For volume analysis
        std::deque<PricePoint> price_history; // For drift measurement
        Quote last_quote;
        
        // Volume statistics
        mutable double volume_threshold = 0.0;
        mutable Timestamp threshold_update = 0;
        
        // Drift statistics
        double drift_sum = 0.0;
        double drift_sum_sq = 0.0;
        size_t drift_samples = 0;
    };
    
    // Determine if trade is large
    bool is_large_trade(const Trade& trade, const SymbolData& data) const;
    
    // Update volume threshold dynamically
    void update_volume_threshold(SymbolData& data) const;
    
    // Calculate reference price before trade
    Price calculate_pre_trade_price(const Trade& trade, const SymbolData& data) const;
    
    // Update post-trade prices for tracked trades
    void update_post_trade_prices(SymbolData& data, Timestamp current_time);
    
    // Calculate drift for a tracked trade
    DriftResult calculate_drift(const TrackedTrade& trade, const SymbolData& data) const;
    
    // Check if drift indicates information leakage
    bool is_information_leakage(const DriftResult& drift, const SymbolData& data) const;
    
    // Clean old data
    void clean_old_data(SymbolData& data, Timestamp current_time);
    
    Config config_;
    mutable std::unordered_map<Symbol, SymbolData> symbol_data_;
    mutable std::shared_mutex data_mutex_;
};

} 
