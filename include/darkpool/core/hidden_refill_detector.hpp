#pragma once

#include <unordered_map>
#include <deque>
#include <shared_mutex>
#include "darkpool/types.hpp"

namespace darkpool {

class HiddenRefillDetector {
public:
    struct Config {
        size_t min_refills = 3;                    // Min refills to detect iceberg
        Timestamp refill_window_ms = 5000;         // Time window for refills
        double size_consistency_threshold = 0.2;    // Max deviation in refill sizes
        double price_consistency_threshold = 0.001; // Max price deviation
        size_t max_tracking_orders = 10000;        // Memory limit
        double hidden_multiplier = 5.0;            // Estimate total size multiplier
    };
    
    explicit HiddenRefillDetector(const Config& config = Config{});
    
    // Track order lifecycle
    void on_order(const Order& order);
    void on_trade(const Trade& trade);
    void on_order_cancel(OrderId order_id, Quantity canceled_quantity);
    void on_order_replace(OrderId order_id, Price new_price, Quantity new_quantity);
    
    // Detect iceberg patterns
    struct IcebergPattern {
        Symbol symbol;
        Price price_level;
        Side side;
        Quantity visible_size;      // Typical refill size
        Quantity total_executed;    // Total volume so far
        Quantity estimated_total;   // Estimated iceberg size
        size_t refill_count;
        double confidence;
        Timestamp first_seen;
        Timestamp last_refill;
    };
    
    std::vector<IcebergPattern> get_active_icebergs(Symbol symbol) const;
    std::optional<Anomaly> check_anomaly(Symbol symbol) const;
    
    // Statistics
    struct Stats {
        size_t icebergs_detected = 0;
        size_t false_positives = 0;
        Quantity hidden_volume_detected = 0;
        double avg_detection_time_ms = 0.0;
    };
    
    Stats get_stats() const { return stats_; }
    void reset_stats() { stats_ = Stats{}; }
    
private:
    struct OrderTracking {
        Symbol symbol;
        Price price;
        Side side;
        Quantity original_quantity;
        Quantity remaining_quantity;
        Timestamp first_seen;
        std::vector<Timestamp> execution_times;
        std::vector<Quantity> execution_sizes;
        bool potential_iceberg = false;
    };
    
    struct PriceLevel {
        std::deque<OrderId> order_queue;
        Quantity total_visible;
        Quantity total_executed;
        std::vector<Timestamp> refill_times;
        std::vector<Quantity> refill_sizes;
        Timestamp last_activity;
    };
    
    struct SymbolData {
        // Price -> PriceLevel mapping for each side
        std::unordered_map<Price, PriceLevel> bid_levels;
        std::unordered_map<Price, PriceLevel> ask_levels;
        
        // Active iceberg patterns
        std::vector<IcebergPattern> active_icebergs;
    };
    
    // Detect refill pattern at price level
    bool detect_refill_pattern(const PriceLevel& level) const;
    
    // Analyze execution pattern for iceberg characteristics
    bool analyze_execution_pattern(const OrderTracking& tracking) const;
    
    // Update price level with new order
    void update_price_level(PriceLevel& level, const Order& order);
    
    // Clean old data
    void clean_old_data(SymbolData& data, Timestamp current_time);
    
    Config config_;
    mutable std::shared_mutex data_mutex_;
    std::unordered_map<OrderId, OrderTracking> order_tracking_;
    mutable std::unordered_map<Symbol, SymbolData> symbol_data_;
    mutable Stats stats_;
};

} 
