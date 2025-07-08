#pragma once

#include <vector>
#include <deque>
#include <unordered_map>
#include <shared_mutex>
#include "darkpool/types.hpp"

namespace darkpool {

// Volume-Synchronized Probability of Informed Trading
class VPINCalculator {
public:
    struct Config {
        size_t volume_bucket_size = 50000;    // Volume per bucket
        size_t num_buckets = 50;              // Number of buckets for VPIN calculation
        double support_window = 0.01;         // Price support window (1%)
        size_t min_buckets_for_vpin = 10;     // Minimum buckets needed
        double toxicity_threshold = 0.3;      // VPIN threshold for toxicity
        bool use_bulk_classification = true;  // Use bulk volume classification
        double time_bar_seconds = 60.0;       // For time bars if volume is low
    };
    
    explicit VPINCalculator(const Config& config = Config{});
    
    // Process market events
    void on_trade(const Trade& trade);
    void on_quote(const Quote& quote);
    
    // Calculate VPIN
    struct VPINResult {
        double vpin;                    // Current VPIN value
        double avg_vpin;               // Average VPIN over window
        double vpin_std_dev;           // VPIN standard deviation
        size_t completed_buckets;      // Number of completed buckets
        double order_imbalance;        // Current bucket order imbalance
        Timestamp calculation_time;
        
        bool is_toxic() const { return vpin > 0.3; }
        double toxicity_zscore() const { 
            return vpin_std_dev > 0 ? (vpin - avg_vpin) / vpin_std_dev : 0.0; 
        }
    };
    
    VPINResult calculate_vpin(Symbol symbol) const;
    
    // Detect flow toxicity
    std::optional<Anomaly> check_anomaly(Symbol symbol) const;
    
    // Get VPIN time series
    struct VPINPoint {
        Timestamp timestamp;
        double vpin;
        Quantity bucket_volume;
        double buy_volume_ratio;
    };
    
    std::vector<VPINPoint> get_vpin_history(Symbol symbol, size_t num_points = 50) const;
    
    // Get current bucket progress
    struct BucketProgress {
        Quantity current_volume;
        Quantity target_volume;
        double buy_volume;
        double sell_volume;
        Price volume_weighted_price;
        size_t trade_count;
    };
    
    BucketProgress get_current_bucket(Symbol symbol) const;
    
private:
    // Volume bucket for VPIN calculation
    struct VolumeBucket {
        Timestamp start_time;
        Timestamp end_time;
        Quantity total_volume;
        double buy_volume;      // Classified buy volume
        double sell_volume;     // Classified sell volume
        Price vwap;            // Volume-weighted average price
        Price open_price;
        Price close_price;
        size_t trade_count;
        
        double get_order_imbalance() const {
            return (total_volume > 0) ? 
                std::abs(buy_volume - sell_volume) / total_volume : 0.0;
        }
    };
    
    struct CurrentBucket {
        VolumeBucket bucket;
        std::vector<Trade> trades;  // Trades in current bucket
        bool is_complete = false;
    };
    
    struct SymbolData {
        std::deque<VolumeBucket> completed_buckets;
        CurrentBucket current_bucket;
        Quote last_quote;
        
        // VPIN statistics
        double vpin_sum = 0.0;
        double vpin_sum_sq = 0.0;
        size_t vpin_samples = 0;
        
        // Price levels for bulk classification
        std::vector<Price> price_levels;
        Timestamp price_levels_update = 0;
    };
    
    // Trade classification methods
    enum class Classification {
        BUY,
        SELL,
        UNKNOWN
    };
    
    // Lee-Ready algorithm for trade classification
    Classification classify_trade_lee_ready(const Trade& trade, const Quote& quote) const;
    
    // Bulk Volume Classification (BVC)
    void classify_bulk_volume(CurrentBucket& bucket, const std::vector<Price>& price_levels) const;
    
    // Tick rule for classification
    Classification classify_trade_tick_rule(const Trade& current, const Trade& previous) const;
    
    // Update price levels for BVC
    void update_price_levels(SymbolData& data, const std::deque<Trade>& recent_trades) const;
    
    // Complete current bucket and start new one
    void complete_bucket(SymbolData& data);
    
    // Calculate support levels
    std::vector<Price> calculate_support_levels(const std::deque<Trade>& trades) const;
    
    Config config_;
    mutable std::unordered_map<Symbol, SymbolData> symbol_data_;
    mutable std::shared_mutex data_mutex_;
};

} 
