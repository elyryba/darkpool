#pragma once

#include <vector>
#include <array>
#include <unordered_map>
#include <shared_mutex>
#include <chrono>
#include "darkpool/types.hpp"

namespace darkpool {

class ExecutionHeatmap {
public:
    struct Config {
        size_t price_buckets = 100;           // Number of price levels
        size_t time_buckets = 60;             // Time buckets (seconds)
        double price_bucket_width = 0.0001;   // 1 basis point buckets
        size_t max_symbols = 100;             // Maximum tracked symbols
        size_t update_interval_ms = 100;      // Update frequency
        bool track_hidden_only = false;       // Only track suspected hidden orders
        double volume_decay_rate = 0.95;      // Decay for historical data
    };
    
    explicit ExecutionHeatmap(const Config& config = Config{});
    
    // Process market events
    void on_trade(const Trade& trade);
    void on_anomaly(const Anomaly& anomaly);
    
    // Heatmap cell data
    struct HeatmapCell {
        double buy_volume = 0.0;
        double sell_volume = 0.0;
        double hidden_volume = 0.0;
        uint32_t trade_count = 0;
        uint32_t anomaly_count = 0;
        float intensity = 0.0f;  // Normalized 0-1
    };
    
    // Get heatmap data for visualization
    struct HeatmapData {
        Symbol symbol;
        Timestamp timestamp;
        Price center_price;
        Price price_range_start;
        Price price_range_end;
        std::vector<std::vector<HeatmapCell>> cells; // [time][price]
        
        // Summary statistics
        double total_volume;
        double hidden_volume_ratio;
        Price volume_weighted_price;
        size_t anomaly_count;
    };
    
    HeatmapData get_heatmap(Symbol symbol) const;
    
    // Get multiple symbols for dashboard
    std::vector<HeatmapData> get_top_symbols(size_t count = 10) const;
    
    // Real-time activity metrics
    struct ActivityMetrics {
        double trades_per_second;
        double volume_per_second;
        double anomaly_rate;
        std::array<double, 24> hourly_volume;  // 24-hour profile
        std::array<double, 5> venue_distribution; // Volume by venue
    };
    
    ActivityMetrics get_activity_metrics() const;
    
    // WebSocket-ready JSON format
    struct JsonHeatmap {
        std::string symbol;
        std::vector<std::vector<float>> intensity;  // [time][price] 0-255
        std::vector<double> price_labels;
        std::vector<std::string> time_labels;
        std::string metadata;  // JSON string with stats
    };
    
    JsonHeatmap get_json_heatmap(Symbol symbol) const;
    
private:
    struct SymbolHeatmap {
        std::array<std::array<HeatmapCell, 100>, 60> grid;  // [time][price]
        Price reference_price = 0;
        Timestamp last_update = 0;
        
        // Activity tracking
        std::deque<Trade> recent_trades;
        std::deque<Anomaly> recent_anomalies;
        
        // Statistics
        double total_volume = 0.0;
        double hidden_volume = 0.0;
        size_t total_trades = 0;
        size_t total_anomalies = 0;
        
        // Circular buffer indices
        size_t current_time_bucket = 0;
        
        void reset() {
            for (auto& row : grid) {
                for (auto& cell : row) {
                    cell = HeatmapCell{};
                }
            }
        }
    };
    
    // Map price to bucket index
    size_t price_to_bucket(Price price, Price reference) const;
    
    // Map time to bucket index
    size_t time_to_bucket(Timestamp timestamp, Timestamp reference) const;
    
    // Update heatmap with trade
    void update_heatmap(SymbolHeatmap& heatmap, const Trade& trade);
    
    // Decay old values
    void decay_heatmap(SymbolHeatmap& heatmap, Timestamp current_time);
    
    // Normalize intensities
    void normalize_intensities(SymbolHeatmap& heatmap) const;
    
    // Calculate activity score for symbol ranking
    double calculate_activity_score(const SymbolHeatmap& heatmap) const;
    
    Config config_;
    mutable std::unordered_map<Symbol, SymbolHeatmap> symbol_heatmaps_;
    mutable std::shared_mutex data_mutex_;
    
    // Global metrics
    mutable ActivityMetrics global_metrics_;
    mutable Timestamp metrics_update_time_ = 0;
};

} 
