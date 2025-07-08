#include "darkpool/core/execution_heatmap.hpp"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace darkpool {

ExecutionHeatmap::ExecutionHeatmap(const Config& config) : config_(config) {
    symbol_heatmaps_.reserve(config.max_symbols);
}

void ExecutionHeatmap::on_trade(const Trade& trade) {
    std::unique_lock lock(data_mutex_);
    auto& heatmap = symbol_heatmaps_[trade.symbol];
    
    update_heatmap(heatmap, trade);
    
    // Update global metrics
    Timestamp now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    if (now - metrics_update_time_ > config_.update_interval_ms * 1000000) {
        // Update activity metrics
        size_t hour = (now / 3600000000000) % 24; // Hour of day
        global_metrics_.hourly_volume[hour] += trade.quantity;
        
        // Venue distribution
        if (static_cast<size_t>(trade.venue) < 5) {
            global_metrics_.venue_distribution[static_cast<size_t>(trade.venue)] += trade.quantity;
        }
        
        metrics_update_time_ = now;
    }
}

void ExecutionHeatmap::on_anomaly(const Anomaly& anomaly) {
    std::unique_lock lock(data_mutex_);
    auto& heatmap = symbol_heatmaps_[anomaly.symbol];
    
    heatmap.recent_anomalies.push_back(anomaly);
    heatmap.total_anomalies++;
    
    // Update anomaly count in relevant cells
    if (heatmap.reference_price > 0) {
        size_t time_bucket = time_to_bucket(anomaly.timestamp, heatmap.last_update);
        
        // Mark anomaly across price range
        for (size_t p = 0; p < config_.price_buckets; ++p) {
            heatmap.grid[time_bucket][p].anomaly_count++;
        }
    }
}

ExecutionHeatmap::HeatmapData ExecutionHeatmap::get_heatmap(Symbol symbol) const {
    std::shared_lock lock(data_mutex_);
    
    HeatmapData data;
    data.symbol = symbol;
    data.timestamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    
    auto it = symbol_heatmaps_.find(symbol);
    if (it == symbol_heatmaps_.end()) {
        return data;
    }
    
    const auto& heatmap = it->second;
    
    // Set price range
    data.center_price = heatmap.reference_price;
    data.price_range_start = heatmap.reference_price - 
        (config_.price_buckets / 2) * config_.price_bucket_width * heatmap.reference_price;
    data.price_range_end = heatmap.reference_price + 
        (config_.price_buckets / 2) * config_.price_bucket_width * heatmap.reference_price;
    
    // Copy grid data
    data.cells.resize(config_.time_buckets);
    for (size_t t = 0; t < config_.time_buckets; ++t) {
        data.cells[t].resize(config_.price_buckets);
        for (size_t p = 0; p < config_.price_buckets; ++p) {
            data.cells[t][p] = heatmap.grid[t][p];
        }
    }
    
    // Calculate statistics
    data.total_volume = heatmap.total_volume;
    data.hidden_volume_ratio = heatmap.total_volume > 0 ? 
        heatmap.hidden_volume / heatmap.total_volume : 0.0;
    data.anomaly_count = heatmap.total_anomalies;
    
    // Calculate VWAP
    double pv_sum = 0.0;
    double v_sum = 0.0;
    for (const auto& trade : heatmap.recent_trades) {
        pv_sum += trade.price * trade.quantity;
        v_sum += trade.quantity;
    }
    data.volume_weighted_price = v_sum > 0 ? static_cast<Price>(pv_sum / v_sum) : 0;
    
    return data;
}

std::vector<ExecutionHeatmap::HeatmapData> ExecutionHeatmap::get_top_symbols(size_t count) const {
    std::shared_lock lock(data_mutex_);
    
    // Calculate activity scores
    std::vector<std::pair<Symbol, double>> symbol_scores;
    for (const auto& [symbol, heatmap] : symbol_heatmaps_) {
        double score = calculate_activity_score(heatmap);
        symbol_scores.emplace_back(symbol, score);
    }
    
    // Sort by score
    std::sort(symbol_scores.begin(), symbol_scores.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Get top symbols
    std::vector<HeatmapData> result;
    size_t limit = std::min(count, symbol_scores.size());
    
    for (size_t i = 0; i < limit; ++i) {
        result.push_back(get_heatmap(symbol_scores[i].first));
    }
    
    return result;
}

ExecutionHeatmap::ActivityMetrics ExecutionHeatmap::get_activity_metrics() const {
    std::shared_lock lock(data_mutex_);
    
    ActivityMetrics metrics = global_metrics_;
    
    // Calculate rates
    Timestamp now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    double total_trades = 0;
    double total_volume = 0;
    double total_anomalies = 0;
    
    for (const auto& [symbol, heatmap] : symbol_heatmaps_) {
        total_trades += heatmap.total_trades;
        total_volume += heatmap.total_volume;
        total_anomalies += heatmap.total_anomalies;
    }
    
    // Simple rate calculation (would need time tracking for accurate rates)
    metrics.trades_per_second = total_trades / 3600.0; // Approximation
    metrics.volume_per_second = total_volume / 3600.0;
    metrics.anomaly_rate = total_anomalies / total_trades;
    
    return metrics;
}

ExecutionHeatmap::JsonHeatmap ExecutionHeatmap::get_json_heatmap(Symbol symbol) const {
    auto data = get_heatmap(symbol);
    JsonHeatmap json;
    
    json.symbol = std::to_string(symbol); // Would map to string in production
    
    // Convert intensity grid
    json.intensity.resize(config_.time_buckets);
    for (size_t t = 0; t < config_.time_buckets; ++t) {
        json.intensity[t].resize(config_.price_buckets);
        for (size_t p = 0; p < config_.price_buckets; ++p) {
            json.intensity[t][p] = data.cells[t][p].intensity * 255;
        }
    }
    
    // Create price labels
    for (size_t p = 0; p < config_.price_buckets; ++p) {
        double price = data.price_range_start + 
            p * (data.price_range_end - data.price_range_start) / config_.price_buckets;
        json.price_labels.push_back(price);
    }
    
    // Create time labels
    for (size_t t = 0; t < config_.time_buckets; ++t) {
        json.time_labels.push_back(std::to_string(t) + "s");
    }
    
    // Create metadata JSON
    std::stringstream meta;
    meta << "{";
    meta << "\"total_volume\":" << data.total_volume << ",";
    meta << "\"hidden_ratio\":" << std::fixed << std::setprecision(3) << data.hidden_volume_ratio << ",";
    meta << "\"vwap\":" << data.volume_weighted_price << ",";
    meta << "\"anomalies\":" << data.anomaly_count;
    meta << "}";
    json.metadata = meta.str();
    
    return json;
}

size_t ExecutionHeatmap::price_to_bucket(Price price, Price reference) const {
    if (reference == 0) return config_.price_buckets / 2;
    
    double relative_change = (price - reference) / static_cast<double>(reference);
    int bucket_offset = static_cast<int>(relative_change / config_.price_bucket_width);
    
    size_t bucket = config_.price_buckets / 2 + bucket_offset;
    return std::min(std::max(size_t(0), bucket), config_.price_buckets - 1);
}

size_t ExecutionHeatmap::time_to_bucket(Timestamp timestamp, Timestamp reference) const {
    if (reference == 0) return 0;
    
    int64_t elapsed_ns = timestamp - reference;
    int64_t elapsed_s = elapsed_ns / 1000000000;
    
    return elapsed_s % config_.time_buckets;
}

void ExecutionHeatmap::update_heatmap(SymbolHeatmap& heatmap, const Trade& trade) {
    // Initialize reference price
    if (heatmap.reference_price == 0) {
        heatmap.reference_price = trade.price;
        heatmap.last_update = trade.timestamp;
    }
    
    // Add to recent trades
    heatmap.recent_trades.push_back(trade);
    while (heatmap.recent_trades.size() > 1000) {
        heatmap.recent_trades.pop_front();
    }
    
    // Update statistics
    heatmap.total_volume += trade.quantity;
    heatmap.total_trades++;
    
    if (trade.is_hidden || config_.track_hidden_only) {
        heatmap.hidden_volume += trade.quantity;
    }
    
    // Decay old values if needed
    decay_heatmap(heatmap, trade.timestamp);
    
    // Update grid
    size_t time_bucket = time_to_bucket(trade.timestamp, heatmap.last_update);
    size_t price_bucket = price_to_bucket(trade.price, heatmap.reference_price);
    
    auto& cell = heatmap.grid[time_bucket][price_bucket];
    
    if (trade.aggressor_side == Side::BUY) {
        cell.buy_volume += trade.quantity;
    } else {
        cell.sell_volume += trade.quantity;
    }
    
    if (trade.is_hidden) {
        cell.hidden_volume += trade.quantity;
    }
    
    cell.trade_count++;
    
    // Update intensity
    double total_cell_volume = cell.buy_volume + cell.sell_volume;
    cell.intensity = std::min(1.0f, static_cast<float>(total_cell_volume / 100000.0));
    
    // Normalize intensities periodically
    if (heatmap.total_trades % 100 == 0) {
        normalize_intensities(heatmap);
    }
}

void ExecutionHeatmap::decay_heatmap(SymbolHeatmap& heatmap, Timestamp current_time) {
    // Check if we need to shift time buckets
    int64_t elapsed_s = (current_time - heatmap.last_update) / 1000000000;
    
    if (elapsed_s >= static_cast<int64_t>(config_.time_buckets)) {
        // Clear entire grid if too much time has passed
        heatmap.reset();
        heatmap.last_update = current_time;
        return;
    }
    
    if (elapsed_s > 0) {
        // Shift time buckets
        size_t shift = std::min(static_cast<size_t>(elapsed_s), config_.time_buckets);
        
        // Apply decay and shift
        for (size_t t = 0; t < config_.time_buckets - shift; ++t) {
            for (size_t p = 0; p < config_.price_buckets; ++p) {
                auto& old_cell = heatmap.grid[t][p];
                auto& new_cell = heatmap.grid[t + shift][p];
                
                // Decay values
                old_cell.buy_volume *= config_.volume_decay_rate;
                old_cell.sell_volume *= config_.volume_decay_rate;
                old_cell.hidden_volume *= config_.volume_decay_rate;
                old_cell.intensity *= config_.volume_decay_rate;
                
                // Shift
                new_cell = old_cell;
            }
        }
        
        // Clear old buckets
        for (size_t t = 0; t < shift; ++t) {
            for (size_t p = 0; p < config_.price_buckets; ++p) {
                heatmap.grid[t][p] = HeatmapCell{};
            }
        }
        
        heatmap.last_update = current_time;
    }
}

void ExecutionHeatmap::normalize_intensities(SymbolHeatmap& heatmap) const {
    // Find max intensity
    float max_intensity = 0.0f;
    
    for (const auto& row : heatmap.grid) {
        for (const auto& cell : row) {
            max_intensity = std::max(max_intensity, cell.intensity);
        }
    }
    
    if (max_intensity > 0) {
        // Normalize to [0, 1]
        float scale = 1.0f / max_intensity;
        
        for (auto& row : heatmap.grid) {
            for (auto& cell : row) {
                cell.intensity *= scale;
            }
        }
    }
}

double ExecutionHeatmap::calculate_activity_score(const SymbolHeatmap& heatmap) const {
    // Composite score based on volume, trades, and anomalies
    double volume_score = std::log1p(heatmap.total_volume);
    double trade_score = std::log1p(heatmap.total_trades);
    double anomaly_score = heatmap.total_anomalies * 10.0;
    double hidden_score = heatmap.hidden_volume / (heatmap.total_volume + 1.0) * 100.0;
    
    return volume_score + trade_score + anomaly_score + hidden_score;
}

} 
