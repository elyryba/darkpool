#include "darkpool/core/post_trade_drift.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace darkpool {

PostTradeDrift::PostTradeDrift(const Config& config) : config_(config) {
    symbol_data_.reserve(1000);
}

void PostTradeDrift::on_trade(const Trade& trade) {
    std::unique_lock lock(data_mutex_);
    auto& data = symbol_data_[trade.symbol];
    
    // Update price history
    PricePoint point;
    point.timestamp = trade.timestamp;
    point.last_trade = trade.price;
    point.mid_price = data.last_quote.timestamp > 0 ? 
        (data.last_quote.bid_price + data.last_quote.ask_price) / 2 : trade.price;
    
    data.price_history.push_back(point);
    
    // Store recent trades for volume analysis
    data.recent_trades.push_back(trade);
    
    // Check if this is a large trade to track
    if (config_.track_all_trades || is_large_trade(trade, data)) {
        TrackedTrade tracked;
        tracked.id = trade.order_id;
        tracked.symbol = trade.symbol;
        tracked.timestamp = trade.timestamp;
        tracked.execution_price = trade.price;
        tracked.size = trade.quantity;
        tracked.side = trade.aggressor_side;
        tracked.pre_trade_mid = calculate_pre_trade_price(trade, data);
        
        // Initialize post-trade price tracking
        for (size_t i = 0; i < config_.num_windows; ++i) {
            tracked.window_complete[i] = false;
            tracked.post_trade_prices[i] = trade.price;
        }
        
        data.tracked_trades.push_back(tracked);
        
        // Maintain memory limit
        while (data.tracked_trades.size() > config_.max_tracked_trades) {
            data.tracked_trades.pop_front();
        }
    }
    
    // Update post-trade prices for existing tracked trades
    update_post_trade_prices(data, trade.timestamp);
    
    // Clean old data
    clean_old_data(data, trade.timestamp);
}

void PostTradeDrift::on_quote(const Quote& quote) {
    std::unique_lock lock(data_mutex_);
    auto& data = symbol_data_[quote.symbol];
    
    data.last_quote = quote;
    
    // Add to price history
    PricePoint point;
    point.timestamp = quote.timestamp;
    point.mid_price = (quote.bid_price + quote.ask_price) / 2;
    point.last_trade = data.price_history.empty() ? point.mid_price : 
                       data.price_history.back().last_trade;
    
    data.price_history.push_back(point);
    
    // Update post-trade prices
    update_post_trade_prices(data, quote.timestamp);
}

PostTradeDrift::DriftResult PostTradeDrift::analyze_drift(Symbol symbol, OrderId trade_id) const {
    std::shared_lock lock(data_mutex_);
    
    auto it = symbol_data_.find(symbol);
    if (it == symbol_data_.end()) {
        return DriftResult{};
    }
    
    const auto& data = it->second;
    
    // Find tracked trade
    auto trade_it = std::find_if(data.tracked_trades.begin(), data.tracked_trades.end(),
        [trade_id](const TrackedTrade& t) { return t.id == trade_id; });
    
    if (trade_it == data.tracked_trades.end()) {
        return DriftResult{};
    }
    
    return calculate_drift(*trade_it, data);
}

PostTradeDrift::DriftStats PostTradeDrift::get_drift_stats(Symbol symbol) const {
    std::shared_lock lock(data_mutex_);
    
    DriftStats stats{};
    
    auto it = symbol_data_.find(symbol);
    if (it == symbol_data_.end()) {
        return stats;
    }
    
    const auto& data = it->second;
    
    if (data.drift_samples > 0) {
        stats.avg_drift_60s = data.drift_sum / data.drift_samples;
        double variance = (data.drift_sum_sq / data.drift_samples) - 
                         (stats.avg_drift_60s * stats.avg_drift_60s);
        stats.drift_std_dev = std::sqrt(std::max(0.0, variance));
        
        stats.information_ratio = stats.drift_std_dev > 0 ? 
            stats.avg_drift_60s / stats.drift_std_dev : 0.0;
    }
    
    // Count large trades and calculate stats
    double total_size = 0.0;
    int permanent_count = 0;
    
    for (const auto& trade : data.tracked_trades) {
        stats.large_trades_count++;
        total_size += trade.size;
        
        if (trade.drift_calculated) {
            if (trade.cached_drift.is_permanent) {
                permanent_count++;
            }
        }
    }
    
    if (stats.large_trades_count > 0) {
        stats.avg_large_trade_size = total_size / stats.large_trades_count;
        stats.permanent_impact_ratio = static_cast<double>(permanent_count) / 
                                      stats.large_trades_count;
    }
    
    return stats;
}

std::optional<Anomaly> PostTradeDrift::check_anomaly(Symbol symbol) const {
    auto stats = get_drift_stats(symbol);
    
    if (stats.large_trades_count < 5) {
        return std::nullopt;
    }
    
    // Check for systematic drift (information leakage)
    if (std::abs(stats.information_ratio) > 2.0 && 
        std::abs(stats.avg_drift_60s) > config_.significant_drift) {
        
        Anomaly anomaly;
        anomaly.symbol = symbol;
        anomaly.timestamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        anomaly.type = AnomalyType::POST_TRADE_DRIFT;
        anomaly.confidence = std::min(1.0, std::abs(stats.information_ratio) / 3.0);
        anomaly.magnitude = std::abs(stats.avg_drift_60s) / config_.significant_drift;
        
        // Estimate hidden size based on drift pattern
        anomaly.estimated_hidden_size = static_cast<Quantity>(
            stats.avg_large_trade_size * stats.permanent_impact_ratio * 10
        );
        
        snprintf(anomaly.description.data(), anomaly.description.size(),
                "Information leakage detected: avg_drift=%.3f%%, IR=%.2f, permanent=%.1f%%",
                stats.avg_drift_60s * 100, stats.information_ratio,
                stats.permanent_impact_ratio * 100);
        
        return anomaly;
    }
    
    // Check recent large trades for unusual drift
    auto large_trades = get_large_trades(symbol, 5);
    
    for (const auto& trade_info : large_trades) {
        if (trade_info.z_score > 3.0 && trade_info.drift.is_permanent) {
            Anomaly anomaly;
            anomaly.symbol = symbol;
            anomaly.timestamp = trade_info.timestamp;
            anomaly.type = AnomalyType::POST_TRADE_DRIFT;
            anomaly.confidence = std::min(1.0, trade_info.z_score / 4.0);
            anomaly.magnitude = std::abs(trade_info.drift.max_drift) / config_.significant_drift;
            anomaly.estimated_hidden_size = trade_info.size * 5; // Estimate follow-on size
            
            const char* direction = trade_info.side == Side::BUY ? "buy" : "sell";
            snprintf(anomaly.description.data(), anomaly.description.size(),
                    "Large %s trade with unusual drift: size=%llu, drift=%.3f%% (z=%.2f)",
                    direction, trade_info.size, trade_info.drift.max_drift * 100,
                    trade_info.z_score);
            
            return anomaly;
        }
    }
    
    return std::nullopt;
}

std::vector<PostTradeDrift::LargeTradeInfo> PostTradeDrift::get_large_trades(
    Symbol symbol, size_t count) const {
    
    std::shared_lock lock(data_mutex_);
    
    std::vector<LargeTradeInfo> result;
    
    auto it = symbol_data_.find(symbol);
    if (it == symbol_data_.end()) {
        return result;
    }
    
    const auto& data = it->second;
    
    // Get recent tracked trades
    size_t start = data.tracked_trades.size() > count ? 
                   data.tracked_trades.size() - count : 0;
    
    for (size_t i = start; i < data.tracked_trades.size(); ++i) {
        const auto& trade = data.tracked_trades[i];
        
        LargeTradeInfo info;
        info.trade_id = trade.id;
        info.timestamp = trade.timestamp;
        info.execution_price = trade.execution_price;
        info.size = trade.size;
        info.side = trade.side;
        info.drift = calculate_drift(trade, data);
        
        // Calculate z-score
        if (data.drift_samples > 0 && data.drift_sum_sq > 0) {
            double avg = data.drift_sum / data.drift_samples;
            double std_dev = std::sqrt(data.drift_sum_sq / data.drift_samples - avg * avg);
            info.z_score = std_dev > 0 ? 
                          (info.drift.drift_60s - avg) / std_dev : 0.0;
        } else {
            info.z_score = 0.0;
        }
        
        result.push_back(info);
    }
    
    return result;
}

bool PostTradeDrift::is_large_trade(const Trade& trade, const SymbolData& data) const {
    // Fixed threshold
    if (trade.quantity >= config_.large_trade_threshold) {
        return true;
    }
    
    // Dynamic threshold based on percentile
    update_volume_threshold(const_cast<SymbolData&>(data));
    
    return data.volume_threshold > 0 && trade.quantity >= data.volume_threshold;
}

void PostTradeDrift::update_volume_threshold(SymbolData& data) const {
    Timestamp now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    
    // Update every minute
    if (now - data.threshold_update < 60000000000) {
        return;
    }
    
    if (data.recent_trades.size() < 100) {
        return;
    }
    
    // Calculate volume percentile
    std::vector<Quantity> volumes;
    for (const auto& trade : data.recent_trades) {
        volumes.push_back(trade.quantity);
    }
    
    std::sort(volumes.begin(), volumes.end());
    size_t idx = static_cast<size_t>(volumes.size() * config_.volume_percentile);
    data.volume_threshold = volumes[idx];
    data.threshold_update = now;
}

Price PostTradeDrift::calculate_pre_trade_price(const Trade& trade, 
                                               const SymbolData& data) const {
    // Find the most recent price before this trade
    for (auto it = data.price_history.rbegin(); it != data.price_history.rend(); ++it) {
        if (it->timestamp < trade.timestamp) {
            return it->mid_price;
        }
    }
    
    // Fallback to trade price
    return trade.price;
}

void PostTradeDrift::update_post_trade_prices(SymbolData& data, Timestamp current_time) {
    for (auto& trade : data.tracked_trades) {
        for (size_t i = 0; i < config_.num_windows; ++i) {
            if (!trade.window_complete[i]) {
                Timestamp window_end = trade.timestamp + 
                    config_.drift_windows[i] * 1000000000; // Convert to nanoseconds
                
                if (current_time >= window_end) {
                    // Find price at window end
                    for (auto it = data.price_history.rbegin(); 
                         it != data.price_history.rend(); ++it) {
                        if (it->timestamp >= window_end) {
                            trade.post_trade_prices[i] = it->mid_price;
                            trade.window_complete[i] = true;
                            break;
                        }
                    }
                }
            }
        }
    }
}

PostTradeDrift::DriftResult PostTradeDrift::calculate_drift(
    const TrackedTrade& trade, const SymbolData& data) const {
    
    if (trade.drift_calculated) {
        return trade.cached_drift;
    }
    
    DriftResult result{};
    
    // Calculate drift for each window
    double drifts[5] = {0};
    for (size_t i = 0; i < config_.num_windows; ++i) {
        if (trade.window_complete[i]) {
            double price_change = trade.side == Side::BUY ?
                (trade.post_trade_prices[i] - trade.execution_price) :
                (trade.execution_price - trade.post_trade_prices[i]);
            
            drifts[i] = price_change / static_cast<double>(trade.execution_price);
        }
    }
    
    result.drift_10s = drifts[0];
    result.drift_30s = drifts[1];
    result.drift_60s = drifts[2];
    result.drift_300s = drifts[3];
    result.drift_600s = drifts[4];
    
    // Find maximum drift
    result.max_drift = *std::max_element(drifts, drifts + config_.num_windows);
    
    // Find when max drift occurred
    for (size_t i = 0; i < config_.num_windows; ++i) {
        if (std::abs(drifts[i] - result.max_drift) < 1e-10) {
            result.max_drift_time = trade.timestamp + 
                config_.drift_windows[i] * 1000000000;
            break;
        }
    }
    
    // Calculate reversion
    if (trade.window_complete[2] && trade.window_complete[4]) {
        result.reversion_ratio = result.drift_60s != 0 ?
            (result.drift_60s - result.drift_600s) / result.drift_60s : 0.0;
    }
    
    // Determine if impact is permanent
    result.is_permanent = trade.window_complete[4] && 
                         std::abs(result.drift_600s) > config_.significant_drift * 0.5;
    
    // Update statistics
    if (trade.window_complete[2] && !trade.drift_calculated) {
        const_cast<SymbolData&>(data).drift_sum += result.drift_60s;
        const_cast<SymbolData&>(data).drift_sum_sq += result.drift_60s * result.drift_60s;
        const_cast<SymbolData&>(data).drift_samples++;
    }
    
    // Cache result
    const_cast<TrackedTrade&>(trade).cached_drift = result;
    const_cast<TrackedTrade&>(trade).drift_calculated = true;
    
    return result;
}

bool PostTradeDrift::is_information_leakage(const DriftResult& drift,
                                           const SymbolData& data) const {
    // Check for significant and persistent drift
    if (std::abs(drift.drift_60s) < config_.significant_drift) {
        return false;
    }
    
    // Check if drift is unusual compared to historical
    if (data.drift_samples > 20) {
        double avg = data.drift_sum / data.drift_samples;
        double std_dev = std::sqrt(data.drift_sum_sq / data.drift_samples - avg * avg);
        
        double z_score = std_dev > 0 ? (drift.drift_60s - avg) / std_dev : 0.0;
        return std::abs(z_score) > 2.5;
    }
    
    return drift.is_permanent;
}

void PostTradeDrift::clean_old_data(SymbolData& data, Timestamp current_time) {
    // Remove old price history (keep 20 minutes)
    Timestamp cutoff = current_time - 1200000000000; // 20 minutes in nanoseconds
    
    while (!data.price_history.empty() && data.price_history.front().timestamp < cutoff) {
        data.price_history.pop_front();
    }
    
    // Remove old trades
    while (!data.recent_trades.empty() && data.recent_trades.front().timestamp < cutoff) {
        data.recent_trades.pop_front();
    }
    
    // Remove completed tracked trades older than 20 minutes
    data.tracked_trades.erase(
        std::remove_if(data.tracked_trades.begin(), data.tracked_trades.end(),
            [cutoff](const TrackedTrade& t) {
                return t.window_complete[4] && t.timestamp < cutoff;
            }),
        data.tracked_trades.end()
    );
}

} 
