#include "darkpool/core/trade_to_quote_ratio.hpp"
#include <cmath>
#include <algorithm>

namespace darkpool {

TradeToQuoteRatio::TradeToQuoteRatio(const Config& config) : config_(config) {
    symbol_data_.reserve(1000);
}

double TradeToQuoteRatio::calculate_tqr(Symbol symbol) const {
    std::shared_lock lock(data_mutex_);
    
    auto it = symbol_data_.find(symbol);
    if (it == symbol_data_.end()) {
        return 0.0;
    }
    
    const auto& data = it->second;
    
    // Check cache
    auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    if (data.cache_timestamp > 0 && (now - data.cache_timestamp) < 1000000) { // 1ms cache
        return data.cached_tqr;
    }
    
    // Clean window first
    const_cast<TradeToQuoteRatio*>(this)->clean_window(const_cast<SymbolData&>(data), now);
    
    if (data.quotes.empty() || data.trades.size() < config_.min_trades) {
        data.cached_tqr = 0.0;
        data.cache_timestamp = now;
        return 0.0;
    }
    
    // Calculate weighted TQR
    double trade_volume = 0.0;
    double quote_volume = 0.0;
    
    for (const auto& trade : data.trades) {
        trade_volume += trade.quantity;
    }
    
    for (const auto& quote : data.quotes) {
        // Weight quotes by time decay
        double age_ms = (now - quote.timestamp) / 1000000.0;
        double decay = std::exp(-age_ms / config_.quote_decay_ms);
        quote_volume += (quote.bid_size + quote.ask_size) * decay;
    }
    
    double tqr = quote_volume > 0 ? trade_volume / quote_volume : 0.0;
    
    // Update cache and statistics
    data.cached_tqr = tqr;
    data.cache_timestamp = now;
    const_cast<TradeToQuoteRatio*>(this)->update_statistics(const_cast<SymbolData&>(data), tqr);
    
    return tqr;
}

std::optional<Anomaly> TradeToQuoteRatio::check_anomaly(Symbol symbol) const {
    auto stats = get_stats(symbol);
    
    if (stats.trade_count < config_.min_trades) {
        return std::nullopt;
    }
    
    double threshold = config_.threshold;
    if (config_.adaptive_threshold && stats.std_dev > 0) {
        threshold = stats.avg_tqr + config_.adaptive_factor * stats.std_dev;
    }
    
    if (stats.current_tqr > threshold) {
        Anomaly anomaly;
        anomaly.symbol = symbol;
        anomaly.timestamp = stats.last_update;
        anomaly.type = AnomalyType::HIGH_TQR;
        anomaly.confidence = std::min(1.0, (stats.current_tqr - threshold) / threshold);
        anomaly.magnitude = stats.current_tqr / stats.avg_tqr;
        
        // Estimate hidden size based on excess trade volume
        double expected_trade_volume = stats.quote_count * stats.avg_tqr;
        double excess_volume = stats.trade_count - expected_trade_volume;
        anomaly.estimated_hidden_size = static_cast<Quantity>(std::max(0.0, excess_volume));
        
        snprintf(anomaly.description.data(), anomaly.description.size(),
                "High TQR detected: %.2f (threshold: %.2f, avg: %.2f)",
                stats.current_tqr, threshold, stats.avg_tqr);
        
        return anomaly;
    }
    
    return std::nullopt;
}

TradeToQuoteRatio::SymbolStats TradeToQuoteRatio::get_stats(Symbol symbol) const {
    std::shared_lock lock(data_mutex_);
    
    auto it = symbol_data_.find(symbol);
    if (it == symbol_data_.end()) {
        return SymbolStats{};
    }
    
    const auto& data = it->second;
    
    SymbolStats stats;
    stats.trade_count = data.trades.size();
    stats.quote_count = data.quotes.size();
    stats.current_tqr = calculate_tqr(symbol);
    
    if (data.tqr_samples > 0) {
        stats.avg_tqr = data.tqr_sum / data.tqr_samples;
        double variance = (data.tqr_sum_sq / data.tqr_samples) - (stats.avg_tqr * stats.avg_tqr);
        stats.std_dev = std::sqrt(std::max(0.0, variance));
    }
    
    if (!data.trades.empty()) {
        stats.last_update = data.trades.back().timestamp;
    }
    
    return stats;
}

void TradeToQuoteRatio::reset(Symbol symbol) {
    std::unique_lock lock(data_mutex_);
    symbol_data_.erase(symbol);
}

void TradeToQuoteRatio::reset_all() {
    std::unique_lock lock(data_mutex_);
    symbol_data_.clear();
}

void TradeToQuoteRatio::clean_window(SymbolData& data, Timestamp current_time) const {
    Timestamp cutoff = current_time - (config_.window_size * 1000000); // Convert ms to ns
    
    // Remove old trades
    while (!data.trades.empty() && data.trades.front().timestamp < cutoff) {
        data.trades.pop_front();
    }
    
    // Remove old quotes
    while (!data.quotes.empty() && data.quotes.front().timestamp < cutoff) {
        data.quotes.pop_front();
    }
}

double TradeToQuoteRatio::calculate_adaptive_threshold(const SymbolData& data) const {
    if (data.tqr_samples < 10) {
        return config_.threshold;
    }
    
    double avg = data.tqr_sum / data.tqr_samples;
    double variance = (data.tqr_sum_sq / data.tqr_samples) - (avg * avg);
    double std_dev = std::sqrt(std::max(0.0, variance));
    
    return avg + config_.adaptive_factor * std_dev;
}

void TradeToQuoteRatio::update_statistics(SymbolData& data, double tqr) const {
    data.tqr_sum += tqr;
    data.tqr_sum_sq += tqr * tqr;
    data.tqr_samples++;
    
    // Limit history to prevent overflow
    if (data.tqr_samples > 10000) {
        data.tqr_sum *= 0.9;
        data.tqr_sum_sq *= 0.9;
        data.tqr_samples = static_cast<size_t>(data.tqr_samples * 0.9);
    }
}

} 
