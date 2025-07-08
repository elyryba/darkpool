#include "darkpool/core/slippage_tracker.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace darkpool {

SlippageTracker::SlippageTracker(const Config& config) : config_(config) {
    symbol_data_.reserve(1000);
}

void SlippageTracker::on_trade(const Trade& trade) {
    std::unique_lock lock(data_mutex_);
    auto& data = symbol_data_[trade.symbol];
    
    TradeRecord record;
    record.timestamp = trade.timestamp;
    record.execution_price = trade.price;
    record.quantity = trade.quantity;
    record.side = trade.aggressor_side;
    
    // Set mid quote if available
    if (data.current_quote.timestamp > 0) {
        record.mid_quote = (data.current_quote.bid + data.current_quote.ask) / 2;
    } else {
        record.mid_quote = trade.price;
    }
    
    // Set arrival price (look back in price history)
    record.arrival_price = record.mid_quote;
    if (!data.price_history.empty()) {
        // Find price ~100ms before trade
        Timestamp arrival_time = trade.timestamp - 100000000; // 100ms in ns
        for (auto it = data.price_history.rbegin(); it != data.price_history.rend(); ++it) {
            if (it->timestamp <= arrival_time) {
                record.arrival_price = (it->bid + it->ask) / 2;
                break;
            }
        }
    }
    
    data.trades.push_back(record);
    
    // Maintain window size
    while (data.trades.size() > config_.lookback_trades) {
        data.trades.pop_front();
    }
    
    // Update statistics
    double slippage = std::abs(static_cast<double>(record.execution_price - record.mid_quote)) 
                     / record.mid_quote;
    data.slippage_sum += slippage;
    data.slippage_sum_sq += slippage * slippage;
    data.slippage_samples++;
    data.max_slippage = std::max(data.max_slippage, slippage);
}

void SlippageTracker::on_quote(const Quote& quote) {
    std::unique_lock lock(data_mutex_);
    auto& data = symbol_data_[quote.symbol];
    
    data.current_quote = {quote.bid_price, quote.ask_price, quote.timestamp};
    
    // Store price history for arrival price calculation
    data.price_history.push_back(data.current_quote);
    
    // Keep only recent history (1 second)
    Timestamp cutoff = quote.timestamp - 1000000000;
    while (!data.price_history.empty() && data.price_history.front().timestamp < cutoff) {
        data.price_history.pop_front();
    }
}

void SlippageTracker::on_order_book(const OrderBookSnapshot& book) {
    if (book.bids[0].quantity > 0 && book.asks[0].quantity > 0) {
        Quote quote;
        quote.symbol = book.symbol;
        quote.bid_price = book.bids[0].price;
        quote.ask_price = book.asks[0].price;
        quote.bid_size = book.bids[0].quantity;
        quote.ask_size = book.asks[0].quantity;
        quote.timestamp = book.timestamp;
        quote.venue = book.venue;
        
        on_quote(quote);
    }
}

SlippageTracker::SlippageMetrics SlippageTracker::calculate_slippage(Symbol symbol) const {
    std::shared_lock lock(data_mutex_);
    
    auto it = symbol_data_.find(symbol);
    if (it == symbol_data_.end()) {
        return SlippageMetrics{};
    }
    
    const auto& data = it->second;
    if (data.trades.size() < config_.min_trades_for_analysis) {
        return SlippageMetrics{};
    }
    
    SlippageMetrics metrics;
    metrics.sample_size = data.trades.size();
    
    double total_immediate = 0.0;
    double total_realized = 0.0;
    double total_permanent = 0.0;
    double total_temporary = 0.0;
    double total_volume = 0.0;
    
    for (const auto& trade : data.trades) {
        double weight = trade.quantity;
        total_volume += weight;
        
        // Immediate slippage (execution vs quote)
        double immediate = (trade.side == Side::BUY) ?
            (trade.execution_price - trade.mid_quote) / trade.mid_quote :
            (trade.mid_quote - trade.execution_price) / trade.mid_quote;
        total_immediate += immediate * weight;
        
        // Realized slippage (execution vs arrival)
        double realized = (trade.side == Side::BUY) ?
            (trade.execution_price - trade.arrival_price) / trade.arrival_price :
            (trade.arrival_price - trade.execution_price) / trade.arrival_price;
        total_realized += realized * weight;
        
        // Impact decomposition
        auto impact = decompose_impact(trade, data);
        total_permanent += impact.permanent * weight;
        total_temporary += impact.temporary * weight;
    }
    
    if (total_volume > 0) {
        metrics.immediate_slippage = total_immediate / total_volume;
        metrics.realized_slippage = total_realized / total_volume;
        metrics.permanent_impact = total_permanent / total_volume;
        metrics.temporary_impact = total_temporary / total_volume;
        metrics.total_cost = metrics.realized_slippage;
    }
    
    return metrics;
}

std::optional<Anomaly> SlippageTracker::check_anomaly(Symbol symbol) const {
    auto metrics = calculate_slippage(symbol);
    auto stats = get_historical_stats(symbol);
    
    if (metrics.sample_size < config_.min_trades_for_analysis || stats.std_dev == 0) {
        return std::nullopt;
    }
    
    double z_score = (metrics.realized_slippage - stats.avg_slippage) / stats.std_dev;
    
    if (std::abs(z_score) > config_.abnormal_slippage_threshold) {
        Anomaly anomaly;
        anomaly.symbol = symbol;
        anomaly.timestamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        anomaly.type = AnomalyType::ABNORMAL_SLIPPAGE;
        anomaly.confidence = std::min(1.0, std::abs(z_score) / 5.0);
        anomaly.magnitude = metrics.realized_slippage / stats.avg_slippage;
        
        // Estimate hidden size based on permanent impact
        anomaly.estimated_hidden_size = static_cast<Quantity>(
            metrics.permanent_impact * 10000  // Scale factor
        );
        
        snprintf(anomaly.description.data(), anomaly.description.size(),
                "Abnormal slippage: %.4f%% (avg: %.4f%%, z-score: %.2f)",
                metrics.realized_slippage * 100, stats.avg_slippage * 100, z_score);
        
        return anomaly;
    }
    
    return std::nullopt;
}

SlippageTracker::HistoricalStats SlippageTracker::get_historical_stats(Symbol symbol) const {
    std::shared_lock lock(data_mutex_);
    
    auto it = symbol_data_.find(symbol);
    if (it == symbol_data_.end()) {
        return HistoricalStats{};
    }
    
    const auto& data = it->second;
    HistoricalStats stats;
    
    if (data.slippage_samples > 0) {
        stats.avg_slippage = data.slippage_sum / data.slippage_samples;
        double variance = (data.slippage_sum_sq / data.slippage_samples) - 
                         (stats.avg_slippage * stats.avg_slippage);
        stats.std_dev = std::sqrt(std::max(0.0, variance));
        stats.max_slippage = data.max_slippage;
        stats.total_trades = data.slippage_samples;
        
        // Calculate 95th percentile (simplified)
        stats.percentile_95 = stats.avg_slippage + 1.96 * stats.std_dev;
    }
    
    return stats;
}

void SlippageTracker::reset(Symbol symbol) {
    std::unique_lock lock(data_mutex_);
    symbol_data_.erase(symbol);
}

void SlippageTracker::reset_all() {
    std::unique_lock lock(data_mutex_);
    symbol_data_.clear();
}

SlippageTracker::ImpactComponents SlippageTracker::decompose_impact(
    const TradeRecord& trade, const SymbolData& data) const {
    
    ImpactComponents impact;
    
    // Find post-trade price (simplified)
    Price post_trade_price = trade.mid_quote;
    for (const auto& price : data.price_history) {
        if (price.timestamp > trade.timestamp + 10000000) { // 10ms after
            post_trade_price = (price.bid + price.ask) / 2;
            break;
        }
    }
    
    // Calculate components
    double execution_impact = (trade.side == Side::BUY) ?
        (trade.execution_price - trade.arrival_price) / trade.arrival_price :
        (trade.arrival_price - trade.execution_price) / trade.arrival_price;
    
    double post_trade_impact = (trade.side == Side::BUY) ?
        (post_trade_price - trade.arrival_price) / trade.arrival_price :
        (trade.arrival_price - post_trade_price) / trade.arrival_price;
    
    impact.immediate = execution_impact;
    impact.permanent = post_trade_impact * config_.impact_decay;
    impact.temporary = execution_impact - impact.permanent;
    
    return impact;
}

Price SlippageTracker::calculate_vwap(const std::deque<TradeRecord>& trades, 
                                     size_t lookback) const {
    if (trades.empty()) return 0;
    
    double sum_pq = 0.0;
    double sum_q = 0.0;
    
    size_t count = 0;
    for (auto it = trades.rbegin(); it != trades.rend() && count < lookback; ++it, ++count) {
        sum_pq += it->execution_price * it->quantity;
        sum_q += it->quantity;
    }
    
    return sum_q > 0 ? static_cast<Price>(sum_pq / sum_q) : 0;
}

bool SlippageTracker::is_outlier(double slippage, const SymbolData& data) const {
    if (data.slippage_samples < 10) return false;
    
    double avg = data.slippage_sum / data.slippage_samples;
    double std_dev = std::sqrt((data.slippage_sum_sq / data.slippage_samples) - avg * avg);
    
    return std::abs(slippage - avg) > config_.outlier_threshold * std_dev;
}

} 
