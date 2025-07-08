#include "darkpool/core/vpin_calculator.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace darkpool {

VPINCalculator::VPINCalculator(const Config& config) : config_(config) {
    symbol_data_.reserve(1000);
}

void VPINCalculator::on_trade(const Trade& trade) {
    std::unique_lock lock(data_mutex_);
    auto& data = symbol_data_[trade.symbol];
    
    // Add trade to current bucket
    data.current_bucket.trades.push_back(trade);
    
    auto& bucket = data.current_bucket.bucket;
    if (bucket.trade_count == 0) {
        bucket.start_time = trade.timestamp;
        bucket.open_price = trade.price;
    }
    
    // Update bucket metrics
    bucket.total_volume += trade.quantity;
    bucket.vwap = ((bucket.vwap * (bucket.total_volume - trade.quantity)) + 
                   (trade.price * trade.quantity)) / bucket.total_volume;
    bucket.close_price = trade.price;
    bucket.end_time = trade.timestamp;
    bucket.trade_count++;
    
    // Simple trade classification based on aggressor side
    if (config_.use_bulk_classification) {
        // Store for bulk classification later
    } else {
        // Use Lee-Ready or tick rule
        Classification classification = Classification::UNKNOWN;
        
        if (trade.aggressor_side != Side::UNKNOWN) {
            classification = trade.aggressor_side == Side::BUY ? 
                           Classification::BUY : Classification::SELL;
        } else if (!data.last_quote.timestamp == 0) {
            classification = classify_trade_lee_ready(trade, data.last_quote);
        }
        
        if (classification == Classification::BUY) {
            bucket.buy_volume += trade.quantity;
        } else if (classification == Classification::SELL) {
            bucket.sell_volume += trade.quantity;
        } else {
            // Split evenly if unknown
            bucket.buy_volume += trade.quantity / 2.0;
            bucket.sell_volume += trade.quantity / 2.0;
        }
    }
    
    // Check if bucket is complete
    if (bucket.total_volume >= config_.volume_bucket_size) {
        complete_bucket(data);
    }
    
    // Time-based buckets as fallback
    double time_elapsed = (trade.timestamp - bucket.start_time) / 1000000000.0;
    if (time_elapsed > config_.time_bar_seconds && bucket.total_volume > 0) {
        complete_bucket(data);
    }
}

void VPINCalculator::on_quote(const Quote& quote) {
    std::unique_lock lock(data_mutex_);
    auto& data = symbol_data_[quote.symbol];
    data.last_quote = quote;
}

VPINCalculator::VPINResult VPINCalculator::calculate_vpin(Symbol symbol) const {
    std::shared_lock lock(data_mutex_);
    
    auto it = symbol_data_.find(symbol);
    if (it == symbol_data_.end()) {
        return VPINResult{0.0, 0.0, 0.0, 0, 0.0, 0};
    }
    
    const auto& data = it->second;
    VPINResult result;
    result.calculation_time = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    result.completed_buckets = data.completed_buckets.size();
    
    if (data.completed_buckets.size() < config_.min_buckets_for_vpin) {
        return result;
    }
    
    // Calculate VPIN over the specified number of buckets
    size_t n = std::min(config_.num_buckets, data.completed_buckets.size());
    double vpin_sum = 0.0;
    
    auto start_it = data.completed_buckets.end() - n;
    for (auto it = start_it; it != data.completed_buckets.end(); ++it) {
        vpin_sum += it->get_order_imbalance();
    }
    
    result.vpin = vpin_sum / n;
    
    // Calculate average and standard deviation
    if (data.vpin_samples > 0) {
        result.avg_vpin = data.vpin_sum / data.vpin_samples;
        double variance = (data.vpin_sum_sq / data.vpin_samples) - 
                         (result.avg_vpin * result.avg_vpin);
        result.vpin_std_dev = std::sqrt(std::max(0.0, variance));
    }
    
    // Current bucket imbalance
    if (data.current_bucket.bucket.total_volume > 0) {
        result.order_imbalance = data.current_bucket.bucket.get_order_imbalance();
    }
    
    return result;
}

std::optional<Anomaly> VPINCalculator::check_anomaly(Symbol symbol) const {
    auto vpin_result = calculate_vpin(symbol);
    
    if (vpin_result.completed_buckets < config_.min_buckets_for_vpin) {
        return std::nullopt;
    }
    
    // Check for toxic flow
    if (vpin_result.vpin > config_.toxicity_threshold) {
        double z_score = vpin_result.toxicity_zscore();
        
        if (z_score > 2.0) {  // 2 standard deviations
            Anomaly anomaly;
            anomaly.symbol = symbol;
            anomaly.timestamp = vpin_result.calculation_time;
            anomaly.type = AnomalyType::POST_TRADE_DRIFT;  // Using as proxy for toxicity
            anomaly.confidence = std::min(1.0, z_score / 4.0);
            anomaly.magnitude = vpin_result.vpin / config_.toxicity_threshold;
            
            // Estimate hidden size based on order imbalance
            std::shared_lock lock(data_mutex_);
            auto it = symbol_data_.find(symbol);
            if (it != symbol_data_.end()) {
                Quantity avg_bucket_volume = config_.volume_bucket_size;
                anomaly.estimated_hidden_size = static_cast<Quantity>(
                    vpin_result.order_imbalance * avg_bucket_volume * 10
                );
            }
            
            snprintf(anomaly.description.data(), anomaly.description.size(),
                    "Toxic flow detected: VPIN=%.3f (z-score=%.2f), threshold=%.3f",
                    vpin_result.vpin, z_score, config_.toxicity_threshold);
            
            return anomaly;
        }
    }
    
    return std::nullopt;
}

std::vector<VPINCalculator::VPINPoint> VPINCalculator::get_vpin_history(
    Symbol symbol, size_t num_points) const {
    
    std::shared_lock lock(data_mutex_);
    
    auto it = symbol_data_.find(symbol);
    if (it == symbol_data_.end()) {
        return {};
    }
    
    const auto& buckets = it->second.completed_buckets;
    std::vector<VPINPoint> history;
    
    if (buckets.size() < config_.num_buckets) {
        return history;
    }
    
    size_t step = std::max(size_t(1), buckets.size() / num_points);
    
    for (size_t i = config_.num_buckets - 1; i < buckets.size(); i += step) {
        // Calculate VPIN at this point
        double vpin_sum = 0.0;
        size_t start = i >= config_.num_buckets ? i - config_.num_buckets + 1 : 0;
        
        for (size_t j = start; j <= i; ++j) {
            vpin_sum += buckets[j].get_order_imbalance();
        }
        
        double vpin = vpin_sum / (i - start + 1);
        
        VPINPoint point;
        point.timestamp = buckets[i].end_time;
        point.vpin = vpin;
        point.bucket_volume = buckets[i].total_volume;
        point.buy_volume_ratio = buckets[i].total_volume > 0 ?
            buckets[i].buy_volume / buckets[i].total_volume : 0.5;
        
        history.push_back(point);
    }
    
    return history;
}

VPINCalculator::BucketProgress VPINCalculator::get_current_bucket(Symbol symbol) const {
    std::shared_lock lock(data_mutex_);
    
    auto it = symbol_data_.find(symbol);
    if (it == symbol_data_.end()) {
        return BucketProgress{0, config_.volume_bucket_size, 0, 0, 0, 0};
    }
    
    const auto& bucket = it->second.current_bucket.bucket;
    
    BucketProgress progress;
    progress.current_volume = bucket.total_volume;
    progress.target_volume = config_.volume_bucket_size;
    progress.buy_volume = bucket.buy_volume;
    progress.sell_volume = bucket.sell_volume;
    progress.volume_weighted_price = bucket.vwap;
    progress.trade_count = bucket.trade_count;
    
    return progress;
}

VPINCalculator::Classification VPINCalculator::classify_trade_lee_ready(
    const Trade& trade, const Quote& quote) const {
    
    if (quote.bid_price == 0 || quote.ask_price == 0) {
        return Classification::UNKNOWN;
    }
    
    Price mid = (quote.bid_price + quote.ask_price) / 2;
    
    // Lee-Ready algorithm
    if (trade.price > mid) {
        return Classification::BUY;
    } else if (trade.price < mid) {
        return Classification::SELL;
    } else {
        // At midpoint - use tick rule
        return Classification::UNKNOWN;  // Would need previous trade
    }
}

void VPINCalculator::classify_bulk_volume(CurrentBucket& bucket, 
                                         const std::vector<Price>& price_levels) const {
    if (bucket.trades.empty() || price_levels.empty()) {
        return;
    }
    
    // Find support and resistance levels
    std::vector<double> volume_at_level(price_levels.size(), 0.0);
    
    for (const auto& trade : bucket.trades) {
        // Find nearest price level
        auto it = std::lower_bound(price_levels.begin(), price_levels.end(), trade.price);
        if (it != price_levels.end()) {
            size_t idx = std::distance(price_levels.begin(), it);
            volume_at_level[idx] += trade.quantity;
        }
    }
    
    // Classify based on volume distribution
    double total_volume = std::accumulate(volume_at_level.begin(), 
                                         volume_at_level.end(), 0.0);
    
    if (total_volume > 0) {
        // Calculate CDF
        double cumulative = 0.0;
        size_t median_level = 0;
        
        for (size_t i = 0; i < volume_at_level.size(); ++i) {
            cumulative += volume_at_level[i];
            if (cumulative >= total_volume * 0.5) {
                median_level = i;
                break;
            }
        }
        
        // Classify trades based on position relative to median
        bucket.bucket.buy_volume = 0;
        bucket.bucket.sell_volume = 0;
        
        for (const auto& trade : bucket.trades) {
            auto it = std::lower_bound(price_levels.begin(), 
                                      price_levels.end(), trade.price);
            if (it != price_levels.end()) {
                size_t idx = std::distance(price_levels.begin(), it);
                if (idx > median_level) {
                    bucket.bucket.buy_volume += trade.quantity;
                } else {
                    bucket.bucket.sell_volume += trade.quantity;
                }
            }
        }
    }
}

VPINCalculator::Classification VPINCalculator::classify_trade_tick_rule(
    const Trade& current, const Trade& previous) const {
    
    if (current.price > previous.price) {
        return Classification::BUY;
    } else if (current.price < previous.price) {
        return Classification::SELL;
    }
    
    return Classification::UNKNOWN;
}

void VPINCalculator::update_price_levels(SymbolData& data, 
                                        const std::deque<Trade>& recent_trades) const {
    if (recent_trades.empty()) {
        return;
    }
    
    // Update every minute
    Timestamp current_time = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    if (current_time - data.price_levels_update < 60000000000) { // 60 seconds
        return;
    }
    
    data.price_levels = calculate_support_levels(recent_trades);
    data.price_levels_update = current_time;
}

void VPINCalculator::complete_bucket(SymbolData& data) {
    auto& current = data.current_bucket;
    
    // Bulk volume classification if enabled
    if (config_.use_bulk_classification && !current.trades.empty()) {
        // Get recent trades for price level calculation
        std::deque<Trade> recent_trades;
        for (const auto& bucket : data.completed_buckets) {
            // Approximate - would need to store trades
        }
        for (const auto& trade : current.trades) {
            recent_trades.push_back(trade);
        }
        
        update_price_levels(data, recent_trades);
        classify_bulk_volume(current, data.price_levels);
    }
    
    // Store completed bucket
    data.completed_buckets.push_back(current.bucket);
    
    // Maintain window
    while (data.completed_buckets.size() > config_.num_buckets * 2) {
        data.completed_buckets.pop_front();
    }
    
    // Update VPIN statistics
    double vpin = current.bucket.get_order_imbalance();
    data.vpin_sum += vpin;
    data.vpin_sum_sq += vpin * vpin;
    data.vpin_samples++;
    
    // Reset current bucket
    current = CurrentBucket{};
}

std::vector<Price> VPINCalculator::calculate_support_levels(
    const std::deque<Trade>& trades) const {
    
    if (trades.empty()) {
        return {};
    }
    
    // Find price range
    auto [min_it, max_it] = std::minmax_element(trades.begin(), trades.end(),
        [](const Trade& a, const Trade& b) { return a.price < b.price; });
    
    Price min_price = min_it->price;
    Price max_price = max_it->price;
    
    // Create price levels with support window
    std::vector<Price> levels;
    Price step = static_cast<Price>((max_price - min_price) * config_.support_window);
    
    if (step == 0) {
        levels.push_back(min_price);
        return levels;
    }
    
    for (Price p = min_price; p <= max_price; p += step) {
        levels.push_back(p);
    }
    
    return levels;
}

} 
