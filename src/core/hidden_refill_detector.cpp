#include "darkpool/core/hidden_refill_detector.hpp"
#include <cmath>
#include <algorithm>

namespace darkpool {

HiddenRefillDetector::HiddenRefillDetector(const Config& config) : config_(config) {
    order_tracking_.reserve(config.max_tracking_orders);
}

void HiddenRefillDetector::on_order(const Order& order) {
    std::unique_lock lock(data_mutex_);
    
    // Track order
    OrderTracking& tracking = order_tracking_[order.id];
    tracking.symbol = order.symbol;
    tracking.price = order.price;
    tracking.side = order.side;
    tracking.original_quantity = order.quantity;
    tracking.remaining_quantity = order.remaining_quantity;
    tracking.first_seen = order.timestamp;
    
    // Update price level
    auto& symbol_data = symbol_data_[order.symbol];
    auto& levels = (order.side == Side::BUY) ? symbol_data.bid_levels : symbol_data.ask_levels;
    auto& level = levels[order.price];
    
    update_price_level(level, order);
    
    // Check for refill pattern
    if (detect_refill_pattern(level)) {
        // Create or update iceberg pattern
        IcebergPattern pattern;
        pattern.symbol = order.symbol;
        pattern.price_level = order.price;
        pattern.side = order.side;
        pattern.visible_size = order.quantity;
        pattern.total_executed = level.total_executed;
        pattern.refill_count = level.refill_times.size();
        pattern.first_seen = level.refill_times.front();
        pattern.last_refill = order.timestamp;
        
        // Calculate confidence based on consistency
        double size_variance = 0.0;
        if (level.refill_sizes.size() > 1) {
            double avg_size = 0.0;
            for (auto size : level.refill_sizes) {
                avg_size += size;
            }
            avg_size /= level.refill_sizes.size();
            
            for (auto size : level.refill_sizes) {
                size_variance += std::pow(size - avg_size, 2);
            }
            size_variance = std::sqrt(size_variance / level.refill_sizes.size());
        }
        
        pattern.confidence = 1.0 - std::min(1.0, size_variance / pattern.visible_size);
        pattern.estimated_total = pattern.total_executed + 
                                 pattern.visible_size * config_.hidden_multiplier;
        
        // Add to active icebergs
        bool found = false;
        for (auto& existing : symbol_data.active_icebergs) {
            if (existing.price_level == pattern.price_level && 
                existing.side == pattern.side) {
                existing = pattern;
                found = true;
                break;
            }
        }
        
        if (!found) {
            symbol_data.active_icebergs.push_back(pattern);
            stats_.icebergs_detected++;
        }
    }
    
    // Clean old data
    clean_old_data(symbol_data, order.timestamp);
}

void HiddenRefillDetector::on_trade(const Trade& trade) {
    std::unique_lock lock(data_mutex_);
    
    auto it = order_tracking_.find(trade.order_id);
    if (it == order_tracking_.end()) {
        return;
    }
    
    auto& tracking = it->second;
    tracking.execution_times.push_back(trade.timestamp);
    tracking.execution_sizes.push_back(trade.quantity);
    tracking.remaining_quantity -= trade.quantity;
    
    // Update price level execution
    auto& symbol_data = symbol_data_[trade.symbol];
    auto& levels = (trade.aggressor_side == Side::BUY) ? 
                   symbol_data.ask_levels : symbol_data.bid_levels;
    
    auto level_it = levels.find(trade.price);
    if (level_it != levels.end()) {
        level_it->second.total_executed += trade.quantity;
        level_it->second.last_activity = trade.timestamp;
    }
    
    // Check if this completes an iceberg slice
    if (tracking.remaining_quantity == 0 && 
        analyze_execution_pattern(tracking)) {
        tracking.potential_iceberg = true;
    }
}

void HiddenRefillDetector::on_order_cancel(OrderId order_id, Quantity canceled_quantity) {
    std::unique_lock lock(data_mutex_);
    
    auto it = order_tracking_.find(order_id);
    if (it != order_tracking_.end()) {
        it->second.remaining_quantity -= canceled_quantity;
        
        // Remove from tracking if fully canceled
        if (it->second.remaining_quantity == 0) {
            order_tracking_.erase(it);
        }
    }
}

void HiddenRefillDetector::on_order_replace(OrderId order_id, Price new_price, 
                                           Quantity new_quantity) {
    std::unique_lock lock(data_mutex_);
    
    auto it = order_tracking_.find(order_id);
    if (it != order_tracking_.end()) {
        it->second.price = new_price;
        it->second.remaining_quantity = new_quantity;
    }
}

std::vector<HiddenRefillDetector::IcebergPattern> 
HiddenRefillDetector::get_active_icebergs(Symbol symbol) const {
    std::shared_lock lock(data_mutex_);
    
    auto it = symbol_data_.find(symbol);
    if (it == symbol_data_.end()) {
        return {};
    }
    
    return it->second.active_icebergs;
}

std::optional<Anomaly> HiddenRefillDetector::check_anomaly(Symbol symbol) const {
    auto icebergs = get_active_icebergs(symbol);
    
    if (icebergs.empty()) {
        return std::nullopt;
    }
    
    // Find most significant iceberg
    const IcebergPattern* best = nullptr;
    double best_score = 0.0;
    
    for (const auto& iceberg : icebergs) {
        double score = iceberg.confidence * iceberg.estimated_total;
        if (score > best_score) {
            best_score = score;
            best = &iceberg;
        }
    }
    
    if (best && best->confidence > 0.7) {
        Anomaly anomaly;
        anomaly.symbol = symbol;
        anomaly.timestamp = best->last_refill;
        anomaly.type = AnomalyType::HIDDEN_REFILL;
        anomaly.confidence = best->confidence;
        anomaly.magnitude = static_cast<double>(best->estimated_total) / best->visible_size;
        anomaly.estimated_hidden_size = best->estimated_total - best->total_executed;
        anomaly.expected_impact = 0; // Would calculate based on order book
        
        snprintf(anomaly.description.data(), anomaly.description.size(),
                "Iceberg order detected at %.4f: %zu refills, est. total %llu",
                from_price(best->price_level), best->refill_count, best->estimated_total);
        
        return anomaly;
    }
    
    return std::nullopt;
}

bool HiddenRefillDetector::detect_refill_pattern(const PriceLevel& level) const {
    if (level.refill_times.size() < config_.min_refills) {
        return false;
    }
    
    // Check time consistency
    Timestamp window_start = level.refill_times.back() - 
                           config_.refill_window_ms * 1000000;
    
    size_t recent_refills = 0;
    for (auto time : level.refill_times) {
        if (time >= window_start) {
            recent_refills++;
        }
    }
    
    if (recent_refills < config_.min_refills) {
        return false;
    }
    
    // Check size consistency
    if (level.refill_sizes.size() >= config_.min_refills) {
        double avg_size = 0.0;
        for (size_t i = level.refill_sizes.size() - config_.min_refills; 
             i < level.refill_sizes.size(); ++i) {
            avg_size += level.refill_sizes[i];
        }
        avg_size /= config_.min_refills;
        
        // Check variance
        for (size_t i = level.refill_sizes.size() - config_.min_refills; 
             i < level.refill_sizes.size(); ++i) {
            double deviation = std::abs(level.refill_sizes[i] - avg_size) / avg_size;
            if (deviation > config_.size_consistency_threshold) {
                return false;
            }
        }
    }
    
    return true;
}

bool HiddenRefillDetector::analyze_execution_pattern(const OrderTracking& tracking) const {
    if (tracking.execution_times.size() < 2) {
        return false;
    }
    
    // Check for rapid execution (characteristic of hitting hidden liquidity)
    Timestamp duration = tracking.execution_times.back() - tracking.execution_times.front();
    double exec_rate = tracking.execution_sizes.size() / (duration / 1000000000.0);
    
    // High execution rate suggests algorithmic refilling
    return exec_rate > 1.0; // More than 1 execution per second
}

void HiddenRefillDetector::update_price_level(PriceLevel& level, const Order& order) {
    level.order_queue.push_back(order.id);
    level.total_visible += order.quantity;
    level.refill_times.push_back(order.timestamp);
    level.refill_sizes.push_back(order.quantity);
    level.last_activity = order.timestamp;
    
    // Limit memory usage
    while (level.order_queue.size() > 100) {
        level.order_queue.pop_front();
    }
    while (level.refill_times.size() > 100) {
        level.refill_times.erase(level.refill_times.begin());
        level.refill_sizes.erase(level.refill_sizes.begin());
    }
}

void HiddenRefillDetector::clean_old_data(SymbolData& data, Timestamp current_time) {
    Timestamp cutoff = current_time - config_.refill_window_ms * 1000000;
    
    // Clean price levels
    auto clean_levels = [&](auto& levels) {
        for (auto it = levels.begin(); it != levels.end(); ) {
            if (it->second.last_activity < cutoff) {
                it = levels.erase(it);
            } else {
                ++it;
            }
        }
    };
    
    clean_levels(data.bid_levels);
    clean_levels(data.ask_levels);
    
    // Clean inactive icebergs
    data.active_icebergs.erase(
        std::remove_if(data.active_icebergs.begin(), data.active_icebergs.end(),
                       [cutoff](const IcebergPattern& p) { 
                           return p.last_refill < cutoff; 
                       }),
        data.active_icebergs.end()
    );
    
    // Limit order tracking size
    if (order_tracking_.size() > config_.max_tracking_orders) {
        // Remove oldest orders
        std::vector<OrderId> to_remove;
        for (const auto& [id, tracking] : order_tracking_) {
            if (tracking.first_seen < cutoff) {
                to_remove.push_back(id);
            }
        }
        
        for (auto id : to_remove) {
            order_tracking_.erase(id);
        }
    }
}

} 
