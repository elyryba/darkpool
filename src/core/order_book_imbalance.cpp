#include "darkpool/core/order_book_imbalance.hpp"
#include <cmath>
#include <algorithm>

namespace darkpool {

OrderBookImbalance::OrderBookImbalance(const Config& config) : config_(config) {
    symbol_data_.reserve(1000);
}

void OrderBookImbalance::on_order_book(const OrderBookSnapshot& book) {
    std::unique_lock lock(data_mutex_);
    auto& data = symbol_data_[book.symbol];
    
    // Update current book state
    BookState& state = data.current_book;
    state.timestamp = book.timestamp;
    
    // Copy bid levels
    for (size_t i = 0; i < std::min(book.bids.size(), state.bids.size()); ++i) {
        state.bids[i] = book.bids[i];
    }
    
    // Copy ask levels
    for (size_t i = 0; i < std::min(book.asks.size(), state.asks.size()); ++i) {
        state.asks[i] = book.asks[i];
    }
    
    // Store history
    data.book_history.push_back(state);
    
    // Clean old history
    clean_window(data, book.timestamp);
    
    // Calculate and store pressure point
    auto metrics = calculate_imbalance(book.symbol);
    if (metrics.pressure_score > 0.5) {
        PressurePoint point{
            book.timestamp,
            metrics.pressure_score,
            metrics.pressure_side,
            std::min(1.0, metrics.pressure_score)
        };
        data.pressure_history.push_back(point);
        
        // Limit history size
        while (data.pressure_history.size() > 1000) {
            data.pressure_history.pop_front();
        }
    }
    
    // Update statistics
    double imbalance = state.calculate_weighted_imbalance(config_.depth_levels);
    data.imbalance_sum += std::abs(imbalance);
    data.imbalance_sum_sq += imbalance * imbalance;
    data.imbalance_samples++;
}

void OrderBookImbalance::on_order(const Order& order) {
    std::unique_lock lock(data_mutex_);
    auto& data = symbol_data_[order.symbol];
    update_order_flow(data, order);
}

void OrderBookImbalance::on_trade(const Trade& trade) {
    std::unique_lock lock(data_mutex_);
    auto& data = symbol_data_[trade.symbol];
    update_order_flow(data, trade);
}

OrderBookImbalance::ImbalanceMetrics OrderBookImbalance::calculate_imbalance(Symbol symbol) const {
    std::shared_lock lock(data_mutex_);
    
    auto it = symbol_data_.find(symbol);
    if (it == symbol_data_.end()) {
        return ImbalanceMetrics{};
    }
    
    const auto& data = it->second;
    ImbalanceMetrics metrics;
    
    // Level 1 imbalance
    if (data.current_book.bids[0].quantity > 0 || data.current_book.asks[0].quantity > 0) {
        double bid_vol = data.current_book.bids[0].quantity;
        double ask_vol = data.current_book.asks[0].quantity;
        metrics.level1_imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol);
    }
    
    // Weighted imbalance
    metrics.weighted_imbalance = data.current_book.calculate_weighted_imbalance(config_.depth_levels);
    
    // Calculate pressure score
    double pressure = 0.0;
    double decay_factor = 1.0;
    
    for (auto it = data.book_history.rbegin(); it != data.book_history.rend(); ++it) {
        double imbalance = it->calculate_weighted_imbalance(config_.depth_levels);
        pressure += imbalance * decay_factor;
        decay_factor *= config_.pressure_decay;
        
        if (decay_factor < 0.01) break;
    }
    
    metrics.pressure_score = std::abs(pressure);
    metrics.pressure_side = pressure > 0 ? Side::BUY : Side::SELL;
    
    // Hidden ratio
    metrics.hidden_ratio = estimate_hidden_ratio(data);
    
    metrics.last_update = data.current_book.timestamp;
    
    return metrics;
}

std::optional<Anomaly> OrderBookImbalance::check_anomaly(Symbol symbol) const {
    auto metrics = calculate_imbalance(symbol);
    
    std::shared_lock lock(data_mutex_);
    auto it = symbol_data_.find(symbol);
    if (it == symbol_data_.end() || it->second.imbalance_samples < config_.min_samples) {
        return std::nullopt;
    }
    
    const auto& data = it->second;
    
    // Check for persistent imbalance
    bool persistent_imbalance = std::abs(metrics.weighted_imbalance) > config_.imbalance_threshold;
    
    // Check for hidden liquidity
    bool hidden_pressure = metrics.hidden_ratio > config_.hidden_ratio_threshold;
    
    // Check for pressure buildup
    bool pressure_buildup = detect_pressure_buildup(data);
    
    if (persistent_imbalance && (hidden_pressure || pressure_buildup)) {
        Anomaly anomaly;
        anomaly.symbol = symbol;
        anomaly.timestamp = metrics.last_update;
        anomaly.type = AnomalyType::ORDER_BOOK_PRESSURE;
        
        // Confidence based on multiple factors
        double confidence = 0.0;
        if (persistent_imbalance) confidence += 0.3;
        if (hidden_pressure) confidence += 0.4;
        if (pressure_buildup) confidence += 0.3;
        anomaly.confidence = std::min(1.0, confidence);
        
        anomaly.magnitude = metrics.pressure_score;
        
        // Estimate hidden size based on imbalance and hidden ratio
        double visible_imbalance = std::abs(metrics.weighted_imbalance);
        anomaly.estimated_hidden_size = static_cast<Quantity>(
            visible_imbalance * metrics.hidden_ratio * 10000
        );
        
        snprintf(anomaly.description.data(), anomaly.description.size(),
                "Order book pressure detected: imbalance=%.2f, hidden_ratio=%.2f, pressure=%.2f",
                metrics.weighted_imbalance, metrics.hidden_ratio, metrics.pressure_score);
        
        return anomaly;
    }
    
    return std::nullopt;
}

std::vector<OrderBookImbalance::PressurePoint> OrderBookImbalance::get_pressure_history(
    Symbol symbol, size_t max_points) const {
    
    std::shared_lock lock(data_mutex_);
    
    auto it = symbol_data_.find(symbol);
    if (it == symbol_data_.end()) {
        return {};
    }
    
    const auto& history = it->second.pressure_history;
    
    std::vector<PressurePoint> result;
    size_t start = history.size() > max_points ? history.size() - max_points : 0;
    
    for (size_t i = start; i < history.size(); ++i) {
        result.push_back(history[i]);
    }
    
    return result;
}

void OrderBookImbalance::reset(Symbol symbol) {
    std::unique_lock lock(data_mutex_);
    symbol_data_.erase(symbol);
}

void OrderBookImbalance::reset_all() {
    std::unique_lock lock(data_mutex_);
    symbol_data_.clear();
}

double OrderBookImbalance::estimate_hidden_ratio(const SymbolData& data) const {
    if (data.order_flow.window_start == 0) {
        return 1.0;
    }
    
    double visible_volume = 0.0;
    for (size_t i = 0; i < config_.depth_levels && i < data.current_book.bids.size(); ++i) {
        visible_volume += data.current_book.bids[i].quantity;
        visible_volume += data.current_book.asks[i].quantity;
    }
    
    if (visible_volume == 0) return 1.0;
    
    double flow_volume = data.order_flow.buy_volume + data.order_flow.sell_volume;
    double hidden_volume = data.order_flow.hidden_buy_volume + data.order_flow.hidden_sell_volume;
    
    // Ratio of flow to visible volume, adjusted for hidden trades
    double flow_ratio = flow_volume / visible_volume;
    double hidden_ratio = hidden_volume > 0 ? hidden_volume / flow_volume : 0.0;
    
    return flow_ratio * (1.0 + hidden_ratio);
}

bool OrderBookImbalance::detect_pressure_buildup(const SymbolData& data) const {
    if (data.pressure_history.size() < 5) {
        return false;
    }
    
    // Check for increasing pressure over time
    double recent_avg = 0.0;
    double older_avg = 0.0;
    size_t recent_count = 0;
    size_t older_count = 0;
    
    size_t mid_point = data.pressure_history.size() - 5;
    
    for (size_t i = 0; i < data.pressure_history.size(); ++i) {
        if (i < mid_point) {
            older_avg += data.pressure_history[i].pressure;
            older_count++;
        } else {
            recent_avg += data.pressure_history[i].pressure;
            recent_count++;
        }
    }
    
    if (older_count > 0) older_avg /= older_count;
    if (recent_count > 0) recent_avg /= recent_count;
    
    // Pressure is building if recent > older by significant margin
    return recent_avg > older_avg * 1.5;
}

void OrderBookImbalance::update_order_flow(SymbolData& data, const Order& order) {
    Timestamp current_time = order.timestamp;
    
    // Reset window if needed
    if (data.order_flow.window_start == 0 || 
        current_time - data.order_flow.window_start > config_.time_window_ms * 1000000) {
        data.order_flow = OrderFlow{};
        data.order_flow.window_start = current_time;
    }
    
    // Update flow volumes
    if (order.side == Side::BUY) {
        data.order_flow.buy_volume += order.quantity;
        if (order.type == OrderType::HIDDEN) {
            data.order_flow.hidden_buy_volume += order.quantity;
        }
    } else {
        data.order_flow.sell_volume += order.quantity;
        if (order.type == OrderType::HIDDEN) {
            data.order_flow.hidden_sell_volume += order.quantity;
        }
    }
}

void OrderBookImbalance::update_order_flow(SymbolData& data, const Trade& trade) {
    Timestamp current_time = trade.timestamp;
    
    // Reset window if needed
    if (data.order_flow.window_start == 0 || 
        current_time - data.order_flow.window_start > config_.time_window_ms * 1000000) {
        data.order_flow = OrderFlow{};
        data.order_flow.window_start = current_time;
    }
    
    // Update flow volumes
    if (trade.aggressor_side == Side::BUY) {
        data.order_flow.buy_volume += trade.quantity;
        if (trade.is_hidden) {
            data.order_flow.hidden_buy_volume += trade.quantity;
        }
    } else {
        data.order_flow.sell_volume += trade.quantity;
        if (trade.is_hidden) {
            data.order_flow.hidden_sell_volume += trade.quantity;
        }
    }
}

void OrderBookImbalance::clean_window(SymbolData& data, Timestamp current_time) const {
    Timestamp cutoff = current_time - (config_.time_window_ms * 1000000);
    
    // Remove old book snapshots
    while (!data.book_history.empty() && data.book_history.front().timestamp < cutoff) {
        data.book_history.pop_front();
    }
    
    // Reset order flow if window expired
    if (data.order_flow.window_start > 0 && 
        current_time - data.order_flow.window_start > config_.time_window_ms * 1000000) {
        data.order_flow = OrderFlow{};
    }
}

} 
