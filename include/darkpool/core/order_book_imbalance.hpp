#pragma once

#include <array>
#include <deque>
#include <unordered_map>
#include <shared_mutex>
#include "darkpool/types.hpp"

namespace darkpool {

class OrderBookImbalance {
public:
    struct Config {
        size_t depth_levels = 5;
        double imbalance_threshold = 0.7;
        size_t time_window_ms = 1000;
        double pressure_decay = 0.95;
        size_t min_samples = 10;
        bool use_volume_weighting = true;
        double hidden_ratio_threshold = 2.0;
    };
    
    explicit OrderBookImbalance(const Config& config = Config{});
    
    // Update order book state
    void on_order_book(const OrderBookSnapshot& book);
    void on_order(const Order& order);
    void on_trade(const Trade& trade);
    
    // Calculate imbalance metrics
    struct ImbalanceMetrics {
        double level1_imbalance = 0.0;     // Top of book imbalance
        double weighted_imbalance = 0.0;    // Volume-weighted across levels
        double pressure_score = 0.0;        // Directional pressure indicator
        double hidden_ratio = 0.0;          // Hidden vs visible volume
        Side pressure_side = Side::UNKNOWN;
        Timestamp last_update = 0;
    };
    
    ImbalanceMetrics calculate_imbalance(Symbol symbol) const;
    
    // Detect hidden liquidity pressure
    std::optional<Anomaly> check_anomaly(Symbol symbol) const;
    
    // Get pressure history
    struct PressurePoint {
        Timestamp timestamp;
        double pressure;
        Side direction;
        double confidence;
    };
    
    std::vector<PressurePoint> get_pressure_history(Symbol symbol, size_t max_points = 100) const;
    
    // Reset tracking
    void reset(Symbol symbol);
    void reset_all();
    
private:
    struct BookState {
        std::array<OrderBookLevel, 10> bids;
        std::array<OrderBookLevel, 10> asks;
        Timestamp timestamp;
        
        double calculate_imbalance(size_t levels) const;
        double calculate_weighted_imbalance(size_t levels) const;
    };
    
    struct OrderFlow {
        Quantity buy_volume = 0;
        Quantity sell_volume = 0;
        Quantity hidden_buy_volume = 0;
        Quantity hidden_sell_volume = 0;
        Timestamp window_start = 0;
    };
    
    struct SymbolData {
        BookState current_book;
        std::deque<BookState> book_history;
        OrderFlow order_flow;
        std::deque<PressurePoint> pressure_history;
        
        // Running statistics
        double imbalance_sum = 0.0;
        double imbalance_sum_sq = 0.0;
        size_t imbalance_samples = 0;
    };
    
    // Calculate hidden volume indicators
    double estimate_hidden_ratio(const SymbolData& data) const;
    
    // Detect pressure anomalies using time series analysis
    bool detect_pressure_buildup(const SymbolData& data) const;
    
    // Update order flow metrics
    void update_order_flow(SymbolData& data, const Order& order);
    void update_order_flow(SymbolData& data, const Trade& trade);
    
    // Clean old data
    void clean_window(SymbolData& data, Timestamp current_time) const;
    
    Config config_;
    mutable std::unordered_map<Symbol, SymbolData> symbol_data_;
    mutable std::shared_mutex data_mutex_;
};

// Inline implementations for performance
inline double OrderBookImbalance::BookState::calculate_imbalance(size_t levels) const {
    double bid_volume = 0.0;
    double ask_volume = 0.0;
    
    for (size_t i = 0; i < std::min(levels, bids.size()); ++i) {
        bid_volume += bids[i].quantity;
        ask_volume += asks[i].quantity;
    }
    
    double total = bid_volume + ask_volume;
    return total > 0 ? (bid_volume - ask_volume) / total : 0.0;
}

inline double OrderBookImbalance::BookState::calculate_weighted_imbalance(size_t levels) const {
    double weighted_bid = 0.0;
    double weighted_ask = 0.0;
    double total_weight = 0.0;
    
    for (size_t i = 0; i < std::min(levels, bids.size()); ++i) {
        double weight = 1.0 / (i + 1);  // Decay by level
        weighted_bid += bids[i].quantity * weight;
        weighted_ask += asks[i].quantity * weight;
        total_weight += weight * 2;
    }
    
    double total = weighted_bid + weighted_ask;
    return total > 0 ? (weighted_bid - weighted_ask) / total : 0.0;
}

} 
