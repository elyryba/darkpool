#pragma once

#include <deque>
#include <unordered_map>
#include <mutex>
#include "darkpool/types.hpp"
#include "darkpool/utils/ring_buffer.hpp"

namespace darkpool {

class TradeToQuoteRatio {
public:
    struct Config {
        size_t window_size = 1000;
        double threshold = 2.5;
        size_t min_trades = 10;
        bool adaptive_threshold = true;
        double adaptive_factor = 1.5;
        size_t quote_decay_ms = 100;
    };
    
    explicit TradeToQuoteRatio(const Config& config = Config{});
    
    // Process market events
    void on_trade(const Trade& trade);
    void on_quote(const Quote& quote);
    
    // Calculate TQR for a symbol
    double calculate_tqr(Symbol symbol) const;
    
    // Check for anomaly
    std::optional<Anomaly> check_anomaly(Symbol symbol) const;
    
    // Get statistics
    struct SymbolStats {
        size_t trade_count = 0;
        size_t quote_count = 0;
        double current_tqr = 0.0;
        double avg_tqr = 0.0;
        double std_dev = 0.0;
        Timestamp last_update = 0;
    };
    
    SymbolStats get_stats(Symbol symbol) const;
    
    // Reset symbol data
    void reset(Symbol symbol);
    void reset_all();
    
private:
    struct TradeEvent {
        Timestamp timestamp;
        Quantity quantity;
        Price price;
        Side side;
    };
    
    struct QuoteEvent {
        Timestamp timestamp;
        Price bid_price;
        Price ask_price;
        Quantity bid_size;
        Quantity ask_size;
    };
    
    struct SymbolData {
        std::deque<TradeEvent> trades;
        std::deque<QuoteEvent> quotes;
        mutable double cached_tqr = 0.0;
        mutable Timestamp cache_timestamp = 0;
        
        // Running statistics
        size_t total_trades = 0;
        size_t total_quotes = 0;
        double tqr_sum = 0.0;
        double tqr_sum_sq = 0.0;
        size_t tqr_samples = 0;
    };
    
    // Clean old events outside window
    void clean_window(SymbolData& data, Timestamp current_time) const;
    
    // Calculate adaptive threshold
    double calculate_adaptive_threshold(const SymbolData& data) const;
    
    // Update running statistics
    void update_statistics(SymbolData& data, double tqr) const;
    
    Config config_;
    mutable std::unordered_map<Symbol, SymbolData> symbol_data_;
    mutable std::shared_mutex data_mutex_;
};

// Inline implementation for hot path
inline void TradeToQuoteRatio::on_trade(const Trade& trade) {
    std::unique_lock lock(data_mutex_);
    auto& data = symbol_data_[trade.symbol];
    
    data.trades.push_back({
        trade.timestamp,
        trade.quantity,
        trade.price,
        trade.aggressor_side
    });
    
    data.total_trades++;
    
    // Clean old events
    clean_window(data, trade.timestamp);
}

inline void TradeToQuoteRatio::on_quote(const Quote& quote) {
    std::unique_lock lock(data_mutex_);
    auto& data = symbol_data_[quote.symbol];
    
    data.quotes.push_back({
        quote.timestamp,
        quote.bid_price,
        quote.ask_price,
        quote.bid_size,
        quote.ask_size
    });
    
    data.total_quotes++;
    
    // Clean old events
    clean_window(data, quote.timestamp);
}

} 
