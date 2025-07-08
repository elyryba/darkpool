#pragma once

#include <deque>
#include <unordered_map>
#include <shared_mutex>
#include "darkpool/types.hpp"

namespace darkpool {

class SlippageTracker {
public:
    struct Config {
        size_t lookback_trades = 100;
        double impact_decay = 0.95;
        bool use_vwap = true;
        double outlier_threshold = 3.0;
        size_t min_trades_for_analysis = 5;
        double abnormal_slippage_threshold = 2.0; // std devs
    };
    
    explicit SlippageTracker(const Config& config = Config{});
    
    // Track trade execution
    void on_trade(const Trade& trade);
    
    // Track order book state for reference prices
    void on_quote(const Quote& quote);
    void on_order_book(const OrderBookSnapshot& book);
    
    // Calculate slippage metrics
    struct SlippageMetrics {
        double immediate_slippage = 0.0;     // vs quote at trade time
        double realized_slippage = 0.0;      // vs arrival price
        double permanent_impact = 0.0;        // lasting price impact
        double temporary_impact = 0.0;        // transient impact
        double total_cost = 0.0;             // in price units
        size_t sample_size = 0;
    };
    
    SlippageMetrics calculate_slippage(Symbol symbol) const;
    
    // Detect abnormal slippage
    std::optional<Anomaly> check_anomaly(Symbol symbol) const;
    
    // Get historical slippage stats
    struct HistoricalStats {
        double avg_slippage = 0.0;
        double std_dev = 0.0;
        double max_slippage = 0.0;
        double percentile_95 = 0.0;
        size_t total_trades = 0;
    };
    
    HistoricalStats get_historical_stats(Symbol symbol) const;
    
    // Reset tracking
    void reset(Symbol symbol);
    void reset_all();
    
private:
    struct TradeRecord {
        Timestamp timestamp;
        Price execution_price;
        Quantity quantity;
        Side side;
        Price mid_quote;      // Mid quote at execution
        Price arrival_price;  // Price when order arrived
    };
    
    struct PriceReference {
        Price bid;
        Price ask;
        Timestamp timestamp;
    };
    
    struct SymbolData {
        std::deque<TradeRecord> trades;
        PriceReference current_quote;
        std::deque<PriceReference> price_history;
        
        // Running statistics
        double slippage_sum = 0.0;
        double slippage_sum_sq = 0.0;
        double max_slippage = 0.0;
        size_t slippage_samples = 0;
    };
    
    // Calculate price impact components
    struct ImpactComponents {
        double immediate = 0.0;
        double permanent = 0.0;
        double temporary = 0.0;
    };
    
    ImpactComponents decompose_impact(const TradeRecord& trade, 
                                     const SymbolData& data) const;
    
    // Calculate VWAP for reference
    Price calculate_vwap(const std::deque<TradeRecord>& trades, 
                        size_t lookback) const;
    
    // Detect outliers using robust statistics
    bool is_outlier(double slippage, const SymbolData& data) const;
    
    Config config_;
    mutable std::unordered_map<Symbol, SymbolData> symbol_data_;
    mutable std::shared_mutex data_mutex_;
};

} 
