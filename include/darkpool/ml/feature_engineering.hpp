#pragma once

#include <vector>
#include <deque>
#include <unordered_map>
#include <memory>
#include <Eigen/Dense>
#include "darkpool/types.hpp"

namespace darkpool {

// Forward declarations for detector outputs
class TradeToQuoteRatio;
class SlippageTracker;
class OrderBookImbalance;
class HiddenRefillDetector;
class TradeClustering;
class HawkesProcess;
class HiddenMarkovModel;
class VPINCalculator;
class PINModel;
class PostTradeDrift;

class FeatureEngineering {
public:
    struct Config {
        size_t lookback_window = 100;          // Number of events for features
        size_t time_buckets = 10;              // Time-based features
        bool use_technical_features = true;     // Price-based indicators
        bool use_microstructure = true;        // Market microstructure
        bool use_network_features = true;      // Cross-asset correlations
        bool normalize_features = true;        // Z-score normalization
        size_t feature_update_interval = 10;   // Update every N events
        double outlier_threshold = 4.0;        // Clip outliers at N std devs
    };
    
    explicit FeatureEngineering(const Config& config = Config{});
    
    // Set detector references for derived features
    void set_detectors(
        std::shared_ptr<TradeToQuoteRatio> tqr,
        std::shared_ptr<SlippageTracker> slippage,
        std::shared_ptr<OrderBookImbalance> order_book,
        std::shared_ptr<HiddenRefillDetector> refill,
        std::shared_ptr<TradeClustering> clustering,
        std::shared_ptr<HawkesProcess> hawkes,
        std::shared_ptr<HiddenMarkovModel> hmm,
        std::shared_ptr<VPINCalculator> vpin,
        std::shared_ptr<PINModel> pin,
        std::shared_ptr<PostTradeDrift> drift
    );
    
    // Process market events
    void on_trade(const Trade& trade);
    void on_quote(const Quote& quote);
    void on_order_book(const OrderBookSnapshot& book);
    
    // Feature extraction result
    struct FeatureVector {
        Symbol symbol;
        Timestamp timestamp;
        Eigen::VectorXf features;           // Dense feature vector
        std::vector<std::string> names;     // Feature names
        std::vector<float> importances;     // Feature importances
        bool is_valid;                      // Has enough data
        
        size_t size() const { return features.size(); }
        float operator[](size_t i) const { return features(i); }
    };
    
    // Get current feature vector for symbol
    FeatureVector get_features(Symbol symbol) const;
    
    // Get feature names and descriptions
    std::vector<std::string> get_feature_names() const;
    size_t get_feature_count() const { return feature_names_.size(); }
    
    // Feature statistics for monitoring
    struct FeatureStats {
        std::vector<float> means;
        std::vector<float> std_devs;
        std::vector<float> mins;
        std::vector<float> maxs;
        std::vector<float> missing_rates;
    };
    
    FeatureStats get_feature_stats() const;
    
private:
    // Feature categories
    struct MicrostructureFeatures {
        float bid_ask_spread;
        float bid_ask_imbalance;
        float depth_imbalance[5];    // Multiple levels
        float quote_intensity;
        float effective_spread;
        float realized_spread;
        float price_impact;
    };
    
    struct OrderFlowFeatures {
        float trade_intensity;
        float buy_sell_imbalance;
        float trade_size_mean;
        float trade_size_std;
        float large_trade_ratio;
        float order_flow_toxicity;
        float kyle_lambda;
    };
    
    struct PriceFeatures {
        float returns[5];            // Multiple horizons
        float volatility;
        float realized_volatility;
        float price_efficiency;
        float autocorrelation;
        float momentum[3];           // Short/medium/long
        float mean_reversion_speed;
    };
    
    struct VolumeFeatures {
        float volume_mean;
        float volume_std;
        float vwap_deviation;
        float volume_clock_intensity;
        float participation_rate;
        float volume_concentration;
    };
    
    struct TimeFeatures {
        float time_of_day;           // Normalized 0-1
        float time_since_open;
        float time_to_close;
        float day_of_week;
        float is_opening_auction;
        float is_closing_auction;
        float time_since_last_trade;
    };
    
    struct NetworkFeatures {
        float market_correlation;
        float sector_correlation;
        float lead_lag_ratio;
        float information_share;
        float common_factor_exposure;
    };
    
    struct DetectorFeatures {
        float tqr_value;
        float tqr_zscore;
        float slippage_immediate;
        float slippage_permanent;
        float order_book_pressure;
        float hidden_ratio;
        float hawkes_intensity;
        float hmm_state_prob[3];
        float vpin_value;
        float pin_value;
        float drift_magnitude;
    };
    
    struct SymbolFeatures {
        // Raw market data
        std::deque<Trade> trades;
        std::deque<Quote> quotes;
        std::deque<OrderBookSnapshot> books;
        
        // Computed features
        MicrostructureFeatures microstructure;
        OrderFlowFeatures order_flow;
        PriceFeatures price;
        VolumeFeatures volume;
        TimeFeatures time;
        NetworkFeatures network;
        DetectorFeatures detectors;
        
        // Feature vector cache
        mutable Eigen::VectorXf cached_features;
        mutable Timestamp cache_timestamp = 0;
        
        // Running statistics for normalization
        Eigen::VectorXf feature_sums;
        Eigen::VectorXf feature_sum_squares;
        size_t sample_count = 0;
    };
    
    // Feature computation methods
    void compute_microstructure_features(SymbolFeatures& features) const;
    void compute_orderflow_features(SymbolFeatures& features) const;
    void compute_price_features(SymbolFeatures& features) const;
    void compute_volume_features(SymbolFeatures& features) const;
    void compute_time_features(SymbolFeatures& features, Timestamp current_time) const;
    void compute_network_features(SymbolFeatures& features, Symbol symbol) const;
    void compute_detector_features(SymbolFeatures& features, Symbol symbol) const;
    
    // Build feature vector from components
    Eigen::VectorXf build_feature_vector(const SymbolFeatures& features) const;
    
    // Normalize features using running statistics
    void normalize_features(Eigen::VectorXf& features, const SymbolFeatures& symbol_features) const;
    
    // Detect and handle outliers
    void handle_outliers(Eigen::VectorXf& features) const;
    
    Config config_;
    
    // Symbol data
    mutable std::unordered_map<Symbol, SymbolFeatures> symbol_features_;
    mutable std::shared_mutex data_mutex_;
    
    // Detector references
    std::shared_ptr<TradeToQuoteRatio> tqr_;
    std::shared_ptr<SlippageTracker> slippage_;
    std::shared_ptr<OrderBookImbalance> order_book_;
    std::shared_ptr<HiddenRefillDetector> refill_;
    std::shared_ptr<TradeClustering> clustering_;
    std::shared_ptr<HawkesProcess> hawkes_;
    std::shared_ptr<HiddenMarkovModel> hmm_;
    std::shared_ptr<VPINCalculator> vpin_;
    std::shared_ptr<PINModel> pin_;
    std::shared_ptr<PostTradeDrift> drift_;
    
    // Feature metadata
    std::vector<std::string> feature_names_;
    std::vector<float> feature_importances_;
    
    // Global statistics
    mutable FeatureStats global_stats_;
    
    void initialize_feature_names();
};

// Feature vector batch for ML inference
struct FeatureBatch {
    std::vector<Symbol> symbols;
    std::vector<Timestamp> timestamps;
    Eigen::MatrixXf features;  // Each row is a feature vector
    
    size_t batch_size() const { return symbols.size(); }
    size_t feature_dim() const { return features.cols(); }
};

} 
