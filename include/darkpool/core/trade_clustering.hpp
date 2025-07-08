#pragma once

#include <vector>
#include <deque>
#include <unordered_map>
#include <shared_mutex>
#include <Eigen/Dense>
#include "darkpool/types.hpp"

namespace darkpool {

class TradeClustering {
public:
    struct Config {
        size_t window_size = 500;              // Number of trades to analyze
        size_t num_clusters = 5;               // Number of HMM states
        double convergence_threshold = 1e-6;    // EM convergence criteria
        size_t max_iterations = 100;           // Max EM iterations
        double anomaly_threshold = 0.001;      // State probability threshold
        size_t min_cluster_size = 10;          // Min trades per cluster
        double dark_pool_signature = 0.7;      // Threshold for dark pool detection
    };
    
    explicit TradeClustering(const Config& config = Config{});
    
    // Process market events
    void on_trade(const Trade& trade);
    void on_quote(const Quote& quote);
    
    // Clustering results
    struct ClusterInfo {
        size_t cluster_id;
        double probability;
        size_t trade_count;
        Quantity avg_size;
        double price_volatility;
        double time_clustering;     // Burst intensity
        bool likely_dark_pool;
    };
    
    struct RegimeInfo {
        enum class Regime {
            NORMAL,
            AGGRESSIVE_BUYING,
            AGGRESSIVE_SELLING,
            ACCUMULATION,
            DISTRIBUTION,
            DARK_POOL_ACTIVE
        };
        
        Regime current_regime;
        double confidence;
        Timestamp regime_start;
        std::vector<ClusterInfo> active_clusters;
    };
    
    // Get current market regime
    RegimeInfo get_current_regime(Symbol symbol) const;
    
    // Detect anomalous trading patterns
    std::optional<Anomaly> check_anomaly(Symbol symbol) const;
    
    // Get cluster assignment for recent trades
    std::vector<size_t> get_trade_clusters(Symbol symbol, size_t n_trades = 100) const;
    
    // Statistics
    struct Stats {
        size_t regime_changes = 0;
        size_t dark_pool_detections = 0;
        double avg_cluster_purity = 0.0;
        size_t total_trades_processed = 0;
    };
    
    Stats get_stats() const { return stats_; }
    
private:
    // Trade features for clustering
    struct TradeFeatures {
        double normalized_size;      // Size relative to average
        double price_change;        // Price change from previous
        double time_delta;          // Time since last trade
        double spread_ratio;        // Execution vs quoted spread
        double side_imbalance;      // Buy/sell pressure
        double volume_rate;         // Volume per time unit
    };
    
    // Hidden Markov Model components
    struct HMMState {
        Eigen::VectorXd mean;           // Feature means
        Eigen::MatrixXd covariance;     // Feature covariance
        double prior;                   // State probability
    };
    
    struct HMMModel {
        std::vector<HMMState> states;
        Eigen::MatrixXd transition_matrix;
        Eigen::MatrixXd emission_probs;
        bool is_trained = false;
    };
    
    struct SymbolData {
        std::deque<Trade> trades;
        std::deque<Quote> quotes;
        std::vector<TradeFeatures> features;
        HMMModel hmm_model;
        RegimeInfo current_regime;
        
        // Running statistics
        double avg_trade_size = 0.0;
        double avg_time_between = 0.0;
        double price_volatility = 0.0;
    };
    
    // Feature extraction
    TradeFeatures extract_features(const Trade& trade, const SymbolData& data) const;
    
    // HMM training using Baum-Welch algorithm
    void train_hmm(HMMModel& model, const std::vector<TradeFeatures>& features) const;
    
    // Viterbi algorithm for most likely state sequence
    std::vector<size_t> viterbi_decode(const HMMModel& model, 
                                       const std::vector<TradeFeatures>& features) const;
    
    // Detect regime from cluster patterns
    RegimeInfo::Regime detect_regime(const std::vector<size_t>& clusters,
                                     const std::vector<ClusterInfo>& cluster_info) const;
    
    // Check for dark pool signatures
    bool has_dark_pool_signature(const ClusterInfo& cluster) const;
    
    // Update running statistics
    void update_statistics(SymbolData& data, const Trade& trade);
    
    Config config_;
    mutable std::unordered_map<Symbol, SymbolData> symbol_data_;
    mutable std::shared_mutex data_mutex_;
    mutable Stats stats_;
};

} 
