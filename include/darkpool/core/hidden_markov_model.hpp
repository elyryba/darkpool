#pragma once

#include <vector>
#include <array>
#include <unordered_map>
#include <shared_mutex>
#include <Eigen/Dense>
#include "darkpool/types.hpp"

namespace darkpool {

class HiddenMarkovModel {
public:
    struct Config {
        size_t num_states = 3;                    // Number of hidden states
        size_t observation_window = 500;          // Number of observations for training
        double transition_asymmetry = 0.2;        // Asymmetry in buy/sell transitions
        double convergence_threshold = 1e-6;      // EM convergence criteria
        size_t max_iterations = 100;              // Max EM iterations
        size_t min_observations = 50;             // Min observations for training
        double regime_stability_threshold = 0.8;  // Min probability for regime
        bool use_volume_states = true;            // Include volume in state definition
    };
    
    // Market regime states
    enum class MarketRegime {
        NORMAL = 0,
        ACCUMULATION = 1,      // Slow buying, possible institutional
        DISTRIBUTION = 2,      // Slow selling
        BUYING_PRESSURE = 3,   // Aggressive buying
        SELLING_PRESSURE = 4,  // Aggressive selling
        DARK_POOL_BUY = 5,    // Hidden buying
        DARK_POOL_SELL = 6    // Hidden selling
    };
    
    explicit HiddenMarkovModel(const Config& config = Config{});
    
    // Process market events
    void on_trade(const Trade& trade);
    void on_quote(const Quote& quote);
    void on_order_book(const OrderBookSnapshot& book);
    
    // Get current market regime
    struct RegimeState {
        MarketRegime regime;
        double probability;
        std::vector<double> state_probabilities;
        Timestamp last_transition;
        size_t consecutive_states;
    };
    
    RegimeState get_current_regime(Symbol symbol) const;
    
    // Detect regime changes and anomalies
    std::optional<Anomaly> check_anomaly(Symbol symbol) const;
    
    // Get most likely state sequence (Viterbi)
    std::vector<size_t> decode_states(Symbol symbol, size_t lookback = 100) const;
    
    // Get regime transition probabilities
    Eigen::MatrixXd get_transition_matrix(Symbol symbol) const;
    
    // Statistics
    struct Stats {
        size_t regime_changes = 0;
        size_t dark_pool_detections = 0;
        double avg_regime_duration_ms = 0.0;
        size_t total_observations = 0;
    };
    
    Stats get_stats() const { return stats_; }
    
private:
    // Observation features
    struct Observation {
        double normalized_volume;     // Volume relative to average
        double price_change;         // Return
        double bid_ask_imbalance;    // Order book pressure
        double trade_intensity;      // Trades per second
        double avg_trade_size;       // Average size
        double quote_intensity;      // Quote updates per second
        Side net_aggressor;         // Buy/sell pressure
        Timestamp timestamp;
    };
    
    // HMM parameters
    struct ModelParameters {
        Eigen::MatrixXd transition_matrix;      // State transitions (asymmetric)
        std::vector<Eigen::VectorXd> emission_means;     // Observation means per state
        std::vector<Eigen::MatrixXd> emission_covariances; // Observation covariances
        Eigen::VectorXd initial_probs;          // Initial state probabilities
        bool is_trained = false;
    };
    
    struct SymbolData {
        std::deque<Trade> trades;
        std::deque<Quote> quotes;
        std::deque<OrderBookSnapshot> books;
        std::vector<Observation> observations;
        ModelParameters model;
        RegimeState current_regime;
        
        // Running statistics
        double avg_volume = 0.0;
        double avg_trade_size = 0.0;
        double baseline_intensity = 0.0;
    };
    
    // Feature extraction
    Observation extract_observation(const SymbolData& data) const;
    
    // Baum-Welch algorithm for parameter estimation
    void train_model(ModelParameters& model, const std::vector<Observation>& obs) const;
    
    // Forward-backward algorithm
    struct ForwardBackwardResult {
        Eigen::MatrixXd alpha;  // Forward probabilities
        Eigen::MatrixXd beta;   // Backward probabilities
        Eigen::MatrixXd gamma;  // State probabilities
        Eigen::MatrixXd xi;     // Transition probabilities
        double log_likelihood;
    };
    
    ForwardBackwardResult forward_backward(const ModelParameters& model,
                                          const std::vector<Observation>& obs) const;
    
    // Viterbi algorithm
    std::vector<size_t> viterbi(const ModelParameters& model,
                               const std::vector<Observation>& obs) const;
    
    // Map states to regimes
    MarketRegime classify_regime(const std::vector<size_t>& states,
                                const std::vector<Observation>& obs) const;
    
    // Initialize asymmetric transition matrix
    Eigen::MatrixXd create_asymmetric_transitions() const;
    
    // Calculate observation probability
    double observation_probability(const Observation& obs,
                                 const Eigen::VectorXd& mean,
                                 const Eigen::MatrixXd& cov) const;
    
    Config config_;
    mutable std::unordered_map<Symbol, SymbolData> symbol_data_;
    mutable std::shared_mutex data_mutex_;
    mutable Stats stats_;
};

} 
