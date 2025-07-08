#include "darkpool/core/trade_clustering.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace darkpool {

TradeClustering::TradeClustering(const Config& config) : config_(config) {
    symbol_data_.reserve(1000);
}

void TradeClustering::on_trade(const Trade& trade) {
    std::unique_lock lock(data_mutex_);
    auto& data = symbol_data_[trade.symbol];
    
    // Add trade to history
    data.trades.push_back(trade);
    while (data.trades.size() > config_.window_size) {
        data.trades.pop_front();
    }
    
    // Update statistics
    update_statistics(data, trade);
    
    // Extract features
    if (data.trades.size() >= 2) {
        auto features = extract_features(trade, data);
        data.features.push_back(features);
        
        while (data.features.size() > config_.window_size) {
            data.features.erase(data.features.begin());
        }
    }
    
    // Retrain HMM periodically
    if (data.features.size() >= config_.window_size && 
        data.trades.size() % 50 == 0) {
        train_hmm(data.hmm_model, data.features);
        
        // Update regime
        auto clusters = viterbi_decode(data.hmm_model, data.features);
        
        // Build cluster info
        std::vector<ClusterInfo> cluster_info(config_.num_clusters);
        for (size_t i = 0; i < clusters.size(); ++i) {
            size_t cluster = clusters[i];
            cluster_info[cluster].cluster_id = cluster;
            cluster_info[cluster].trade_count++;
            cluster_info[cluster].avg_size += data.trades[i].quantity;
        }
        
        // Finalize cluster info
        for (auto& info : cluster_info) {
            if (info.trade_count > 0) {
                info.avg_size /= info.trade_count;
                info.probability = static_cast<double>(info.trade_count) / clusters.size();
                info.likely_dark_pool = has_dark_pool_signature(info);
            }
        }
        
        // Detect regime
        data.current_regime.current_regime = detect_regime(clusters, cluster_info);
        data.current_regime.active_clusters = cluster_info;
        data.current_regime.confidence = 0.8; // Simplified
        
        stats_.total_trades_processed++;
    }
}

void TradeClustering::on_quote(const Quote& quote) {
    std::unique_lock lock(data_mutex_);
    auto& data = symbol_data_[quote.symbol];
    
    data.quotes.push_back(quote);
    while (data.quotes.size() > config_.window_size) {
        data.quotes.pop_front();
    }
}

TradeClustering::RegimeInfo TradeClustering::get_current_regime(Symbol symbol) const {
    std::shared_lock lock(data_mutex_);
    
    auto it = symbol_data_.find(symbol);
    if (it == symbol_data_.end()) {
        return RegimeInfo{RegimeInfo::Regime::NORMAL, 0.0, 0, {}};
    }
    
    return it->second.current_regime;
}

std::optional<Anomaly> TradeClustering::check_anomaly(Symbol symbol) const {
    auto regime = get_current_regime(symbol);
    
    // Check for dark pool activity
    bool dark_pool_detected = false;
    double max_dark_pool_prob = 0.0;
    
    for (const auto& cluster : regime.active_clusters) {
        if (cluster.likely_dark_pool && cluster.probability > config_.dark_pool_signature) {
            dark_pool_detected = true;
            max_dark_pool_prob = std::max(max_dark_pool_prob, cluster.probability);
        }
    }
    
    if (dark_pool_detected || regime.current_regime == RegimeInfo::Regime::DARK_POOL_ACTIVE) {
        Anomaly anomaly;
        anomaly.symbol = symbol;
        anomaly.timestamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        anomaly.type = AnomalyType::TRADE_CLUSTERING;
        anomaly.confidence = regime.confidence * max_dark_pool_prob;
        anomaly.magnitude = max_dark_pool_prob;
        
        // Estimate hidden size based on cluster characteristics
        Quantity total_cluster_volume = 0;
        for (const auto& cluster : regime.active_clusters) {
            if (cluster.likely_dark_pool) {
                total_cluster_volume += cluster.avg_size * cluster.trade_count;
            }
        }
        anomaly.estimated_hidden_size = total_cluster_volume;
        
        snprintf(anomaly.description.data(), anomaly.description.size(),
                "Dark pool activity detected via clustering: regime=%d, confidence=%.2f",
                static_cast<int>(regime.current_regime), anomaly.confidence);
        
        stats_.dark_pool_detections++;
        return anomaly;
    }
    
    // Check for regime anomalies
    if (regime.current_regime == RegimeInfo::Regime::AGGRESSIVE_BUYING ||
        regime.current_regime == RegimeInfo::Regime::AGGRESSIVE_SELLING) {
        
        Anomaly anomaly;
        anomaly.symbol = symbol;
        anomaly.timestamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        anomaly.type = AnomalyType::TRADE_CLUSTERING;
        anomaly.confidence = regime.confidence;
        anomaly.magnitude = 1.0;
        
        snprintf(anomaly.description.data(), anomaly.description.size(),
                "Aggressive %s regime detected via clustering",
                regime.current_regime == RegimeInfo::Regime::AGGRESSIVE_BUYING ? "buying" : "selling");
        
        return anomaly;
    }
    
    return std::nullopt;
}

std::vector<size_t> TradeClustering::get_trade_clusters(Symbol symbol, size_t n_trades) const {
    std::shared_lock lock(data_mutex_);
    
    auto it = symbol_data_.find(symbol);
    if (it == symbol_data_.end() || !it->second.hmm_model.is_trained) {
        return {};
    }
    
    const auto& data = it->second;
    size_t start = data.features.size() > n_trades ? data.features.size() - n_trades : 0;
    
    std::vector<TradeFeatures> recent_features(
        data.features.begin() + start, data.features.end()
    );
    
    return viterbi_decode(data.hmm_model, recent_features);
}

TradeClustering::TradeFeatures TradeClustering::extract_features(
    const Trade& trade, const SymbolData& data) const {
    
    TradeFeatures features;
    
    // Normalized size
    features.normalized_size = data.avg_trade_size > 0 ? 
        trade.quantity / data.avg_trade_size : 1.0;
    
    // Price change
    if (!data.trades.empty()) {
        const auto& prev_trade = data.trades[data.trades.size() - 2];
        features.price_change = (trade.price - prev_trade.price) / 
                               static_cast<double>(prev_trade.price);
        
        // Time delta
        features.time_delta = (trade.timestamp - prev_trade.timestamp) / 1000000000.0;
    } else {
        features.price_change = 0.0;
        features.time_delta = 1.0;
    }
    
    // Spread ratio
    if (!data.quotes.empty()) {
        const auto& quote = data.quotes.back();
        double spread = quote.ask_price - quote.bid_price;
        double mid = (quote.ask_price + quote.bid_price) / 2.0;
        features.spread_ratio = spread > 0 ? 
            std::abs(trade.price - mid) / spread : 0.0;
    } else {
        features.spread_ratio = 0.5;
    }
    
    // Side imbalance (simplified)
    features.side_imbalance = trade.aggressor_side == Side::BUY ? 1.0 : -1.0;
    
    // Volume rate
    features.volume_rate = features.time_delta > 0 ? 
        trade.quantity / features.time_delta : trade.quantity;
    
    return features;
}

void TradeClustering::train_hmm(HMMModel& model, 
                               const std::vector<TradeFeatures>& features) const {
    if (features.size() < config_.window_size) {
        return;
    }
    
    const size_t n_features = 6;
    const size_t n_states = config_.num_clusters;
    const size_t n_obs = features.size();
    
    // Convert features to matrix
    Eigen::MatrixXd obs(n_obs, n_features);
    for (size_t i = 0; i < n_obs; ++i) {
        obs(i, 0) = features[i].normalized_size;
        obs(i, 1) = features[i].price_change;
        obs(i, 2) = features[i].time_delta;
        obs(i, 3) = features[i].spread_ratio;
        obs(i, 4) = features[i].side_imbalance;
        obs(i, 5) = features[i].volume_rate;
    }
    
    // Initialize model if needed
    if (model.states.empty()) {
        model.states.resize(n_states);
        model.transition_matrix = Eigen::MatrixXd::Constant(n_states, n_states, 1.0/n_states);
        
        // K-means initialization (simplified)
        for (size_t s = 0; s < n_states; ++s) {
            model.states[s].mean = obs.row(s * n_obs / n_states).transpose();
            model.states[s].covariance = Eigen::MatrixXd::Identity(n_features, n_features);
            model.states[s].prior = 1.0 / n_states;
        }
    }
    
    // Simplified EM algorithm (full implementation would be more complex)
    for (size_t iter = 0; iter < config_.max_iterations; ++iter) {
        // E-step: compute responsibilities
        Eigen::MatrixXd gamma(n_obs, n_states);
        
        for (size_t t = 0; t < n_obs; ++t) {
            Eigen::VectorXd obs_t = obs.row(t).transpose();
            
            for (size_t s = 0; s < n_states; ++s) {
                // Compute Gaussian probability (simplified)
                Eigen::VectorXd diff = obs_t - model.states[s].mean;
                double prob = std::exp(-0.5 * diff.transpose() * diff);
                gamma(t, s) = prob * model.states[s].prior;
            }
            
            // Normalize
            double sum = gamma.row(t).sum();
            if (sum > 0) {
                gamma.row(t) /= sum;
            }
        }
        
        // M-step: update parameters
        for (size_t s = 0; s < n_states; ++s) {
            double gamma_sum = gamma.col(s).sum();
            
            if (gamma_sum > 0) {
                // Update mean
                model.states[s].mean = (gamma.col(s).transpose() * obs).transpose() / gamma_sum;
                
                // Update prior
                model.states[s].prior = gamma_sum / n_obs;
            }
        }
        
        // Check convergence (simplified)
        // In production, would check log-likelihood change
    }
    
    model.is_trained = true;
}

std::vector<size_t> TradeClustering::viterbi_decode(
    const HMMModel& model, const std::vector<TradeFeatures>& features) const {
    
    if (!model.is_trained || features.empty()) {
        return {};
    }
    
    const size_t n_states = model.states.size();
    const size_t n_obs = features.size();
    
    // Simplified Viterbi (full implementation would use log probabilities)
    std::vector<std::vector<double>> viterbi(n_obs, std::vector<double>(n_states, 0.0));
    std::vector<std::vector<size_t>> path(n_obs, std::vector<size_t>(n_states, 0));
    
    // Initialization
    for (size_t s = 0; s < n_states; ++s) {
        viterbi[0][s] = model.states[s].prior;
    }
    
    // Forward pass
    for (size_t t = 1; t < n_obs; ++t) {
        for (size_t s = 0; s < n_states; ++s) {
            double max_prob = 0.0;
            size_t max_state = 0;
            
            for (size_t prev_s = 0; prev_s < n_states; ++prev_s) {
                double trans_prob = 1.0 / n_states; // Simplified
                double prob = viterbi[t-1][prev_s] * trans_prob;
                
                if (prob > max_prob) {
                    max_prob = prob;
                    max_state = prev_s;
                }
            }
            
            viterbi[t][s] = max_prob;
            path[t][s] = max_state;
        }
    }
    
    // Backward pass
    std::vector<size_t> states(n_obs);
    
    // Find best final state
    size_t best_last_state = 0;
    double best_prob = viterbi[n_obs-1][0];
    for (size_t s = 1; s < n_states; ++s) {
        if (viterbi[n_obs-1][s] > best_prob) {
            best_prob = viterbi[n_obs-1][s];
            best_last_state = s;
        }
    }
    
    states[n_obs-1] = best_last_state;
    
    // Backtrack
    for (int t = n_obs - 2; t >= 0; --t) {
        states[t] = path[t+1][states[t+1]];
    }
    
    return states;
}

TradeClustering::RegimeInfo::Regime TradeClustering::detect_regime(
    const std::vector<size_t>& clusters,
    const std::vector<ClusterInfo>& cluster_info) const {
    
    // Count transitions
    size_t transitions = 0;
    for (size_t i = 1; i < clusters.size(); ++i) {
        if (clusters[i] != clusters[i-1]) {
            transitions++;
        }
    }
    
    double transition_rate = static_cast<double>(transitions) / clusters.size();
    
    // Check for dark pool activity
    for (const auto& info : cluster_info) {
        if (info.likely_dark_pool && info.probability > config_.dark_pool_signature) {
            return RegimeInfo::Regime::DARK_POOL_ACTIVE;
        }
    }
    
    // High transition rate indicates aggressive trading
    if (transition_rate > 0.3) {
        // Determine direction based on dominant cluster
        // This is simplified - real implementation would analyze features
        return RegimeInfo::Regime::AGGRESSIVE_BUYING;
    }
    
    return RegimeInfo::Regime::NORMAL;
}

bool TradeClustering::has_dark_pool_signature(const ClusterInfo& cluster) const {
    // Dark pool characteristics:
    // 1. Large average trade size
    // 2. Low time clustering (not bursty)
    // 3. Low price volatility
    
    // Simplified detection
    return cluster.avg_size > 10000 && cluster.time_clustering < 0.3;
}

void TradeClustering::update_statistics(SymbolData& data, const Trade& trade) {
    const double alpha = 0.05; // Exponential decay factor
    
    // Update average trade size
    data.avg_trade_size = (1 - alpha) * data.avg_trade_size + alpha * trade.quantity;
    
    // Update average time between trades
    if (!data.trades.empty() && data.trades.size() > 1) {
        const auto& prev = data.trades[data.trades.size() - 2];
        double time_delta = (trade.timestamp - prev.timestamp) / 1000000000.0;
        data.avg_time_between = (1 - alpha) * data.avg_time_between + alpha * time_delta;
        
        // Update price volatility
        double price_change = std::abs(trade.price - prev.price) / static_cast<double>(prev.price);
        data.price_volatility = (1 - alpha) * data.price_volatility + alpha * price_change;
    }
}

} 
