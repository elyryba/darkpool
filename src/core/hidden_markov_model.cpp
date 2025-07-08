#include "darkpool/core/hidden_markov_model.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace darkpool {

HiddenMarkovModel::HiddenMarkovModel(const Config& config) : config_(config) {
    symbol_data_.reserve(1000);
}

void HiddenMarkovModel::on_trade(const Trade& trade) {
    std::unique_lock lock(data_mutex_);
    auto& data = symbol_data_[trade.symbol];
    
    data.trades.push_back(trade);
    while (data.trades.size() > config_.observation_window) {
        data.trades.pop_front();
    }
    
    // Update running statistics
    const double alpha = 0.05;
    data.avg_volume = (1 - alpha) * data.avg_volume + alpha * trade.quantity;
    data.avg_trade_size = (1 - alpha) * data.avg_trade_size + alpha * trade.quantity;
    
    // Extract observation if we have enough data
    if (data.trades.size() >= 10 && data.quotes.size() >= 10) {
        auto obs = extract_observation(data);
        data.observations.push_back(obs);
        
        while (data.observations.size() > config_.observation_window) {
            data.observations.erase(data.observations.begin());
        }
        
        // Retrain model periodically
        if (data.observations.size() >= config_.min_observations &&
            data.observations.size() % 50 == 0) {
            train_model(data.model, data.observations);
            
            // Update current regime
            if (data.model.is_trained) {
                auto states = viterbi(data.model, data.observations);
                auto regime = classify_regime(states, data.observations);
                
                if (regime != data.current_regime.regime) {
                    data.current_regime.regime = regime;
                    data.current_regime.last_transition = trade.timestamp;
                    data.current_regime.consecutive_states = 1;
                    stats_.regime_changes++;
                } else {
                    data.current_regime.consecutive_states++;
                }
                
                // Calculate regime probability
                if (!states.empty()) {
                    size_t current_state = states.back();
                    auto fb_result = forward_backward(data.model, data.observations);
                    data.current_regime.probability = 
                        fb_result.gamma(data.observations.size() - 1, current_state);
                }
            }
        }
    }
    
    stats_.total_observations++;
}

void HiddenMarkovModel::on_quote(const Quote& quote) {
    std::unique_lock lock(data_mutex_);
    auto& data = symbol_data_[quote.symbol];
    
    data.quotes.push_back(quote);
    while (data.quotes.size() > config_.observation_window) {
        data.quotes.pop_front();
    }
}

void HiddenMarkovModel::on_order_book(const OrderBookSnapshot& book) {
    std::unique_lock lock(data_mutex_);
    auto& data = symbol_data_[book.symbol];
    
    data.books.push_back(book);
    while (data.books.size() > config_.observation_window) {
        data.books.pop_front();
    }
}

HiddenMarkovModel::RegimeState HiddenMarkovModel::get_current_regime(Symbol symbol) const {
    std::shared_lock lock(data_mutex_);
    
    auto it = symbol_data_.find(symbol);
    if (it == symbol_data_.end()) {
        return RegimeState{MarketRegime::NORMAL, 0.5, {}, 0, 0};
    }
    
    return it->second.current_regime;
}

std::optional<Anomaly> HiddenMarkovModel::check_anomaly(Symbol symbol) const {
    auto regime = get_current_regime(symbol);
    
    // Check for dark pool regimes
    if (regime.regime == MarketRegime::DARK_POOL_BUY || 
        regime.regime == MarketRegime::DARK_POOL_SELL) {
        
        if (regime.probability > config_.regime_stability_threshold &&
            regime.consecutive_states > 5) {
            
            Anomaly anomaly;
            anomaly.symbol = symbol;
            anomaly.timestamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            anomaly.type = AnomalyType::HIDDEN_REFILL;
            anomaly.confidence = regime.probability;
            anomaly.magnitude = regime.consecutive_states / 10.0;
            
            // Estimate hidden size based on regime duration and volume
            std::shared_lock lock(data_mutex_);
            auto it = symbol_data_.find(symbol);
            if (it != symbol_data_.end()) {
                const auto& data = it->second;
                anomaly.estimated_hidden_size = static_cast<Quantity>(
                    data.avg_volume * regime.consecutive_states * 2.0
                );
            }
            
            snprintf(anomaly.description.data(), anomaly.description.size(),
                    "Dark pool %s regime detected: probability=%.2f, duration=%zu periods",
                    regime.regime == MarketRegime::DARK_POOL_BUY ? "buying" : "selling",
                    regime.probability, regime.consecutive_states);
            
            stats_.dark_pool_detections++;
            return anomaly;
        }
    }
    
    // Check for sudden regime changes
    if (regime.consecutive_states == 1 && regime.probability > 0.9) {
        bool is_aggressive = (regime.regime == MarketRegime::BUYING_PRESSURE ||
                             regime.regime == MarketRegime::SELLING_PRESSURE);
        
        if (is_aggressive) {
            Anomaly anomaly;
            anomaly.symbol = symbol;
            anomaly.timestamp = regime.last_transition;
            anomaly.type = AnomalyType::TRADE_CLUSTERING;
            anomaly.confidence = regime.probability * 0.8;
            anomaly.magnitude = 1.0;
            
            snprintf(anomaly.description.data(), anomaly.description.size(),
                    "Sudden regime change to %s pressure detected",
                    regime.regime == MarketRegime::BUYING_PRESSURE ? "buying" : "selling");
            
            return anomaly;
        }
    }
    
    return std::nullopt;
}

std::vector<size_t> HiddenMarkovModel::decode_states(Symbol symbol, size_t lookback) const {
    std::shared_lock lock(data_mutex_);
    
    auto it = symbol_data_.find(symbol);
    if (it == symbol_data_.end() || !it->second.model.is_trained) {
        return {};
    }
    
    const auto& data = it->second;
    size_t start = data.observations.size() > lookback ? 
                   data.observations.size() - lookback : 0;
    
    std::vector<Observation> recent_obs(
        data.observations.begin() + start, data.observations.end()
    );
    
    return viterbi(data.model, recent_obs);
}

Eigen::MatrixXd HiddenMarkovModel::get_transition_matrix(Symbol symbol) const {
    std::shared_lock lock(data_mutex_);
    
    auto it = symbol_data_.find(symbol);
    if (it == symbol_data_.end() || !it->second.model.is_trained) {
        return create_asymmetric_transitions();
    }
    
    return it->second.model.transition_matrix;
}

HiddenMarkovModel::Observation HiddenMarkovModel::extract_observation(
    const SymbolData& data) const {
    
    Observation obs;
    obs.timestamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    
    // Volume features
    if (!data.trades.empty()) {
        double recent_volume = 0.0;
        for (const auto& trade : data.trades) {
            recent_volume += trade.quantity;
        }
        obs.normalized_volume = data.avg_volume > 0 ? 
                               recent_volume / data.trades.size() / data.avg_volume : 1.0;
        
        // Price change
        obs.price_change = data.trades.size() > 1 ?
            (data.trades.back().price - data.trades.front().price) / 
             static_cast<double>(data.trades.front().price) : 0.0;
        
        // Trade intensity
        Timestamp duration = data.trades.back().timestamp - data.trades.front().timestamp;
        obs.trade_intensity = duration > 0 ? 
            data.trades.size() / (duration / 1000000000.0) : 1.0;
        
        // Average trade size
        obs.avg_trade_size = recent_volume / data.trades.size();
        
        // Net aggressor
        int buy_count = 0, sell_count = 0;
        for (const auto& trade : data.trades) {
            if (trade.aggressor_side == Side::BUY) buy_count++;
            else sell_count++;
        }
        obs.net_aggressor = buy_count > sell_count ? Side::BUY : Side::SELL;
    }
    
    // Order book features
    if (!data.books.empty()) {
        const auto& book = data.books.back();
        double bid_volume = 0.0, ask_volume = 0.0;
        
        for (size_t i = 0; i < 5 && i < book.bids.size(); ++i) {
            bid_volume += book.bids[i].quantity;
            ask_volume += book.asks[i].quantity;
        }
        
        obs.bid_ask_imbalance = (bid_volume + ask_volume) > 0 ?
            (bid_volume - ask_volume) / (bid_volume + ask_volume) : 0.0;
    }
    
    // Quote intensity
    if (!data.quotes.empty() && data.quotes.size() > 1) {
        Timestamp quote_duration = data.quotes.back().timestamp - 
                                  data.quotes.front().timestamp;
        obs.quote_intensity = quote_duration > 0 ?
            data.quotes.size() / (quote_duration / 1000000000.0) : 1.0;
    }
    
    return obs;
}

void HiddenMarkovModel::train_model(ModelParameters& model, 
                                   const std::vector<Observation>& obs) const {
    if (obs.size() < config_.min_observations) return;
    
    const size_t n_states = config_.num_states;
    const size_t n_features = 6; // Number of observation features
    
    // Initialize model if needed
    if (!model.is_trained) {
        model.transition_matrix = create_asymmetric_transitions();
        model.initial_probs = Eigen::VectorXd::Constant(n_states, 1.0 / n_states);
        
        model.emission_means.resize(n_states);
        model.emission_covariances.resize(n_states);
        
        // Initialize with k-means clustering (simplified)
        for (size_t s = 0; s < n_states; ++s) {
            model.emission_means[s] = Eigen::VectorXd::Zero(n_features);
            model.emission_covariances[s] = Eigen::MatrixXd::Identity(n_features, n_features);
            
            // Set different means for different states
            model.emission_means[s](0) = 0.5 + s * 0.5; // Volume
            model.emission_means[s](1) = (s % 2 == 0) ? 0.01 : -0.01; // Price change
        }
    }
    
    // Baum-Welch algorithm
    double prev_likelihood = -std::numeric_limits<double>::infinity();
    
    for (size_t iter = 0; iter < config_.max_iterations; ++iter) {
        auto fb_result = forward_backward(model, obs);
        
        // Check convergence
        if (std::abs(fb_result.log_likelihood - prev_likelihood) < config_.convergence_threshold) {
            break;
        }
        prev_likelihood = fb_result.log_likelihood;
        
        // M-step: Update parameters
        
        // Update initial probabilities
        model.initial_probs = fb_result.gamma.row(0).transpose();
        
        // Update transition matrix (maintain asymmetry)
        for (size_t i = 0; i < n_states; ++i) {
            for (size_t j = 0; j < n_states; ++j) {
                double numerator = 0.0;
                double denominator = 0.0;
                
                for (size_t t = 0; t < obs.size() - 1; ++t) {
                    numerator += fb_result.xi(t * n_states + i, j);
                    denominator += fb_result.gamma(t, i);
                }
                
                if (denominator > 0) {
                    model.transition_matrix(i, j) = numerator / denominator;
                }
            }
        }
        
        // Maintain asymmetry
        if (config_.transition_asymmetry > 0) {
            // Buying states transition more easily to buying
            model.transition_matrix(1, 3) *= (1 + config_.transition_asymmetry);
            model.transition_matrix(3, 1) *= (1 + config_.transition_asymmetry);
            
            // Selling states transition more easily to selling
            model.transition_matrix(2, 4) *= (1 + config_.transition_asymmetry);
            model.transition_matrix(4, 2) *= (1 + config_.transition_asymmetry);
            
            // Normalize rows
            for (size_t i = 0; i < n_states; ++i) {
                double row_sum = model.transition_matrix.row(i).sum();
                if (row_sum > 0) {
                    model.transition_matrix.row(i) /= row_sum;
                }
            }
        }
        
        // Update emission parameters
        for (size_t s = 0; s < n_states; ++s) {
            Eigen::VectorXd mean = Eigen::VectorXd::Zero(n_features);
            double gamma_sum = 0.0;
            
            for (size_t t = 0; t < obs.size(); ++t) {
                Eigen::VectorXd obs_vec(n_features);
                obs_vec << obs[t].normalized_volume,
                          obs[t].price_change,
                          obs[t].bid_ask_imbalance,
                          obs[t].trade_intensity,
                          obs[t].avg_trade_size / 1000.0,
                          obs[t].quote_intensity;
                
                mean += fb_result.gamma(t, s) * obs_vec;
                gamma_sum += fb_result.gamma(t, s);
            }
            
            if (gamma_sum > 0) {
                model.emission_means[s] = mean / gamma_sum;
            }
        }
    }
    
    model.is_trained = true;
}

HiddenMarkovModel::ForwardBackwardResult HiddenMarkovModel::forward_backward(
    const ModelParameters& model, const std::vector<Observation>& obs) const {
    
    const size_t T = obs.size();
    const size_t n_states = config_.num_states;
    
    ForwardBackwardResult result;
    result.alpha = Eigen::MatrixXd::Zero(T, n_states);
    result.beta = Eigen::MatrixXd::Zero(T, n_states);
    result.gamma = Eigen::MatrixXd::Zero(T, n_states);
    result.xi = Eigen::MatrixXd::Zero(T * n_states, n_states);
    
    // Forward pass
    for (size_t s = 0; s < n_states; ++s) {
        Eigen::VectorXd obs_vec(6);
        obs_vec << obs[0].normalized_volume, obs[0].price_change,
                   obs[0].bid_ask_imbalance, obs[0].trade_intensity,
                   obs[0].avg_trade_size / 1000.0, obs[0].quote_intensity;
        
        result.alpha(0, s) = model.initial_probs(s) * 
            observation_probability(obs[0], model.emission_means[s], 
                                  model.emission_covariances[s]);
    }
    
    // Normalize alpha
    double scale = result.alpha.row(0).sum();
    if (scale > 0) result.alpha.row(0) /= scale;
    result.log_likelihood = std::log(scale);
    
    for (size_t t = 1; t < T; ++t) {
        for (size_t j = 0; j < n_states; ++j) {
            double sum = 0.0;
            for (size_t i = 0; i < n_states; ++i) {
                sum += result.alpha(t-1, i) * model.transition_matrix(i, j);
            }
            
            result.alpha(t, j) = sum * observation_probability(
                obs[t], model.emission_means[j], model.emission_covariances[j]
            );
        }
        
        scale = result.alpha.row(t).sum();
        if (scale > 0) {
            result.alpha.row(t) /= scale;
            result.log_likelihood += std::log(scale);
        }
    }
    
    // Backward pass
    result.beta.row(T-1).setOnes();
    
    for (int t = T - 2; t >= 0; --t) {
        for (size_t i = 0; i < n_states; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < n_states; ++j) {
                sum += model.transition_matrix(i, j) * 
                       observation_probability(obs[t+1], model.emission_means[j],
                                             model.emission_covariances[j]) *
                       result.beta(t+1, j);
            }
            result.beta(t, i) = sum;
        }
        
        // Normalize beta
        scale = result.beta.row(t).sum();
        if (scale > 0) result.beta.row(t) /= scale;
    }
    
    // Calculate gamma and xi
    for (size_t t = 0; t < T; ++t) {
        double sum = 0.0;
        for (size_t i = 0; i < n_states; ++i) {
            result.gamma(t, i) = result.alpha(t, i) * result.beta(t, i);
            sum += result.gamma(t, i);
        }
        if (sum > 0) result.gamma.row(t) /= sum;
        
        if (t < T - 1) {
            for (size_t i = 0; i < n_states; ++i) {
                for (size_t j = 0; j < n_states; ++j) {
                    result.xi(t * n_states + i, j) = 
                        result.alpha(t, i) * model.transition_matrix(i, j) *
                        observation_probability(obs[t+1], model.emission_means[j],
                                              model.emission_covariances[j]) *
                        result.beta(t+1, j);
                }
            }
        }
    }
    
    return result;
}

std::vector<size_t> HiddenMarkovModel::viterbi(const ModelParameters& model,
                                              const std::vector<Observation>& obs) const {
    if (obs.empty() || !model.is_trained) return {};
    
    const size_t T = obs.size();
    const size_t n_states = config_.num_states;
    
    // Log probabilities to avoid underflow
    Eigen::MatrixXd delta(T, n_states);
    Eigen::MatrixXi psi(T, n_states);
    
    // Initialization
    for (size_t s = 0; s < n_states; ++s) {
        delta(0, s) = std::log(model.initial_probs(s)) + 
            std::log(observation_probability(obs[0], model.emission_means[s],
                                           model.emission_covariances[s]));
    }
    
    // Recursion
    for (size_t t = 1; t < T; ++t) {
        for (size_t j = 0; j < n_states; ++j) {
            double max_val = -std::numeric_limits<double>::infinity();
            size_t max_state = 0;
            
            for (size_t i = 0; i < n_states; ++i) {
                double val = delta(t-1, i) + std::log(model.transition_matrix(i, j));
                if (val > max_val) {
                    max_val = val;
                    max_state = i;
                }
            }
            
            delta(t, j) = max_val + std::log(observation_probability(
                obs[t], model.emission_means[j], model.emission_covariances[j]
            ));
            psi(t, j) = max_state;
        }
    }
    
    // Backtrack
    std::vector<size_t> states(T);
    
    // Find best final state
    double max_val = delta.row(T-1).maxCoeff(&states[T-1]);
    
    for (int t = T - 2; t >= 0; --t) {
        states[t] = psi(t+1, states[t+1]);
    }
    
    return states;
}

HiddenMarkovModel::MarketRegime HiddenMarkovModel::classify_regime(
    const std::vector<size_t>& states, const std::vector<Observation>& obs) const {
    
    if (states.empty() || obs.empty()) return MarketRegime::NORMAL;
    
    // Count state occurrences in recent window
    size_t window = std::min(size_t(20), states.size());
    std::vector<int> state_counts(config_.num_states, 0);
    
    for (size_t i = states.size() - window; i < states.size(); ++i) {
        state_counts[states[i]]++;
    }
    
    // Find dominant state
    size_t dominant_state = std::distance(state_counts.begin(),
        std::max_element(state_counts.begin(), state_counts.end()));
    
    // Analyze recent observations
    double avg_volume = 0.0;
    double avg_price_change = 0.0;
    double avg_imbalance = 0.0;
    int buy_bias = 0;
    
    for (size_t i = obs.size() - window; i < obs.size(); ++i) {
        avg_volume += obs[i].normalized_volume;
        avg_price_change += obs[i].price_change;
        avg_imbalance += obs[i].bid_ask_imbalance;
        if (obs[i].net_aggressor == Side::BUY) buy_bias++;
        else buy_bias--;
    }
    
    avg_volume /= window;
    avg_price_change /= window;
    avg_imbalance /= window;
    
    // Map to regime based on state and features
    // States: 0=Normal, 1=Accumulation, 2=Distribution, 3=BuyPressure, 4=SellPressure
    
    // Dark pool detection: low volatility, consistent direction, moderate volume
    bool low_volatility = std::abs(avg_price_change) < 0.001;
    bool consistent_direction = std::abs(buy_bias) > window * 0.7;
    bool moderate_volume = avg_volume > 0.8 && avg_volume < 2.0;
    
    if (low_volatility && consistent_direction && moderate_volume) {
        return buy_bias > 0 ? MarketRegime::DARK_POOL_BUY : MarketRegime::DARK_POOL_SELL;
    }
    
    // High pressure regimes
    if (dominant_state == 3 || (avg_imbalance > 0.5 && buy_bias > window * 0.6)) {
        return MarketRegime::BUYING_PRESSURE;
    }
    if (dominant_state == 4 || (avg_imbalance < -0.5 && buy_bias < -window * 0.6)) {
        return MarketRegime::SELLING_PRESSURE;
    }
    
    // Accumulation/Distribution
    if (dominant_state == 1 || (avg_volume > 1.2 && buy_bias > 0)) {
        return MarketRegime::ACCUMULATION;
    }
    if (dominant_state == 2 || (avg_volume > 1.2 && buy_bias < 0)) {
        return MarketRegime::DISTRIBUTION;
    }
    
    return MarketRegime::NORMAL;
}

Eigen::MatrixXd HiddenMarkovModel::create_asymmetric_transitions() const {
    const size_t n = config_.num_states;
    Eigen::MatrixXd trans = Eigen::MatrixXd::Constant(n, n, 0.1);
    
    // Diagonal dominance (states persist)
    for (size_t i = 0; i < n; ++i) {
        trans(i, i) = 0.6;
    }
    
    // Asymmetric transitions
    // Buying states (1, 3) transition more easily to each other
    trans(1, 3) = 0.15 * (1 + config_.transition_asymmetry);
    trans(3, 1) = 0.15 * (1 + config_.transition_asymmetry);
    
    // Selling states (2, 4) transition more easily to each other
    trans(2, 4) = 0.15 * (1 + config_.transition_asymmetry);
    trans(4, 2) = 0.15 * (1 + config_.transition_asymmetry);
    
    // Normal state (0) has symmetric transitions
    trans(0, 1) = trans(0, 2) = 0.1;
    trans(1, 0) = trans(2, 0) = 0.1;
    
    // Normalize rows
    for (size_t i = 0; i < n; ++i) {
        trans.row(i) /= trans.row(i).sum();
    }
    
    return trans;
}

double HiddenMarkovModel::observation_probability(const Observation& obs,
                                                 const Eigen::VectorXd& mean,
                                                 const Eigen::MatrixXd& cov) const {
    Eigen::VectorXd obs_vec(6);
    obs_vec << obs.normalized_volume, obs.price_change, obs.bid_ask_imbalance,
               obs.trade_intensity, obs.avg_trade_size / 1000.0, obs.quote_intensity;
    
    // Multivariate normal probability (simplified - full implementation would use Cholesky)
    Eigen::VectorXd diff = obs_vec - mean;
    double exponent = -0.5 * diff.transpose() * cov.inverse() * diff;
    
    // Avoid numerical issues
    exponent = std::max(-50.0, std::min(50.0, exponent));
    
    return std::exp(exponent);
}

} 
