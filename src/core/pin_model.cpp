#include "darkpool/core/pin_model.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace darkpool {

PINModel::PINModel(const Config& config) : config_(config) {
    symbol_data_.reserve(1000);
    init_factorial_cache();
}

void PINModel::on_trade(const Trade& trade) {
    std::unique_lock lock(data_mutex_);
    auto& data = symbol_data_[trade.symbol];
    
    update_daily_data(data.current_day, trade);
    
    // Initialize intraday buckets if needed
    if (data.current_day.bucket_buys.empty()) {
        data.current_day.bucket_buys.resize(config_.intraday_buckets, 0);
        data.current_day.bucket_sells.resize(config_.intraday_buckets, 0);
    }
    
    // Update intraday bucket
    if (config_.use_intraday_estimation) {
        size_t bucket = get_time_bucket(trade.timestamp);
        if (bucket < config_.intraday_buckets) {
            if (trade.aggressor_side == Side::BUY) {
                data.current_day.bucket_buys[bucket]++;
            } else if (trade.aggressor_side == Side::SELL) {
                data.current_day.bucket_sells[bucket]++;
            }
        }
    }
}

void PINModel::on_daily_close(Symbol symbol, Timestamp date) {
    std::unique_lock lock(data_mutex_);
    auto& data = symbol_data_[symbol];
    
    if (data.current_day.total_trades >= config_.min_trades_per_day) {
        data.current_day.date = date;
        data.daily_history.push_back(data.current_day);
        
        // Maintain window
        while (data.daily_history.size() > config_.estimation_window) {
            data.daily_history.pop_front();
        }
        
        // Re-estimate parameters
        if (data.daily_history.size() >= 5) {
            data.parameters = estimate_parameters(data.daily_history);
        }
    }
    
    // Reset for new day
    data.current_day = DailyData{};
}

PINModel::PINResult PINModel::calculate_pin(Symbol symbol) const {
    std::shared_lock lock(data_mutex_);
    
    auto it = symbol_data_.find(symbol);
    if (it == symbol_data_.end()) {
        return PINResult{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0};
    }
    
    const auto& data = it->second;
    
    // Check cache
    Timestamp now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    if (data.last_calculation > 0 && (now - data.last_calculation) < 60000000000) { // 1 minute
        return data.last_result;
    }
    
    PINResult result;
    result.calculation_time = now;
    
    if (!data.parameters.is_estimated) {
        // Use default parameters
        result.alpha = 0.2;
        result.delta = 0.5;
        result.mu = 100.0;
        result.epsilon_buy = 50.0;
        result.epsilon_sell = 50.0;
        result.pin = result.alpha * result.mu / 
                     (result.alpha * result.mu + result.epsilon_buy + result.epsilon_sell);
    } else {
        result.alpha = data.parameters.alpha;
        result.delta = data.parameters.delta;
        result.mu = data.parameters.mu;
        result.epsilon_buy = data.parameters.epsilon_b;
        result.epsilon_sell = data.parameters.epsilon_s;
        
        // PIN = α × μ / (α × μ + εb + εs)
        result.pin = result.alpha * result.mu / 
                     (result.alpha * result.mu + result.epsilon_buy + result.epsilon_sell);
    }
    
    result.estimation_days = data.daily_history.size();
    
    // Calculate log-likelihood for model fit
    if (!data.daily_history.empty()) {
        result.log_likelihood = 0.0;
        for (const auto& day : data.daily_history) {
            double day_ll = std::log(
                (1 - result.alpha) * calculate_likelihood(day.buy_orders, day.sell_orders, 
                                                         data.parameters, 0) +
                result.alpha * result.delta * calculate_likelihood(day.buy_orders, day.sell_orders,
                                                                  data.parameters, 2) +
                result.alpha * (1 - result.delta) * calculate_likelihood(day.buy_orders, day.sell_orders,
                                                                        data.parameters, 1)
            );
            result.log_likelihood += day_ll;
        }
    }
    
    // Update cache
    const_cast<SymbolData&>(data).last_result = result;
    const_cast<SymbolData&>(data).last_calculation = now;
    
    return result;
}

PINModel::IntradayPIN PINModel::calculate_intraday_pin(Symbol symbol) const {
    std::shared_lock lock(data_mutex_);
    
    IntradayPIN result;
    result.hourly_pin.resize(config_.intraday_buckets, 0.0);
    
    auto it = symbol_data_.find(symbol);
    if (it == symbol_data_.end()) {
        return result;
    }
    
    const auto& data = it->second;
    
    // Get current bucket
    Timestamp now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    result.current_bucket = get_time_bucket(now);
    
    // Calculate PIN for each bucket using recent history
    if (!data.daily_history.empty()) {
        for (size_t bucket = 0; bucket < config_.intraday_buckets; ++bucket) {
            size_t total_buys = 0;
            size_t total_sells = 0;
            size_t days = 0;
            
            // Aggregate bucket data across days
            for (const auto& day : data.daily_history) {
                if (bucket < day.bucket_buys.size()) {
                    total_buys += day.bucket_buys[bucket];
                    total_sells += day.bucket_sells[bucket];
                    days++;
                }
            }
            
            if (days > 0 && (total_buys + total_sells) > 10) {
                // Simplified PIN calculation for bucket
                double avg_buys = static_cast<double>(total_buys) / days;
                double avg_sells = static_cast<double>(total_sells) / days;
                double imbalance = std::abs(avg_buys - avg_sells) / (avg_buys + avg_sells);
                
                // Scale imbalance to PIN-like value
                result.hourly_pin[bucket] = imbalance * 0.5; // Simplified
            }
        }
        
        // Current PIN
        if (result.current_bucket < result.hourly_pin.size()) {
            result.current_pin = result.hourly_pin[result.current_bucket];
        }
        
        // Daily average
        double sum = std::accumulate(result.hourly_pin.begin(), result.hourly_pin.end(), 0.0);
        result.daily_average = sum / result.hourly_pin.size();
    }
    
    return result;
}

std::optional<Anomaly> PINModel::check_anomaly(Symbol symbol) const {
    auto pin_result = calculate_pin(symbol);
    
    if (pin_result.estimation_days < 5) {
        return std::nullopt;
    }
    
    // Check for high PIN
    if (pin_result.pin > config_.informed_threshold) {
        Anomaly anomaly;
        anomaly.symbol = symbol;
        anomaly.timestamp = pin_result.calculation_time;
        anomaly.type = AnomalyType::MULTI_VENUE_SWEEP; // Using as proxy for informed trading
        anomaly.confidence = std::min(1.0, pin_result.pin / 0.4); // Scale confidence
        anomaly.magnitude = pin_result.pin / config_.informed_threshold;
        
        // Estimate hidden size based on informed flow
        double informed_flow = pin_result.alpha * pin_result.mu;
        anomaly.estimated_hidden_size = static_cast<Quantity>(informed_flow * 100);
        
        snprintf(anomaly.description.data(), anomaly.description.size(),
                "High PIN detected: %.3f (threshold: %.3f), informed flow: %.1f",
                pin_result.pin, config_.informed_threshold, informed_flow);
        
        return anomaly;
    }
    
    // Check for sudden PIN increase using intraday
    if (config_.use_intraday_estimation) {
        auto intraday = calculate_intraday_pin(symbol);
        
        if (intraday.current_pin > intraday.daily_average * 2.0 &&
            intraday.current_pin > 0.2) {
            
            Anomaly anomaly;
            anomaly.symbol = symbol;
            anomaly.timestamp = pin_result.calculation_time;
            anomaly.type = AnomalyType::MULTI_VENUE_SWEEP;
            anomaly.confidence = 0.7;
            anomaly.magnitude = intraday.current_pin / intraday.daily_average;
            
            snprintf(anomaly.description.data(), anomaly.description.size(),
                    "Intraday PIN spike: current=%.3f, average=%.3f",
                    intraday.current_pin, intraday.daily_average);
            
            return anomaly;
        }
    }
    
    return std::nullopt;
}

std::vector<std::pair<Timestamp, double>> PINModel::get_pin_history(
    Symbol symbol, size_t days) const {
    
    std::shared_lock lock(data_mutex_);
    
    std::vector<std::pair<Timestamp, double>> history;
    
    auto it = symbol_data_.find(symbol);
    if (it == symbol_data_.end() || it->second.daily_history.empty()) {
        return history;
    }
    
    const auto& data = it->second;
    size_t start = data.daily_history.size() > days ? 
                   data.daily_history.size() - days : 0;
    
    // Calculate PIN for each historical day
    for (size_t i = start; i < data.daily_history.size(); ++i) {
        // Estimate parameters using data up to this day
        std::deque<DailyData> subset(data.daily_history.begin(), 
                                     data.daily_history.begin() + i + 1);
        
        if (subset.size() >= 5) {
            auto params = estimate_parameters(subset);
            double pin = params.alpha * params.mu / 
                        (params.alpha * params.mu + params.epsilon_b + params.epsilon_s);
            
            history.emplace_back(data.daily_history[i].date, pin);
        }
    }
    
    return history;
}

PINModel::ModelParameters PINModel::estimate_parameters(
    const std::deque<DailyData>& data) const {
    
    if (data.size() < 5) {
        return ModelParameters{};
    }
    
    ModelParameters params;
    params.alpha = 0.2;
    params.delta = 0.5;
    params.mu = 100.0;
    params.epsilon_b = 50.0;
    params.epsilon_s = 50.0;
    
    double prev_likelihood = -std::numeric_limits<double>::infinity();
    
    // EM algorithm
    for (size_t iter = 0; iter < config_.max_iterations; ++iter) {
        // E-step: Calculate posterior probabilities
        std::vector<PosteriorProbs> posteriors;
        double log_likelihood = 0.0;
        
        for (const auto& day : data) {
            auto post = calculate_posteriors(day, params);
            posteriors.push_back(post);
            
            // Update log-likelihood
            double day_ll = (1 - params.alpha) * calculate_likelihood(
                               day.buy_orders, day.sell_orders, params, 0) +
                            params.alpha * params.delta * calculate_likelihood(
                               day.buy_orders, day.sell_orders, params, 2) +
                            params.alpha * (1 - params.delta) * calculate_likelihood(
                               day.buy_orders, day.sell_orders, params, 1);
            
            log_likelihood += std::log(day_ll);
        }
        
        // Check convergence
        if (std::abs(log_likelihood - prev_likelihood) < config_.convergence_threshold) {
            break;
        }
        prev_likelihood = log_likelihood;
        
        // M-step: Update parameters
        double sum_no_event = 0.0;
        double sum_good_event = 0.0;
        double sum_bad_event = 0.0;
        double sum_good_buys = 0.0;
        double sum_bad_sells = 0.0;
        double sum_uninformed_buys = 0.0;
        double sum_uninformed_sells = 0.0;
        
        for (size_t i = 0; i < data.size(); ++i) {
            const auto& day = data[i];
            const auto& post = posteriors[i];
            
            sum_no_event += post.no_event;
            sum_good_event += post.good_event;
            sum_bad_event += post.bad_event;
            
            sum_good_buys += post.good_event * day.buy_orders;
            sum_bad_sells += post.bad_event * day.sell_orders;
            
            sum_uninformed_buys += post.no_event * day.buy_orders + 
                                  post.bad_event * day.buy_orders;
            sum_uninformed_sells += post.no_event * day.sell_orders + 
                                   post.good_event * day.sell_orders;
        }
        
        // Update parameters
        params.alpha = (sum_good_event + sum_bad_event) / data.size();
        params.delta = sum_bad_event / (sum_good_event + sum_bad_event);
        
        if (sum_good_event > 0) {
            params.mu = sum_good_buys / sum_good_event;
        }
        if (sum_bad_event > 0) {
            params.mu = (params.mu + sum_bad_sells / sum_bad_event) / 2.0;
        }
        
        double uninformed_days = sum_no_event + sum_bad_event + sum_good_event;
        if (uninformed_days > 0) {
            params.epsilon_b = sum_uninformed_buys / uninformed_days;
            params.epsilon_s = sum_uninformed_sells / uninformed_days;
        }
        
        // Ensure reasonable bounds
        params.alpha = std::max(0.01, std::min(0.99, params.alpha));
        params.delta = std::max(0.01, std::min(0.99, params.delta));
        params.mu = std::max(1.0, params.mu);
        params.epsilon_b = std::max(1.0, params.epsilon_b);
        params.epsilon_s = std::max(1.0, params.epsilon_s);
    }
    
    params.is_estimated = true;
    return params;
}

PINModel::PosteriorProbs PINModel::calculate_posteriors(
    const DailyData& day, const ModelParameters& params) const {
    
    // Calculate likelihoods for each scenario
    double l_no_event = calculate_likelihood(day.buy_orders, day.sell_orders, params, 0);
    double l_good_event = calculate_likelihood(day.buy_orders, day.sell_orders, params, 1);
    double l_bad_event = calculate_likelihood(day.buy_orders, day.sell_orders, params, 2);
    
    // Prior probabilities
    double p_no_event = 1 - params.alpha;
    double p_good_event = params.alpha * (1 - params.delta);
    double p_bad_event = params.alpha * params.delta;
    
    // Posterior probabilities (Bayes' rule)
    double total = p_no_event * l_no_event + p_good_event * l_good_event + 
                   p_bad_event * l_bad_event;
    
    PosteriorProbs post;
    if (total > 0) {
        post.no_event = p_no_event * l_no_event / total;
        post.good_event = p_good_event * l_good_event / total;
        post.bad_event = p_bad_event * l_bad_event / total;
    } else {
        post.no_event = p_no_event;
        post.good_event = p_good_event;
        post.bad_event = p_bad_event;
    }
    
    return post;
}

double PINModel::calculate_likelihood(size_t buys, size_t sells,
                                     const ModelParameters& params,
                                     int event_type) const {
    double buy_rate, sell_rate;
    
    switch (event_type) {
        case 0:  // No information event
            buy_rate = params.epsilon_b;
            sell_rate = params.epsilon_s;
            break;
        case 1:  // Good news
            buy_rate = params.epsilon_b + params.mu;
            sell_rate = params.epsilon_s;
            break;
        case 2:  // Bad news
            buy_rate = params.epsilon_b;
            sell_rate = params.epsilon_s + params.mu;
            break;
        default:
            return 0.0;
    }
    
    return poisson_prob(buys, buy_rate) * poisson_prob(sells, sell_rate);
}

double PINModel::poisson_prob(size_t k, double lambda) const {
    if (lambda <= 0) return 0.0;
    
    // Use log for numerical stability
    double log_prob = k * std::log(lambda) - lambda;
    
    // Subtract log(k!)
    if (k < log_factorial_cache_.size()) {
        log_prob -= log_factorial_cache_[k];
    } else {
        // Stirling's approximation for large k
        log_prob -= k * std::log(k) - k + 0.5 * std::log(2 * M_PI * k);
    }
    
    return std::exp(log_prob);
}

void PINModel::update_daily_data(DailyData& day, const Trade& trade) const {
    day.total_trades++;
    
    if (trade.aggressor_side == Side::BUY) {
        day.buy_orders++;
        day.buy_volume += trade.quantity;
    } else if (trade.aggressor_side == Side::SELL) {
        day.sell_orders++;
        day.sell_volume += trade.quantity;
    }
}

size_t PINModel::get_time_bucket(Timestamp timestamp) const {
    // Convert nanoseconds to hours of day
    uint64_t seconds = timestamp / 1000000000;
    uint64_t hours = (seconds / 3600) % 24;
    
    // Map to bucket
    return (hours * config_.intraday_buckets) / 24;
}

void PINModel::init_factorial_cache() const {
    log_factorial_cache_.resize(1000);
    log_factorial_cache_[0] = 0.0;
    
    for (size_t i = 1; i < log_factorial_cache_.size(); ++i) {
        log_factorial_cache_[i] = log_factorial_cache_[i-1] + std::log(i);
    }
}

} 
