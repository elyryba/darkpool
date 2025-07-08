#include "darkpool/core/hawkes_process.hpp"
#include <algorithm>
#include <numeric>
#include <random>

namespace darkpool {

HawkesProcess::HawkesProcess(const Config& config) : config_(config) {
    symbol_data_.reserve(1000);
}

void HawkesProcess::on_trade(const Trade& trade) {
    std::unique_lock lock(data_mutex_);
    auto& data = symbol_data_[trade.symbol];
    
    Event event;
    event.timestamp = trade.timestamp;
    event.side = trade.aggressor_side;
    event.magnitude = std::log1p(trade.quantity / 1000.0); // Log-transform size
    event.type = Event::EventType::TRADE;
    
    data.events.push_back(event);
    
    // Maintain window
    while (data.events.size() > config_.max_history) {
        data.events.pop_front();
    }
    
    // Recalibrate periodically
    if (data.events.size() > 50 && data.events.size() % 100 == 0) {
        calibrate_parameters(data);
    }
}

void HawkesProcess::on_order(const Order& order) {
    std::unique_lock lock(data_mutex_);
    auto& data = symbol_data_[order.symbol];
    
    Event event;
    event.timestamp = order.timestamp;
    event.side = order.side;
    event.magnitude = std::log1p(order.quantity / 1000.0);
    event.type = Event::EventType::ORDER;
    
    data.events.push_back(event);
    
    while (data.events.size() > config_.max_history) {
        data.events.pop_front();
    }
}

HawkesProcess::IntensityResult HawkesProcess::calculate_intensity(
    Symbol symbol, Timestamp at_time) const {
    
    std::shared_lock lock(data_mutex_);
    
    auto it = symbol_data_.find(symbol);
    if (it == symbol_data_.end()) {
        return IntensityResult{config_.baseline_intensity, config_.baseline_intensity, 
                              config_.baseline_intensity * 2, 0.5, 0.5, at_time};
    }
    
    const auto& data = it->second;
    
    if (at_time == 0) {
        at_time = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    }
    
    // Use cached value if recent
    if (data.last_calculation > 0 && 
        (at_time - data.last_calculation) < 100000000) { // 100ms cache
        return IntensityResult{
            data.last_buy_intensity,
            data.last_sell_intensity,
            data.last_buy_intensity + data.last_sell_intensity,
            data.last_buy_intensity / (data.last_buy_intensity + data.last_sell_intensity),
            data.last_sell_intensity / (data.last_buy_intensity + data.last_sell_intensity),
            at_time
        };
    }
    
    // Calculate conditional intensity
    double buy_intensity = calculate_conditional_intensity(data.events, at_time, Side::BUY);
    double sell_intensity = calculate_conditional_intensity(data.events, at_time, Side::SELL);
    
    // Update cache
    const_cast<SymbolData&>(data).last_buy_intensity = buy_intensity;
    const_cast<SymbolData&>(data).last_sell_intensity = sell_intensity;
    const_cast<SymbolData&>(data).last_calculation = at_time;
    
    double total = buy_intensity + sell_intensity;
    
    return IntensityResult{
        buy_intensity,
        sell_intensity,
        total,
        buy_intensity / total,
        sell_intensity / total,
        at_time
    };
}

std::optional<Anomaly> HawkesProcess::check_anomaly(Symbol symbol) const {
    auto intensity = calculate_intensity(symbol);
    
    std::shared_lock lock(data_mutex_);
    auto it = symbol_data_.find(symbol);
    if (it == symbol_data_.end() || it->second.events.size() < config_.min_events) {
        return std::nullopt;
    }
    
    const auto& data = it->second;
    
    // Check if intensity exceeds threshold
    double baseline = data.calibrated_baseline > 0 ? 
                     data.calibrated_baseline : config_.baseline_intensity;
    
    double threshold = baseline * config_.intensity_threshold;
    
    if (intensity.total_intensity > threshold) {
        Anomaly anomaly;
        anomaly.symbol = symbol;
        anomaly.timestamp = intensity.calculation_time;
        anomaly.type = AnomalyType::ORDER_BOOK_PRESSURE;
        
        // Calculate branching ratio for confidence
        double branching = calculate_branching_ratio(symbol);
        anomaly.confidence = std::min(1.0, (intensity.total_intensity / threshold - 1.0) * 
                                          (1.0 - branching));
        
        anomaly.magnitude = intensity.total_intensity / baseline;
        
        // Estimate hidden size based on excess intensity
        double excess_rate = intensity.total_intensity - baseline;
        anomaly.estimated_hidden_size = static_cast<Quantity>(excess_rate * 100 * 1000);
        
        // Determine pressure direction
        const char* direction = intensity.buy_pressure > 0.6 ? "buy" : 
                               intensity.sell_pressure > 0.6 ? "sell" : "mixed";
        
        snprintf(anomaly.description.data(), anomaly.description.size(),
                "Hawkes intensity spike: %.2fx baseline, %s pressure (branching=%.2f)",
                anomaly.magnitude, direction, branching);
        
        return anomaly;
    }
    
    return std::nullopt;
}

std::vector<HawkesProcess::IntensityPoint> HawkesProcess::get_intensity_history(
    Symbol symbol, size_t points, Timestamp interval_ns) const {
    
    std::shared_lock lock(data_mutex_);
    
    auto it = symbol_data_.find(symbol);
    if (it == symbol_data_.end() || it->second.events.empty()) {
        return {};
    }
    
    const auto& events = it->second.events;
    
    std::vector<IntensityPoint> history;
    history.reserve(points);
    
    Timestamp end_time = events.back().timestamp;
    Timestamp start_time = end_time - (points * interval_ns);
    
    for (size_t i = 0; i < points; ++i) {
        Timestamp t = start_time + i * interval_ns;
        
        double buy_int = calculate_conditional_intensity(events, t, Side::BUY);
        double sell_int = calculate_conditional_intensity(events, t, Side::SELL);
        double total = buy_int + sell_int;
        
        IntensityPoint point;
        point.timestamp = t;
        point.intensity = total;
        point.dominant_side = buy_int > sell_int ? Side::BUY : Side::SELL;
        
        history.push_back(point);
    }
    
    return history;
}

double HawkesProcess::calculate_branching_ratio(Symbol symbol) const {
    std::shared_lock lock(data_mutex_);
    
    auto it = symbol_data_.find(symbol);
    if (it == symbol_data_.end()) {
        return 0.0;
    }
    
    const auto& data = it->second;
    
    // Branching ratio = alpha / beta
    double alpha = data.calibrated_alpha > 0 ? 
                   data.calibrated_alpha : config_.self_excitation;
    double beta = data.calibrated_beta > 0 ? 
                  data.calibrated_beta : config_.decay_rate;
    
    return alpha / beta;
}

HawkesProcess::IntensityForecast HawkesProcess::forecast_intensity(
    Symbol symbol, Timestamp horizon_ns, size_t steps) const {
    
    auto current = calculate_intensity(symbol);
    
    IntensityForecast forecast;
    forecast.timestamps.reserve(steps);
    forecast.intensities.reserve(steps);
    
    Timestamp current_time = current.calculation_time;
    Timestamp step_size = horizon_ns / steps;
    
    // Get decay parameters
    std::shared_lock lock(data_mutex_);
    auto it = symbol_data_.find(symbol);
    
    double beta = config_.decay_rate;
    double baseline = config_.baseline_intensity;
    
    if (it != symbol_data_.end()) {
        const auto& data = it->second;
        if (data.calibrated_beta > 0) beta = data.calibrated_beta;
        if (data.calibrated_baseline > 0) baseline = data.calibrated_baseline;
    }
    
    // Simple exponential decay forecast
    for (size_t i = 0; i < steps; ++i) {
        Timestamp t = current_time + (i + 1) * step_size;
        double dt = step_size / 1000000000.0; // Convert to seconds
        
        // Intensity decays to baseline
        double intensity = baseline + (current.total_intensity - baseline) * 
                          std::exp(-beta * dt * (i + 1));
        
        forecast.timestamps.push_back(t);
        forecast.intensities.push_back(intensity);
    }
    
    // Confidence interval (simplified)
    forecast.confidence_interval = 2.0 * std::sqrt(baseline);
    
    return forecast;
}

double HawkesProcess::calculate_conditional_intensity(
    const std::deque<Event>& events, Timestamp current_time, Side side) const {
    
    double intensity = config_.baseline_intensity;
    
    for (const auto& event : events) {
        if (event.timestamp >= current_time) break;
        
        // Self-excitation
        if (event.side == side) {
            intensity += config_.self_excitation * event.magnitude * 
                        exponential_kernel(event.timestamp, current_time);
        } 
        // Cross-excitation
        else {
            intensity += config_.cross_excitation * event.magnitude * 
                        exponential_kernel(event.timestamp, current_time);
        }
    }
    
    return intensity;
}

void HawkesProcess::calibrate_parameters(SymbolData& data) const {
    if (data.events.size() < 50) return;
    
    // Simplified MLE estimation
    // In production, would use full log-likelihood optimization
    
    // Estimate baseline intensity
    Timestamp duration = data.events.back().timestamp - data.events.front().timestamp;
    double duration_sec = duration / 1000000000.0;
    data.calibrated_baseline = data.events.size() / duration_sec;
    
    // Estimate decay rate using inter-arrival times
    std::vector<double> inter_arrivals;
    for (size_t i = 1; i < data.events.size(); ++i) {
        double dt = (data.events[i].timestamp - data.events[i-1].timestamp) / 1000000000.0;
        if (dt > 0 && dt < 10.0) { // Reasonable bounds
            inter_arrivals.push_back(dt);
        }
    }
    
    if (!inter_arrivals.empty()) {
        // Method of moments estimation
        double mean_inter = std::accumulate(inter_arrivals.begin(), 
                                          inter_arrivals.end(), 0.0) / 
                           inter_arrivals.size();
        
        // Estimate beta from mean inter-arrival time
        data.calibrated_beta = 1.0 / mean_inter;
        data.calibrated_beta = std::max(0.01, std::min(10.0, data.calibrated_beta));
    }
    
    // Estimate self-excitation using autocorrelation
    double excitation_sum = 0.0;
    int excitation_count = 0;
    
    for (size_t i = 1; i < data.events.size(); ++i) {
        // Look for clustering
        double dt = (data.events[i].timestamp - data.events[i-1].timestamp) / 1000000000.0;
        if (dt < 0.1 && data.events[i].side == data.events[i-1].side) {
            excitation_sum += 1.0 / dt;
            excitation_count++;
        }
    }
    
    if (excitation_count > 0) {
        data.calibrated_alpha = excitation_sum / excitation_count / data.calibrated_baseline;
        data.calibrated_alpha = std::max(0.0, std::min(0.9, data.calibrated_alpha));
    }
}

double HawkesProcess::kolmogorov_smirnov_test(const SymbolData& data) const {
    if (data.events.size() < 20) return 0.0;
    
    // Transform to residual process (time-rescaling theorem)
    std::vector<double> residuals;
    
    for (size_t i = 1; i < data.events.size(); ++i) {
        double intensity = calculate_conditional_intensity(
            data.events, data.events[i].timestamp, data.events[i].side
        );
        
        double dt = (data.events[i].timestamp - data.events[i-1].timestamp) / 1e9;
        double residual = 1.0 - std::exp(-intensity * dt);
        residuals.push_back(residual);
    }
    
    // K-S test against uniform distribution
    std::sort(residuals.begin(), residuals.end());
    
    double max_diff = 0.0;
    for (size_t i = 0; i < residuals.size(); ++i) {
        double empirical = (i + 1.0) / residuals.size();
        double theoretical = residuals[i]; // Should be uniform [0,1]
        max_diff = std::max(max_diff, std::abs(empirical - theoretical));
    }
    
    return max_diff;
}

} 
