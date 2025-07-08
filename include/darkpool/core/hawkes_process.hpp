#pragma once

#include <vector>
#include <deque>
#include <unordered_map>
#include <shared_mutex>
#include <cmath>
#include "darkpool/types.hpp"

namespace darkpool {

class HawkesProcess {
public:
    struct Config {
        double baseline_intensity = 0.5;      // Lambda_0 - baseline event rate
        double decay_rate = 0.1;             // Beta - exponential decay
        double self_excitation = 0.3;        // Alpha - self-excitation strength
        double cross_excitation = 0.1;       // Cross-excitation between buy/sell
        size_t max_history = 1000;           // Maximum events to track
        double kernel_bandwidth = 0.01;      // Kernel function bandwidth
        double intensity_threshold = 3.0;    // Anomaly threshold (multiples of baseline)
        size_t min_events = 20;             // Minimum events for analysis
    };
    
    explicit HawkesProcess(const Config& config = Config{});
    
    // Process market events
    void on_trade(const Trade& trade);
    void on_order(const Order& order);
    
    // Calculate current intensity
    struct IntensityResult {
        double buy_intensity;
        double sell_intensity;
        double total_intensity;
        double buy_pressure;         // Buy intensity / total
        double sell_pressure;        // Sell intensity / total
        Timestamp calculation_time;
    };
    
    IntensityResult calculate_intensity(Symbol symbol, Timestamp at_time = 0) const;
    
    // Detect intensity anomalies
    std::optional<Anomaly> check_anomaly(Symbol symbol) const;
    
    // Get intensity history
    struct IntensityPoint {
        Timestamp timestamp;
        double intensity;
        Side dominant_side;
    };
    
    std::vector<IntensityPoint> get_intensity_history(Symbol symbol, 
                                                      size_t points = 100,
                                                      Timestamp interval_ns = 1000000000) const;
    
    // Branching ratio (measure of endogeneity)
    double calculate_branching_ratio(Symbol symbol) const;
    
    // Predict future intensity
    struct IntensityForecast {
        std::vector<double> intensities;
        std::vector<Timestamp> timestamps;
        double confidence_interval;
    };
    
    IntensityForecast forecast_intensity(Symbol symbol, 
                                        Timestamp horizon_ns,
                                        size_t steps = 10) const;
    
private:
    struct Event {
        Timestamp timestamp;
        Side side;
        double magnitude;    // Size-based impact
        EventType type;     // Trade or Order
        
        enum class EventType {
            TRADE,
            ORDER
        };
    };
    
    struct SymbolData {
        std::deque<Event> events;
        double last_buy_intensity = 0.0;
        double last_sell_intensity = 0.0;
        Timestamp last_calculation = 0;
        
        // Calibrated parameters (can be updated online)
        double calibrated_baseline = 0.0;
        double calibrated_alpha = 0.0;
        double calibrated_beta = 0.0;
    };
    
    // Kernel functions
    double exponential_kernel(Timestamp t1, Timestamp t2) const;
    double power_law_kernel(Timestamp t1, Timestamp t2, double exponent = 1.1) const;
    
    // Intensity calculation
    double calculate_conditional_intensity(const std::deque<Event>& events,
                                         Timestamp current_time,
                                         Side side) const;
    
    // Parameter estimation using MLE
    void calibrate_parameters(SymbolData& data) const;
    
    // Goodness of fit test
    double kolmogorov_smirnov_test(const SymbolData& data) const;
    
    Config config_;
    mutable std::unordered_map<Symbol, SymbolData> symbol_data_;
    mutable std::shared_mutex data_mutex_;
};

// Inline implementations for performance
inline double HawkesProcess::exponential_kernel(Timestamp t1, Timestamp t2) const {
    if (t2 <= t1) return 0.0;
    
    double dt = (t2 - t1) / 1000000000.0; // Convert to seconds
    return config_.decay_rate * std::exp(-config_.decay_rate * dt);
}

inline double HawkesProcess::power_law_kernel(Timestamp t1, Timestamp t2, 
                                              double exponent) const {
    if (t2 <= t1) return 0.0;
    
    double dt = (t2 - t1) / 1000000000.0; // Convert to seconds
    return std::pow(1.0 + dt, -exponent);
}

} 
