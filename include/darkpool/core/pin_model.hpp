#pragma once

#include <vector>
#include <deque>
#include <unordered_map>
#include <shared_mutex>
#include <random>
#include "darkpool/types.hpp"

namespace darkpool {

// Probability of Informed Trading Model
class PINModel {
public:
    struct Config {
        size_t estimation_window = 60;        // Days for parameter estimation
        size_t min_trades_per_day = 100;      // Minimum trades for valid day
        double convergence_threshold = 1e-6;  // EM convergence criteria
        size_t max_iterations = 100;          // Max EM iterations
        double informed_threshold = 0.25;     // PIN threshold for anomaly
        bool use_intraday_estimation = true;  // Enable intraday PIN
        size_t intraday_buckets = 24;         // Hour buckets for intraday
    };
    
    explicit PINModel(const Config& config = Config{});
    
    // Process market events
    void on_trade(const Trade& trade);
    void on_daily_close(Symbol symbol, Timestamp date);
    
    // PIN calculation results
    struct PINResult {
        double pin;                    // Probability of informed trading
        double alpha;                  // Probability of information event
        double delta;                  // Probability of bad news
        double mu;                     // Informed trader arrival rate
        double epsilon_buy;            // Uninformed buy arrival rate
        double epsilon_sell;           // Uninformed sell arrival rate
        size_t estimation_days;        // Days used in estimation
        double log_likelihood;         // Model fit quality
        Timestamp calculation_time;
    };
    
    // Calculate PIN for symbol
    PINResult calculate_pin(Symbol symbol) const;
    
    // Intraday PIN calculation
    struct IntradayPIN {
        std::vector<double> hourly_pin;     // PIN by hour
        double current_pin;                 // Current hour PIN
        double daily_average;               // Average across day
        size_t current_bucket;              // Current time bucket
    };
    
    IntradayPIN calculate_intraday_pin(Symbol symbol) const;
    
    // Detect informed trading anomalies
    std::optional<Anomaly> check_anomaly(Symbol symbol) const;
    
    // Get historical PIN values
    std::vector<std::pair<Timestamp, double>> get_pin_history(Symbol symbol, 
                                                              size_t days = 30) const;
    
private:
    // Daily trading data
    struct DailyData {
        Timestamp date;
        size_t buy_orders = 0;
        size_t sell_orders = 0;
        size_t total_trades = 0;
        Quantity buy_volume = 0;
        Quantity sell_volume = 0;
        
        // Intraday buckets
        std::vector<size_t> bucket_buys;
        std::vector<size_t> bucket_sells;
    };
    
    // PIN model parameters
    struct ModelParameters {
        double alpha = 0.2;      // P(information event)
        double delta = 0.5;      // P(bad news | info event)
        double mu = 100.0;       // Informed arrival rate
        double epsilon_b = 50.0; // Uninformed buy rate
        double epsilon_s = 50.0; // Uninformed sell rate
        bool is_estimated = false;
    };
    
    struct SymbolData {
        std::deque<DailyData> daily_history;
        DailyData current_day;
        ModelParameters parameters;
        ModelParameters intraday_params;
        
        // Cached results
        mutable PINResult last_result;
        mutable Timestamp last_calculation = 0;
    };
    
    // EM algorithm for parameter estimation
    ModelParameters estimate_parameters(const std::deque<DailyData>& data) const;
    
    // E-step: calculate posterior probabilities
    struct PosteriorProbs {
        double no_event;      // P(no info | data)
        double good_event;    // P(good info | data)
        double bad_event;     // P(bad info | data)
    };
    
    PosteriorProbs calculate_posteriors(const DailyData& day, 
                                       const ModelParameters& params) const;
    
    // Likelihood calculation
    double calculate_likelihood(size_t buys, size_t sells, 
                              const ModelParameters& params,
                              int event_type) const; // 0=none, 1=good, 2=bad
    
    // Poisson probability
    double poisson_prob(size_t k, double lambda) const;
    
    // Update daily data
    void update_daily_data(DailyData& day, const Trade& trade) const;
    
    // Get current time bucket
    size_t get_time_bucket(Timestamp timestamp) const;
    
    Config config_;
    mutable std::unordered_map<Symbol, SymbolData> symbol_data_;
    mutable std::shared_mutex data_mutex_;
    
    // Cache for Poisson calculations
    mutable std::vector<double> log_factorial_cache_;
    void init_factorial_cache() const;
};

} 
