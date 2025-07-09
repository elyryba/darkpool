#pragma once

#include "darkpool/strategies/strategy_base.hpp"
#include <deque>
#include <cmath>

namespace darkpool::strategies {

class ExecutionOptimizer : public StrategyBase {
public:
    enum class AlgorithmType {
        VWAP = 0,
        TWAP,
        IMPLEMENTATION_SHORTFALL,
        POV,              // Percentage of Volume
        MOC,              // Market on Close
        ADAPTIVE,         // ML-driven adaptive
        LIQUIDITY_SEEKING,
        ARRIVAL_PRICE
    };
    
    struct OptimizerConfig : Config {
        AlgorithmType algorithm = AlgorithmType::IMPLEMENTATION_SHORTFALL;
        
        // Execution parameters
        int64_t start_time_ms = 0;
        int64_t end_time_ms = 0;
        double participation_rate = 0.15;     // 15% of volume
        double min_participation = 0.05;
        double max_participation = 0.30;
        
        // Cost model parameters
        double temporary_impact_coef = 0.1;   // Kyle lambda approximation
        double permanent_impact_coef = 0.05;
        double fixed_cost_bps = 0.5;
        double spread_cost_multiplier = 0.5;
        
        // Urgency and risk
        double urgency = 0.5;                 // 0=patient, 1=urgent
        double risk_aversion = 0.5;           // 0=risk neutral, 1=very risk averse
        double alpha_decay_rate = 0.0001;    // Expected alpha decay
        
        // Adaptive parameters
        bool enable_adaptive_sizing = true;
        bool enable_opportunistic = true;
        double opportunity_threshold = 2.0;   // Spread standard deviations
        
        // Constraints
        double max_order_size = 10000;
        double min_order_size = 100;
        int64_t min_order_interval_ms = 100;
        double max_spread_cross_bps = 5.0;
    };
    
    explicit ExecutionOptimizer(const OptimizerConfig& config);
    ~ExecutionOptimizer() override;
    
    void on_market_data(const MarketMessage& msg) noexcept override;
    void on_anomaly(const Anomaly& anomaly) noexcept override;
    void on_fill(const Trade& trade, uint64_t order_id) noexcept override;
    void on_order_update(const ExecutionOrder& order) noexcept override;
    
    // Execution control
    void start_execution(uint32_t total_quantity, Side side) noexcept;
    void update_urgency(double new_urgency) noexcept;
    void pause_execution() noexcept;
    void resume_execution() noexcept;
    
    // Performance metrics
    struct ExecutionPerformance {
        double implementation_shortfall_bps;
        double vwap_slippage_bps;
        double arrival_slippage_bps;
        double timing_cost_bps;
        double spread_capture_bps;
        double total_cost_bps;
        double fill_rate;
        int64_t avg_time_to_fill_ms;
        double participation_rate_actual;
        double price_improvement_bps;
    };
    
    ExecutionPerformance get_performance() const noexcept;
    
private:
    struct ExecutionState {
        uint32_t total_quantity;
        uint32_t executed_quantity;
        uint32_t outstanding_quantity;
        Side side;
        int64_t start_time;
        int64_t arrival_price;
        double arrival_spread;
        
        // VWAP tracking
        double market_vwap;
        uint64_t market_volume;
        
        // Cost tracking
        double total_cost;
        double temporary_impact;
        double permanent_impact;
        double timing_cost;
        
        // Schedule
        std::vector<uint32_t> target_schedule;
        std::vector<uint32_t> actual_schedule;
        size_t current_bucket;
    };
    
    struct MarketConditions {
        double volatility;
        double spread_bps;
        double depth_imbalance;
        double trade_intensity;
        double volume_rate;
        double price_trend;
        bool favorable_conditions;
        int64_t last_update;
    };
    
    OptimizerConfig config_;
    ExecutionState exec_state_;
    MarketConditions market_conditions_;
    
    // Algorithm implementations
    uint32_t calculate_vwap_quantity() noexcept;
    uint32_t calculate_twap_quantity() noexcept;
    uint32_t calculate_is_quantity() noexcept;
    uint32_t calculate_pov_quantity() noexcept;
    uint32_t calculate_adaptive_quantity() noexcept;
    
    // Optimal scheduling
    std::vector<uint32_t> generate_optimal_schedule() noexcept;
    void update_schedule_progress() noexcept;
    
    // Market impact models
    double estimate_temporary_impact(uint32_t quantity) noexcept;
    double estimate_permanent_impact(uint32_t quantity) noexcept;
    double estimate_total_cost(const std::vector<uint32_t>& schedule) noexcept;
    
    // Adaptive execution
    void update_market_conditions() noexcept;
    bool detect_opportunity() noexcept;
    uint32_t adjust_size_for_conditions(uint32_t base_size) noexcept;
    
    // Order placement
    void place_next_slice() noexcept;
    ExecutionOrder create_optimal_order(uint32_t quantity) noexcept;
    
    // Performance tracking
    void update_performance_metrics(const Trade& fill) noexcept;
    mutable ExecutionPerformance performance_;
};

// Specialized VWAP executor
class VWAPExecutor : public ExecutionOptimizer {
public:
    struct VWAPConfig : OptimizerConfig {
        enum VWAPType {
            HISTORICAL,      // Based on historical volume curve
            PREDICTED,       // ML predicted volume curve
            ADAPTIVE         // Adjusts to real-time volume
        };
        
        VWAPType vwap_type = ADAPTIVE;
        size_t lookback_days = 20;
        double volume_prediction_confidence = 0.8;
        bool allow_curve_deviation = true;
        double max_curve_deviation = 0.2;
    };
    
    explicit VWAPExecutor(const VWAPConfig& config);
    
private:
    std::vector<double> volume_curve_;
    std::vector<double> predicted_volumes_;
    
    void load_historical_curve() noexcept;
    void update_volume_prediction() noexcept;
};

// Implementation shortfall minimizer
class ISOptimizer : public ExecutionOptimizer {
public:
    struct ISConfig : OptimizerConfig {
        // Almgren-Chriss model parameters
        double eta = 0.1;          // Temporary impact
        double gamma = 0.05;       // Permanent impact  
        double sigma = 0.02;       // Volatility
        double lambda = 2e-6;      // Risk aversion
        
        bool use_adaptive_params = true;
        bool include_opportunity_cost = true;
    };
    
    explicit ISOptimizer(const ISConfig& config);
    
private:
    // Almgren-Chriss optimal trajectory
    std::vector<double> compute_optimal_trajectory() noexcept;
    double solve_characteristic_time() noexcept;
};

} 
