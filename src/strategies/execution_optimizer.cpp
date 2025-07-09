#include "darkpool/strategies/execution_optimizer.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace darkpool::strategies {

ExecutionOptimizer::ExecutionOptimizer(const OptimizerConfig& config)
    : StrategyBase(config)
    , config_(config) {
    
    // Initialize execution state
    exec_state_ = {};
    market_conditions_ = {};
    performance_ = {};
}

ExecutionOptimizer::~ExecutionOptimizer() = default;

void ExecutionOptimizer::on_market_data(const MarketMessage& msg) noexcept {
    StrategyBase::on_market_data(msg);
    
    if (state_ != StrategyState::ACTIVE) return;
    
    // Update market conditions
    update_market_conditions();
    
    // Update VWAP tracking
    if (msg.index() == 1) { // Trade
        const auto& trade = std::get<Trade>(msg);
        if (trade.symbol == config_.symbol) {
            double price = trade.price / 10000.0;
            exec_state_.market_volume += trade.quantity;
            exec_state_.market_vwap = 
                (exec_state_.market_vwap * (exec_state_.market_volume - trade.quantity) + 
                 price * trade.quantity) / exec_state_.market_volume;
        }
    }
    
    // Check if we need to place next slice
    if (exec_state_.total_quantity > 0 && 
        exec_state_.executed_quantity < exec_state_.total_quantity) {
        
        auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        int64_t elapsed_ms = (now - exec_state_.start_time) / 1000000;
        int64_t total_time_ms = config_.end_time_ms - config_.start_time_ms;
        
        if (total_time_ms > 0) {
            // Check if we're behind schedule
            double time_progress = static_cast<double>(elapsed_ms) / total_time_ms;
            double exec_progress = static_cast<double>(exec_state_.executed_quantity) / 
                                 exec_state_.total_quantity;
            
            if (time_progress > exec_progress + 0.05) { // 5% behind
                config_.urgency = std::min(1.0, config_.urgency + 0.1);
            }
        }
        
        place_next_slice();
    }
}

void ExecutionOptimizer::on_anomaly(const Anomaly& anomaly) noexcept {
    // Adjust execution based on anomalies
    switch (anomaly.type) {
        case AnomalyType::UNUSUAL_VOLUME:
            if (config_.enable_opportunistic) {
                // Increase participation in high volume
                config_.participation_rate = std::min(
                    config_.max_participation,
                    config_.participation_rate * 1.2
                );
            }
            break;
            
        case AnomalyType::PRICE_MOVEMENT:
            // Adjust urgency based on adverse price movement
            if (anomaly.magnitude > 0 && exec_state_.side == Side::BUY ||
                anomaly.magnitude < 0 && exec_state_.side == Side::SELL) {
                config_.urgency = std::min(1.0, config_.urgency + 0.2);
            }
            break;
            
        default:
            break;
    }
}

void ExecutionOptimizer::on_fill(const Trade& trade, uint64_t order_id) noexcept {
    StrategyBase::on_fill(trade, order_id);
    
    if (active_orders_.find(order_id) == active_orders_.end()) return;
    
    // Update execution state
    exec_state_.executed_quantity += trade.quantity;
    exec_state_.actual_schedule[exec_state_.current_bucket] += trade.quantity;
    
    // Update performance metrics
    update_performance_metrics(trade);
    
    // Calculate realized impact
    double fill_price = trade.price / 10000.0;
    double mid_price = (market_cache_.last_quote.bid_price + 
                       market_cache_.last_quote.ask_price) / 20000.0;
    double signed_impact = (exec_state_.side == Side::BUY ? 1 : -1) * 
                          (fill_price - mid_price) / mid_price;
    
    exec_state_.temporary_impact += signed_impact * trade.quantity;
}

void ExecutionOptimizer::start_execution(uint32_t total_quantity, Side side) noexcept {
    // Initialize execution
    exec_state_ = {};
    exec_state_.total_quantity = total_quantity;
    exec_state_.side = side;
    exec_state_.start_time = std::chrono::high_resolution_clock::now()
                            .time_since_epoch().count();
    
    // Capture arrival price
    exec_state_.arrival_price = (market_cache_.last_quote.bid_price + 
                                market_cache_.last_quote.ask_price) / 2;
    exec_state_.arrival_spread = market_cache_.last_quote.ask_price - 
                                market_cache_.last_quote.bid_price;
    
    // Generate optimal schedule
    exec_state_.target_schedule = generate_optimal_schedule();
    exec_state_.actual_schedule.resize(exec_state_.target_schedule.size(), 0);
    exec_state_.current_bucket = 0;
    
    // Start execution
    state_ = StrategyState::ACTIVE;
    place_next_slice();
}

std::vector<uint32_t> ExecutionOptimizer::generate_optimal_schedule() noexcept {
    std::vector<uint32_t> schedule;
    
    int64_t total_time_ms = config_.end_time_ms - config_.start_time_ms;
    if (total_time_ms <= 0) {
        // Single bucket for immediate execution
        schedule.push_back(exec_state_.total_quantity);
        return schedule;
    }
    
    // Number of time buckets (1 minute buckets)
    size_t num_buckets = std::max(size_t(1), 
                                 static_cast<size_t>(total_time_ms / 60000));
    schedule.resize(num_buckets, 0);
    
    switch (config_.algorithm) {
        case AlgorithmType::VWAP: {
            // Use historical volume curve
            // Simplified - would use actual historical data
            std::vector<double> volume_curve = {
                0.15, 0.10, 0.08, 0.07, 0.10, 0.10, 0.08, 0.07, 0.10, 0.15
            };
            
            for (size_t i = 0; i < num_buckets; ++i) {
                double curve_value = volume_curve[i % volume_curve.size()];
                schedule[i] = static_cast<uint32_t>(
                    exec_state_.total_quantity * curve_value / num_buckets
                );
            }
            break;
        }
        
        case AlgorithmType::TWAP: {
            // Equal distribution
            uint32_t per_bucket = exec_state_.total_quantity / num_buckets;
            uint32_t remainder = exec_state_.total_quantity % num_buckets;
            
            for (size_t i = 0; i < num_buckets; ++i) {
                schedule[i] = per_bucket;
                if (i < remainder) schedule[i]++;
            }
            break;
        }
        
        case AlgorithmType::IMPLEMENTATION_SHORTFALL: {
            // Almgren-Chriss optimal trajectory
            double T = total_time_ms / 1000.0; // seconds
            double X = exec_state_.total_quantity;
            
            // Market impact parameters
            double eta = config_.temporary_impact_coef;
            double gamma = config_.permanent_impact_coef;
            double sigma = 0.02; // Daily volatility
            double lambda = config_.risk_aversion;
            
            // Characteristic time
            double kappa = sqrt(lambda * sigma * sigma / eta);
            double tau = kappa * T;
            
            // Optimal trajectory
            for (size_t i = 0; i < num_buckets; ++i) {
                double t = (i + 0.5) / num_buckets * T;
                double remaining_time = T - t;
                
                // Remaining quantity at time t
                double x_t = X * (sinh(kappa * remaining_time) / sinh(tau));
                
                // Quantity to trade in this bucket
                if (i == 0) {
                    schedule[i] = static_cast<uint32_t>(X - x_t);
                } else {
                    double x_prev = X * (sinh(kappa * (T - (i-0.5)/num_buckets * T)) / 
                                       sinh(tau));
                    schedule[i] = static_cast<uint32_t>(x_prev - x_t);
                }
            }
            break;
        }
        
        case AlgorithmType::POV: {
            // Percentage of volume - adaptive
            for (size_t i = 0; i < num_buckets; ++i) {
                schedule[i] = 0; // Will be determined dynamically
            }
            break;
        }
        
        default:
            // Default to TWAP
            uint32_t per_bucket = exec_state_.total_quantity / num_buckets;
            for (size_t i = 0; i < num_buckets; ++i) {
                schedule[i] = per_bucket;
            }
    }
    
    // Ensure total matches
    uint32_t scheduled_total = std::accumulate(schedule.begin(), schedule.end(), 0u);
    if (scheduled_total < exec_state_.total_quantity) {
        schedule.back() += exec_state_.total_quantity - scheduled_total;
    }
    
    return schedule;
}

void ExecutionOptimizer::place_next_slice() noexcept {
    if (exec_state_.executed_quantity >= exec_state_.total_quantity) {
        state_ = StrategyState::COMPLETED;
        return;
    }
    
    // Calculate next order size
    uint32_t order_size = 0;
    
    switch (config_.algorithm) {
        case AlgorithmType::VWAP:
            order_size = calculate_vwap_quantity();
            break;
            
        case AlgorithmType::TWAP:
            order_size = calculate_twap_quantity();
            break;
            
        case AlgorithmType::IMPLEMENTATION_SHORTFALL:
            order_size = calculate_is_quantity();
            break;
            
        case AlgorithmType::POV:
            order_size = calculate_pov_quantity();
            break;
            
        case AlgorithmType::ADAPTIVE:
            order_size = calculate_adaptive_quantity();
            break;
            
        default:
            order_size = calculate_twap_quantity();
    }
    
    // Apply constraints
    order_size = std::max(order_size, static_cast<uint32_t>(config_.min_order_size));
    order_size = std::min(order_size, static_cast<uint32_t>(config_.max_order_size));
    order_size = std::min(order_size, exec_state_.total_quantity - 
                         exec_state_.executed_quantity);
    
    if (order_size > 0) {
        auto order = create_optimal_order(order_size);
        send_order(order);
    }
}

uint32_t ExecutionOptimizer::calculate_vwap_quantity() noexcept {
    // Get current bucket
    auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    int64_t elapsed_ms = (now - exec_state_.start_time) / 1000000;
    size_t bucket = elapsed_ms / 60000; // 1-minute buckets
    
    if (bucket >= exec_state_.target_schedule.size()) {
        // Past schedule - execute remaining
        return exec_state_.total_quantity - exec_state_.executed_quantity;
    }
    
    exec_state_.current_bucket = bucket;
    
    // Target for current bucket
    uint32_t bucket_target = exec_state_.target_schedule[bucket];
    uint32_t bucket_executed = exec_state_.actual_schedule[bucket];
    
    if (bucket_executed >= bucket_target) {
        return 0; // Wait for next bucket
    }
    
    // Adjust for market conditions
    uint32_t base_size = bucket_target - bucket_executed;
    
    if (config_.enable_adaptive_sizing) {
        base_size = adjust_size_for_conditions(base_size);
    }
    
    return base_size;
}

uint32_t ExecutionOptimizer::calculate_is_quantity() noexcept {
    // Implementation shortfall - front-loaded execution
    auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    int64_t elapsed_ms = (now - exec_state_.start_time) / 1000000;
    int64_t remaining_ms = config_.end_time_ms - config_.start_time_ms - elapsed_ms;
    
    if (remaining_ms <= 0) {
        return exec_state_.total_quantity - exec_state_.executed_quantity;
    }
    
    // Calculate optimal rate based on Almgren-Chriss
    double remaining_qty = exec_state_.total_quantity - exec_state_.executed_quantity;
    double T = remaining_ms / 1000.0;
    
    // Trading rate
    double eta = config_.temporary_impact_coef;
    double lambda = config_.risk_aversion;
    double sigma = market_conditions_.volatility;
    
    double kappa = sqrt(lambda * sigma * sigma / eta);
    double rate = remaining_qty * kappa / sinh(kappa * T);
    
    // Convert to quantity for next interval
    uint32_t quantity = static_cast<uint32_t>(rate * config_.min_order_interval_ms / 1000.0);
    
    return quantity;
}

uint32_t ExecutionOptimizer::calculate_pov_quantity() noexcept {
    // Match percentage of market volume
    double recent_volume = 0;
    
    for (const auto& trade : market_cache_.recent_trades) {
        recent_volume += trade.quantity;
    }
    
    if (market_cache_.recent_trades.size() > 0) {
        int64_t time_window = market_cache_.recent_trades.back().timestamp - 
                             market_cache_.recent_trades.front().timestamp;
        double volume_rate = recent_volume / (time_window / 1e9);
        
        // Target participation rate
        double target_rate = volume_rate * config_.participation_rate;
        
        // Quantity for next interval
        return static_cast<uint32_t>(target_rate * config_.min_order_interval_ms / 1000.0);
    }
    
    // Fallback to TWAP
    return calculate_twap_quantity();
}

uint32_t ExecutionOptimizer::calculate_adaptive_quantity() noexcept {
    uint32_t base_quantity = calculate_is_quantity();
    
    // Detect opportunities
    if (config_.enable_opportunistic && detect_opportunity()) {
        // Increase size during favorable conditions
        base_quantity = static_cast<uint32_t>(base_quantity * 1.5);
        
        // But respect participation limits
        double recent_volume = 0;
        for (const auto& trade : market_cache_.recent_trades) {
            recent_volume += trade.quantity;
        }
        
        if (recent_volume > 0) {
            uint32_t max_participation = static_cast<uint32_t>(
                recent_volume * config_.max_participation
            );
            base_quantity = std::min(base_quantity, max_participation);
        }
    }
    
    return base_quantity;
}

void ExecutionOptimizer::update_market_conditions() noexcept {
    if (market_cache_.recent_trades.size() < 10) return;
    
    // Calculate volatility (simplified)
    std::vector<double> returns;
    for (size_t i = 1; i < market_cache_.recent_trades.size(); ++i) {
        double p1 = market_cache_.recent_trades[i-1].price / 10000.0;
        double p2 = market_cache_.recent_trades[i].price / 10000.0;
        returns.push_back(log(p2 / p1));
    }
    
    double mean_return = std::accumulate(returns.begin(), returns.end(), 0.0) / 
                        returns.size();
    double variance = 0;
    for (double r : returns) {
        variance += pow(r - mean_return, 2);
    }
    variance /= returns.size();
    
    market_conditions_.volatility = sqrt(variance) * sqrt(252 * 6.5 * 60); // Annualized
    
    // Update spread
    if (market_cache_.last_quote.bid_price > 0) {
        double mid = (market_cache_.last_quote.bid_price + 
                     market_cache_.last_quote.ask_price) / 2.0;
        market_conditions_.spread_bps = 
            (market_cache_.last_quote.ask_price - market_cache_.last_quote.bid_price) / 
            mid * 10000;
    }
    
    // Trade intensity
    if (market_cache_.recent_trades.size() > 1) {
        int64_t time_span = market_cache_.recent_trades.back().timestamp - 
                           market_cache_.recent_trades.front().timestamp;
        market_conditions_.trade_intensity = 
            market_cache_.recent_trades.size() / (time_span / 1e9);
    }
    
    // Depth imbalance
    double total_depth = market_cache_.last_quote.bid_size + 
                        market_cache_.last_quote.ask_size;
    if (total_depth > 0) {
        market_conditions_.depth_imbalance = 
            (market_cache_.last_quote.bid_size - market_cache_.last_quote.ask_size) / 
            total_depth;
    }
    
    // Favorable conditions check
    market_conditions_.favorable_conditions = 
        market_conditions_.spread_bps < 2.0 &&
        market_conditions_.volatility < 0.20 &&
        abs(market_conditions_.depth_imbalance) < 0.3;
    
    market_conditions_.last_update = std::chrono::high_resolution_clock::now()
                                    .time_since_epoch().count();
}

bool ExecutionOptimizer::detect_opportunity() noexcept {
    // Wide spread opportunity
    if (market_conditions_.spread_bps > config_.opportunity_threshold * 2.0) {
        return true;
    }
    
    // Depth imbalance opportunity
    if (exec_state_.side == Side::BUY && market_conditions_.depth_imbalance < -0.5 ||
        exec_state_.side == Side::SELL && market_conditions_.depth_imbalance > 0.5) {
        return true;
    }
    
    // Low volatility opportunity
    if (market_conditions_.volatility < 0.10) {
        return true;
    }
    
    return false;
}

uint32_t ExecutionOptimizer::adjust_size_for_conditions(uint32_t base_size) noexcept {
    double adjustment = 1.0;
    
    // Adjust for spread
    if (market_conditions_.spread_bps > 3.0) {
        adjustment *= 0.8; // Reduce size in wide spreads
    } else if (market_conditions_.spread_bps < 1.0) {
        adjustment *= 1.2; // Increase size in tight spreads
    }
    
    // Adjust for volatility
    if (market_conditions_.volatility > 0.30) {
        adjustment *= 0.7; // Reduce in high volatility
    }
    
    // Adjust for depth imbalance
    if (exec_state_.side == Side::BUY && market_conditions_.depth_imbalance > 0.3) {
        adjustment *= 0.9; // More asks than bids
    } else if (exec_state_.side == Side::SELL && market_conditions_.depth_imbalance < -0.3) {
        adjustment *= 0.9; // More bids than asks
    }
    
    return static_cast<uint32_t>(base_size * adjustment);
}

ExecutionOrder ExecutionOptimizer::create_optimal_order(uint32_t quantity) noexcept {
    ExecutionOrder order;
    order.symbol = config_.symbol;
    order.quantity = quantity;
    order.side = exec_state_.side;
    order.timestamp = std::chrono::high_resolution_clock::now()
                     .time_since_epoch().count();
    
    // Determine order type based on urgency and conditions
    if (config_.urgency > 0.8 || market_conditions_.favorable_conditions) {
        order.type = OrderType::MARKET;
    } else if (config_.enable_opportunistic && detect_opportunity()) {
        order.type = OrderType::PEG;
        order.price = (exec_state_.side == Side::BUY) ? 
                     market_cache_.last_quote.bid_price : 
                     market_cache_.last_quote.ask_price;
    } else {
        order.type = OrderType::LIMIT;
        // Price at mid or better
        int64_t mid = (market_cache_.last_quote.bid_price + 
                      market_cache_.last_quote.ask_price) / 2;
        int64_t offset = market_conditions_.spread_bps > 2.0 ? 1 : 0;
        
        if (exec_state_.side == Side::BUY) {
            order.price = mid - offset;
        } else {
            order.price = mid + offset;
        }
    }
    
    // Set participation rate
    order.participation_rate = config_.participation_rate;
    
    // Time constraints
    order.expire_time = order.timestamp + config_.min_order_interval_ms * 1000000;
    
    return order;
}

void ExecutionOptimizer::update_performance_metrics(const Trade& fill) noexcept {
    double fill_price = fill.price / 10000.0;
    
    // Implementation shortfall
    double arrival_price = exec_state_.arrival_price / 10000.0;
    double shortfall = (exec_state_.side == Side::BUY) ? 
                      (fill_price - arrival_price) / arrival_price : 
                      (arrival_price - fill_price) / arrival_price;
    
    performance_.implementation_shortfall_bps = shortfall * 10000;
    
    // VWAP slippage
    if (exec_state_.market_vwap > 0) {
        double vwap_slip = (exec_state_.side == Side::BUY) ?
                          (fill_price - exec_state_.market_vwap) / exec_state_.market_vwap :
                          (exec_state_.market_vwap - fill_price) / exec_state_.market_vwap;
        performance_.vwap_slippage_bps = vwap_slip * 10000;
    }
    
    // Update fill rate
    performance_.fill_rate = static_cast<double>(exec_state_.executed_quantity) / 
                           exec_state_.total_quantity;
}

ExecutionOptimizer::ExecutionPerformance ExecutionOptimizer::get_performance() const noexcept {
    return performance_;
}

// Risk checks from base class
bool ExecutionOptimizer::should_send_order(const ExecutionOrder& order) noexcept {
    // Check spread limits
    double spread_bps = calculate_spread() * 10000;
    if (spread_bps > config_.max_spread_cross_bps) {
        return false;
    }
    
    // Check order interval
    auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    if (!active_orders_.empty()) {
        int64_t last_order_time = 0;
        for (const auto& [id, o] : active_orders_) {
            last_order_time = std::max(last_order_time, o.timestamp);
        }
        
        if (now - last_order_time < config_.min_order_interval_ms * 1000000) {
            return false;
        }
    }
    
    return StrategyBase::should_send_order(order);
}

// VWAP Executor implementation

VWAPExecutor::VWAPExecutor(const VWAPConfig& config)
    : ExecutionOptimizer(config) {
    
    if (config.vwap_type == VWAPConfig::HISTORICAL) {
        load_historical_curve();
    }
}

void VWAPExecutor::load_historical_curve() noexcept {
    // Load historical intraday volume profile
    // Simplified - would query historical data
    volume_curve_ = {
        0.08, 0.06, 0.05, 0.04, 0.04, 0.05, 0.06, 0.07,  // Morning
        0.08, 0.08, 0.07, 0.06, 0.05, 0.05, 0.06, 0.08,  // Midday  
        0.10, 0.12  // Close
    };
}

// IS Optimizer implementation

ISOptimizer::ISOptimizer(const ISConfig& config)
    : ExecutionOptimizer(config) {
}

std::vector<double> ISOptimizer::compute_optimal_trajectory() noexcept {
    std::vector<double> trajectory;
    
    // Almgren-Chriss optimal execution path
    double T = (config_.end_time_ms - config_.start_time_ms) / 1000.0;
    double X = exec_state_.total_quantity;
    
    ISConfig* is_config = static_cast<ISConfig*>(&config_);
    double tau = solve_characteristic_time();
    
    size_t num_steps = 100;
    for (size_t i = 0; i <= num_steps; ++i) {
        double t = i * T / num_steps;
        double remaining = X * sinh(tau * (1 - t/T)) / sinh(tau);
        trajectory.push_back(X - remaining);
    }
    
    return trajectory;
}

double ISOptimizer::solve_characteristic_time() noexcept {
    ISConfig* is_config = static_cast<ISConfig*>(&config_);
    return sqrt(is_config->lambda * is_config->sigma * is_config->sigma / is_config->eta);
}

} 
