#include "darkpool/strategies/dark_pool_strategy.hpp"
#include <algorithm>
#include <chrono>

namespace darkpool::strategies {

DarkPoolStrategy::DarkPoolStrategy(const DarkPoolConfig& config)
    : StrategyBase(config)
    , config_(config)
    , rng_(std::chrono::steady_clock::now().time_since_epoch().count())
    , size_dist_(1.0 - config.size_randomization, 1.0 + config.size_randomization)
    , time_dist_(-config.randomization_ms, config.randomization_ms) {
    
    if (config_.enable_ml_routing) {
        ml_engine_ = &ml::get_inference_engine();
    }
}

DarkPoolStrategy::~DarkPoolStrategy() = default;

void DarkPoolStrategy::on_market_data(const MarketMessage& msg) noexcept {
    StrategyBase::on_market_data(msg);
    
    if (state_ != StrategyState::ACTIVE) return;
    
    // Update market conditions
    if (msg.index() == 0) { // Quote
        const auto& quote = std::get<Quote>(msg);
        market_cache_.last_quote = quote;
        
        // Check if we should execute
        if (target_quantity_ > 0 && should_use_dark_pool()) {
            // Check parent orders for slicing
            for (auto& [id, parent] : parent_orders_) {
                if (parent.next_slice_time <= quote.timestamp) {
                    send_dark_orders(parent);
                    parent.next_slice_time = quote.timestamp + 
                        calculate_slice_interval(parent) + randomize_timing();
                }
            }
        }
    }
}

void DarkPoolStrategy::on_anomaly(const Anomaly& anomaly) noexcept {
    if (anomaly.type == AnomalyType::DARK_POOL_ACTIVITY) {
        // Adjust strategy based on detected dark pool activity
        if (anomaly.confidence > 0.8) {
            urgency_ = std::min(1.0, urgency_.load() + 0.1);
        }
    }
}

void DarkPoolStrategy::on_fill(const Trade& trade, uint64_t order_id) noexcept {
    StrategyBase::on_fill(trade, order_id);
    
    // Update venue metrics
    auto it = active_orders_.find(order_id);
    if (it != active_orders_.end()) {
        const auto& order = it->second;
        update_venue_learning(order.venue_id, trade);
        
        // Update parent order
        for (auto& [parent_id, parent] : parent_orders_) {
            auto child_it = std::find(parent.child_orders.begin(), 
                                     parent.child_orders.end(), order_id);
            if (child_it != parent.child_orders.end()) {
                parent.remaining_quantity -= trade.quantity;
                if (parent.remaining_quantity == 0) {
                    parent_orders_.erase(parent_id);
                }
                break;
            }
        }
    }
}

void DarkPoolStrategy::set_target_quantity(uint32_t quantity, Side side) noexcept {
    target_quantity_ = quantity;
    target_side_ = side;
    
    if (quantity > 0 && state_ == StrategyState::ACTIVE) {
        // Create parent order for slicing
        SliceOrder parent;
        parent.total_quantity = quantity;
        parent.remaining_quantity = quantity;
        parent.side = side;
        parent.start_time = std::chrono::high_resolution_clock::now()
                           .time_since_epoch().count();
        parent.urgency = urgency_.load();
        parent.slice_size = calculate_slice_size(parent);
        parent.next_slice_time = parent.start_time;
        
        static std::atomic<uint64_t> parent_id_gen{1};
        uint64_t parent_id = parent_id_gen.fetch_add(1);
        parent_orders_[parent_id] = parent;
        
        // Send initial orders
        send_dark_orders(parent);
    }
}

std::vector<std::string> DarkPoolStrategy::select_venues(uint32_t quantity) noexcept {
    std::lock_guard<std::mutex> lock(venue_mutex_);
    
    // Score and rank venues
    std::vector<std::pair<double, std::string>> scored_venues;
    
    for (const auto& [id, venue] : venues_) {
        if (!venue->is_healthy()) continue;
        
        double score = score_venue(*venue, quantity);
        if (score > config_.venue_fill_rate_threshold) {
            scored_venues.emplace_back(score, id);
        }
    }
    
    // Sort by score
    std::sort(scored_venues.begin(), scored_venues.end(), 
              std::greater<std::pair<double, std::string>>());
    
    // Select top venues up to max limit
    std::vector<std::string> selected;
    size_t max_venues = std::min(config_.max_venues_per_order, scored_venues.size());
    
    for (size_t i = 0; i < max_venues; ++i) {
        selected.push_back(scored_venues[i].second);
    }
    
    return selected;
}

void DarkPoolStrategy::send_dark_orders(const SliceOrder& parent) noexcept {
    if (parent.remaining_quantity == 0) return;
    
    uint32_t slice_quantity = std::min(parent.slice_size, parent.remaining_quantity);
    auto venues = select_venues(slice_quantity);
    
    if (venues.empty()) return;
    
    // Distribute across venues
    uint32_t qty_per_venue = slice_quantity / venues.size();
    uint32_t remainder = slice_quantity % venues.size();
    
    for (size_t i = 0; i < venues.size(); ++i) {
        uint32_t venue_qty = qty_per_venue;
        if (i == 0) venue_qty += remainder;
        
        // Apply anti-gaming randomization
        if (config_.enable_anti_gaming) {
            venue_qty = randomize_size(venue_qty);
        }
        
        auto order = create_dark_order(venues[i], venue_qty, parent.side);
        uint64_t order_id = send_order(order);
        
        if (order_id > 0) {
            const_cast<SliceOrder&>(parent).child_orders.push_back(order_id);
            
            // Update venue tracking
            std::lock_guard<std::mutex> lock(venue_mutex_);
            if (auto it = venues_.find(venues[i]); it != venues_.end()) {
                it->second->orders_sent.fetch_add(1);
                it->second->last_order_time = order.timestamp;
            }
        }
    }
}

ExecutionOrder DarkPoolStrategy::create_dark_order(const std::string& venue_id,
                                                  uint32_t quantity,
                                                  Side side) noexcept {
    ExecutionOrder order;
    order.symbol = config_.symbol;
    order.type = OrderType::HIDDEN;
    order.side = side;
    order.quantity = quantity;
    order.venue = VenueType::DARK_POOL;
    order.venue_id = venue_id;
    order.timestamp = std::chrono::high_resolution_clock::now()
                     .time_since_epoch().count();
    
    // Set constraints
    order.min_quantity = config_.min_dark_fill_size;
    order.participation_rate = config_.max_dark_participation;
    
    // Iceberg settings
    if (quantity > config_.min_dark_fill_size * 10) {
        order.type = OrderType::ICEBERG;
        order.max_floor = quantity * config_.iceberg_ratio;
        order.hidden_quantity = quantity - order.max_floor;
    }
    
    return order;
}

uint32_t DarkPoolStrategy::randomize_size(uint32_t base_size) noexcept {
    double factor = size_dist_(rng_);
    return static_cast<uint32_t>(base_size * factor);
}

int64_t DarkPoolStrategy::randomize_timing() noexcept {
    return time_dist_(rng_) * 1000000; // Convert to nanoseconds
}

double DarkPoolStrategy::score_venue(const VenueInfo& venue, uint32_t quantity) noexcept {
    double score = 0.0;
    
    // Fill rate component (40% weight)
    score += 0.4 * venue.fill_rate.load();
    
    // Cost component (30% weight)
    double cost_score = 1.0 - (venue.total_cost_bps.load() / 
                               (venue.orders_filled.load() + 1) / 10.0);
    score += 0.3 * std::max(0.0, cost_score);
    
    // Size match component (20% weight)
    double avg_size = venue.avg_fill_size.load();
    double size_match = 1.0 - std::abs(avg_size - quantity) / (avg_size + quantity);
    score += 0.2 * size_match;
    
    // Latency component (10% weight)
    double latency_score = 1.0 - (venue.avg_fill_time_ms.load() / 1000.0);
    score += 0.1 * std::max(0.0, latency_score);
    
    // ML adjustment if available
    if (ml_engine_ && config_.enable_ml_routing) {
        // Use ML predictions to adjust score
        // Simplified - would use actual features
        score *= 0.8 + 0.4 * venue.get_score();
    }
    
    // Penalize suspected gaming
    if (venue.consecutive_rejects.load() > 3) {
        score *= 0.5;
    }
    
    return score;
}

void DarkPoolStrategy::update_venue_learning(const std::string& venue_id,
                                            const Trade& fill) noexcept {
    std::lock_guard<std::mutex> lock(venue_mutex_);
    
    auto it = venues_.find(venue_id);
    if (it == venues_.end()) return;
    
    auto& venue = *it->second;
    
    // Update fill metrics
    venue.orders_filled.fetch_add(1);
    venue.total_filled_quantity.fetch_add(fill.quantity);
    
    // Update fill rate
    double fill_rate = static_cast<double>(venue.orders_filled.load()) /
                      (venue.orders_sent.load() + 1);
    venue.fill_rate.store(fill_rate);
    
    // Update average fill size
    double avg_size = static_cast<double>(venue.total_filled_quantity.load()) /
                     (venue.orders_filled.load() + 1);
    venue.avg_fill_size.store(avg_size);
    
    // Reset gaming counter on successful fill
    venue.consecutive_rejects.store(0);
}

bool DarkPoolStrategy::should_use_dark_pool() noexcept {
    // Check spread
    double spread_bps = calculate_spread() * 10000;
    if (spread_bps < config_.minimum_spread_bps) {
        return false;
    }
    
    // Check risk limits
    if (!check_risk_limits()) {
        return false;
    }
    
    // Check market conditions
    if (market_cache_.recent_trades.size() > 10) {
        // High activity might indicate better lit market liquidity
        double trade_rate = market_cache_.recent_trades.size() / 
                          (market_cache_.recent_trades.back().timestamp - 
                           market_cache_.recent_trades.front().timestamp) * 1e9;
        if (trade_rate > 100) { // 100 trades/sec
            return urgency_.load() < 0.7; // Only use dark if not urgent
        }
    }
    
    return true;
}

uint32_t DarkPoolStrategy::calculate_slice_size(const SliceOrder& parent) noexcept {
    uint32_t base_size = parent.total_quantity / config_.slice_count;
    
    // Adjust for urgency
    double urgency_factor = 0.5 + parent.urgency;
    base_size = static_cast<uint32_t>(base_size * urgency_factor);
    
    // Apply constraints
    base_size = std::max(base_size, static_cast<uint32_t>(config_.min_order_size));
    base_size = std::min(base_size, static_cast<uint32_t>(config_.max_order_size));
    
    return base_size;
}

int64_t DarkPoolStrategy::calculate_slice_interval(const SliceOrder& parent) noexcept {
    // Base interval
    int64_t interval = config_.min_order_interval_ms * 1000000;
    
    // Adjust for urgency (more urgent = shorter interval)
    double urgency_factor = 2.0 - parent.urgency;
    interval = static_cast<int64_t>(interval * urgency_factor);
    
    return interval;
}

// VenueInfo implementation

double DarkPoolStrategy::VenueInfo::get_score() const noexcept {
    return fill_rate.load() * 0.6 + 
           (1.0 - total_cost_bps.load() / 10.0) * 0.4;
}

bool DarkPoolStrategy::VenueInfo::is_healthy() const noexcept {
    return consecutive_rejects.load() < 5 &&
           fill_rate.load() > 0.3 &&
           orders_sent.load() > 0;
}

// DarkPoolAggregator implementation

DarkPoolAggregator::DarkPoolAggregator(const AggregatorConfig& config)
    : config_(config) {
}

DarkPoolAggregator::LiquiditySnapshot DarkPoolAggregator::get_aggregated_liquidity(
    uint32_t symbol) noexcept {
    
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    auto it = liquidity_cache_.find(symbol);
    if (it != liquidity_cache_.end()) {
        auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        if (now - it->second.timestamp < config_.rebalance_interval_ms * 1000000) {
            return it->second;
        }
    }
    
    // Create new snapshot
    LiquiditySnapshot snapshot;
    snapshot.timestamp = std::chrono::high_resolution_clock::now()
                        .time_since_epoch().count();
    
    // Aggregate from venues (simplified)
    snapshot.total_bid_liquidity = 50000;
    snapshot.total_ask_liquidity = 48000;
    snapshot.weighted_spread = 0.0002; // 2 bps
    
    liquidity_cache_[symbol] = snapshot;
    return snapshot;
}

DarkPoolAggregator::RoutingDecision DarkPoolAggregator::optimize_routing(
    uint32_t symbol, uint32_t quantity, Side side) noexcept {
    
    RoutingDecision decision;
    
    // Simple allocation (would use optimization in production)
    decision.venue_allocations.push_back({"VENUE1", quantity * 0.4});
    decision.venue_allocations.push_back({"VENUE2", quantity * 0.3});
    decision.venue_allocations.push_back({"VENUE3", quantity * 0.3});
    
    decision.expected_cost_bps = 1.5;
    decision.expected_fill_rate = 0.85;
    decision.expected_fill_time_ms = 250;
    
    return decision;
}

} 
