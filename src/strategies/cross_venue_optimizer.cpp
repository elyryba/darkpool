#include "darkpool/strategies/cross_venue_optimizer.hpp"
#include <algorithm>
#include <numeric>
#include <random>

namespace darkpool::strategies {

CrossVenueOptimizer::CrossVenueOptimizer(const CrossVenueConfig& config)
    : StrategyBase(config)
    , config_(config) {
}

CrossVenueOptimizer::~CrossVenueOptimizer() = default;

void CrossVenueOptimizer::on_market_data(const MarketMessage& msg) noexcept {
    StrategyBase::on_market_data(msg);
    
    if (state_ != StrategyState::ACTIVE) return;
    
    // Check for latency arbitrage opportunities
    if (config_.enable_latency_arbitrage && msg.index() == 0) { // Quote
        const auto& quote = std::get<Quote>(msg);
        
        // Track quotes from different venues
        // In production, would track per-venue quotes
        // Here we simulate with venue detection from timestamp patterns
        
        for (const auto& [venue1, state1] : venues_) {
            for (const auto& [venue2, state2] : venues_) {
                if (venue1 != venue2 && state1->enabled && state2->enabled) {
                    if (detect_latency_opportunity(venue1, venue2)) {
                        // Execute arbitrage
                        execute_latency_arbitrage(venue1, venue2, 1000, Side::BUY);
                    }
                }
            }
        }
    }
    
    // Check split orders for rebalancing
    auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    for (auto& [parent_id, split_order] : split_orders_) {
        if (!split_order.completed && 
            now - split_order.start_time > config_.cancel_timeout_ms * 1000000) {
            rebalance_unfilled_quantity(parent_id);
        }
    }
}

void CrossVenueOptimizer::on_anomaly(const Anomaly& anomaly) noexcept {
    // Adjust venue selection based on anomalies
    if (anomaly.type == AnomalyType::VENUE_BREAKDOWN) {
        // Disable problematic venue temporarily
        if (anomaly.metadata.find("venue_id") != anomaly.metadata.end()) {
            std::string venue_id = std::to_string(anomaly.metadata.at("venue_id"));
            disable_venue(venue_id);
        }
    }
}

void CrossVenueOptimizer::on_fill(const Trade& trade, uint64_t order_id) noexcept {
    StrategyBase::on_fill(trade, order_id);
    
    // Update venue model
    auto order_it = active_orders_.find(order_id);
    if (order_it != active_orders_.end()) {
        const auto& order = order_it->second;
        update_venue_model(order.venue_id, trade, order);
        
        // Update split order tracking
        std::lock_guard<std::mutex> lock(orders_mutex_);
        for (auto& [parent_id, split_order] : split_orders_) {
            auto child_it = split_order.child_orders.find(order_id);
            if (child_it != split_order.child_orders.end()) {
                split_order.filled_by_venue[child_it->second] += trade.quantity;
                split_order.total_filled.fetch_add(trade.quantity);
                
                // Check if complete
                if (split_order.total_filled >= split_order.original.total_quantity) {
                    split_order.completed = true;
                }
                break;
            }
        }
    }
}

void CrossVenueOptimizer::on_order_update(const ExecutionOrder& order) noexcept {
    StrategyBase::on_order_update(order);
    
    // Track order rejections for venue health
    if (order.filled_quantity == 0 && 
        std::chrono::high_resolution_clock::now().time_since_epoch().count() - 
        order.timestamp > 1000000000) { // 1 second timeout
        
        std::lock_guard<std::mutex> lock(venues_mutex_);
        if (auto it = venues_.find(order.venue_id); it != venues_.end()) {
            it->second->orders_cancelled.fetch_add(1);
        }
    }
}

void CrossVenueOptimizer::register_venue(const std::string& venue_id,
                                        const VenueCharacteristics& chars) noexcept {
    std::lock_guard<std::mutex> lock(venues_mutex_);
    
    auto venue = std::make_unique<VenueState>();
    venue->characteristics = chars;
    venue->enabled = true;
    venue->score = 0.5; // Initial neutral score
    
    venues_[venue_id] = std::move(venue);
}

void CrossVenueOptimizer::execute_cross_venue(const CrossVenueOrder& order) noexcept {
    if (order.total_quantity == 0) return;
    
    // Optimize venue allocation
    auto allocation = optimize_allocation(order.total_quantity, order.side);
    
    if (allocation.empty()) return;
    
    // Create split order tracking
    static std::atomic<uint64_t> parent_id_gen{1};
    uint64_t parent_id = parent_id_gen.fetch_add(1);
    
    SplitOrder split;
    split.parent_id = parent_id;
    split.original = order;
    split.start_time = std::chrono::high_resolution_clock::now()
                      .time_since_epoch().count();
    
    // Send orders to each venue
    for (const auto& [venue_id, quantity] : allocation) {
        if (quantity > 0) {
            route_order_slice(split, venue_id, quantity);
        }
    }
    
    // Store split order
    {
        std::lock_guard<std::mutex> lock(orders_mutex_);
        split_orders_[parent_id] = std::move(split);
    }
}

std::unordered_map<std::string, uint32_t> CrossVenueOptimizer::optimize_allocation(
    uint32_t quantity, Side side) noexcept {
    
    std::unordered_map<std::string, uint32_t> allocation;
    std::vector<std::pair<double, std::string>> scored_venues;
    
    // Score each venue
    {
        std::lock_guard<std::mutex> lock(venues_mutex_);
        for (const auto& [venue_id, venue] : venues_) {
            if (!venue->enabled) continue;
            
            double score = calculate_venue_score(*venue, quantity, side);
            if (score >= config_.min_venue_score) {
                scored_venues.emplace_back(score, venue_id);
            }
        }
    }
    
    if (scored_venues.empty()) return allocation;
    
    // Sort by score
    std::sort(scored_venues.begin(), scored_venues.end(),
              std::greater<std::pair<double, std::string>>());
    
    // Allocate quantity with concentration limits
    uint32_t remaining = quantity;
    uint32_t max_per_venue = static_cast<uint32_t>(quantity * config_.max_venue_concentration);
    size_t venues_used = 0;
    
    for (const auto& [score, venue_id] : scored_venues) {
        if (venues_used >= config_.max_simultaneous_venues || remaining == 0) break;
        
        // Exploration vs exploitation
        static std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
        std::uniform_real_distribution<> dist(0.0, 1.0);
        
        if (config_.enable_venue_learning && dist(rng) < config_.explore_probability) {
            // Explore: give some allocation to lower-scored venues
            if (venues_used > 0 && score < scored_venues[0].first * 0.7) {
                continue; // Skip very low scores even in exploration
            }
        }
        
        // Calculate allocation based on score
        uint32_t venue_qty = static_cast<uint32_t>(remaining * score / 
                            (scored_venues[0].first + 0.1));
        venue_qty = std::min(venue_qty, max_per_venue);
        venue_qty = std::min(venue_qty, remaining);
        
        if (venue_qty > 0) {
            allocation[venue_id] = venue_qty;
            remaining -= venue_qty;
            venues_used++;
        }
    }
    
    // Distribute any remaining quantity to best venue
    if (remaining > 0 && !scored_venues.empty()) {
        allocation[scored_venues[0].second] += remaining;
    }
    
    return allocation;
}

double CrossVenueOptimizer::calculate_venue_score(const VenueState& venue,
                                                 uint32_t quantity, Side side) noexcept {
    double score = 0.0;
    
    // Base score from historical performance
    score = venue.score * 0.3;
    
    // Fill probability (25% weight)
    score += 0.25 * venue.characteristics.fill_probability;
    
    // Cost efficiency (25% weight)
    double total_cost = venue.characteristics.fee_bps - venue.characteristics.rebate_bps + 
                       venue.characteristics.market_impact_bps;
    double cost_score = 1.0 - (total_cost / config_.max_total_cost_bps);
    score += 0.25 * std::max(0.0, cost_score);
    
    // Latency score (10% weight)
    if (venue.characteristics.avg_latency_us < config_.max_venue_latency_us) {
        double latency_score = 1.0 - (venue.characteristics.avg_latency_us / 
                                     static_cast<double>(config_.max_venue_latency_us));
        score += 0.1 * latency_score;
    }
    
    // Size match (10% weight)
    double size_ratio = quantity / venue.characteristics.avg_fill_size;
    double size_score = 1.0 / (1.0 + std::abs(std::log(size_ratio)));
    score += 0.1 * size_score;
    
    // Hidden liquidity bonus for large orders
    if (quantity > 10000 && venue.characteristics.hidden_liquidity_ratio > 0.2) {
        score *= 1.1;
    }
    
    // Queue position estimate
    double queue_score = 1.0 - venue.avg_queue_position.load();
    score += 0.05 * queue_score;
    
    // Recent performance adjustment
    if (!venue.recent_fill_rates.empty()) {
        double recent_avg = std::accumulate(venue.recent_fill_rates.begin(),
                                          venue.recent_fill_rates.end(), 0.0) /
                           venue.recent_fill_rates.size();
        score = score * 0.7 + recent_avg * 0.3;
    }
    
    return std::min(1.0, std::max(0.0, score));
}

void CrossVenueOptimizer::route_order_slice(const SplitOrder& parent,
                                           const std::string& venue_id,
                                           uint32_t quantity) noexcept {
    ExecutionOrder order;
    order.symbol = config_.symbol;
    order.quantity = quantity;
    order.side = parent.original.side;
    order.type = parent.original.order_type;
    order.venue_id = venue_id;
    order.timestamp = std::chrono::high_resolution_clock::now()
                     .time_since_epoch().count();
    
    // Set venue type
    {
        std::lock_guard<std::mutex> lock(venues_mutex_);
        if (auto it = venues_.find(venue_id); it != venues_.end()) {
            order.venue = it->second->characteristics.supports_hidden_orders ?
                         VenueType::DARK_POOL : VenueType::LIT_EXCHANGE;
        }
    }
    
    // Set constraints based on venue characteristics
    order.participation_rate = 0.15; // Default 15%
    order.expire_time = order.timestamp + parent.original.timeout_ms * 1000000;
    
    uint64_t order_id = send_order(order);
    
    if (order_id > 0) {
        // Update tracking
        const_cast<SplitOrder&>(parent).child_orders[order_id] = venue_id;
        
        std::lock_guard<std::mutex> lock(venues_mutex_);
        if (auto it = venues_.find(venue_id); it != venues_.end()) {
            it->second->orders_sent.fetch_add(1);
        }
    }
}

void CrossVenueOptimizer::rebalance_unfilled_quantity(uint64_t parent_id) noexcept {
    std::lock_guard<std::mutex> lock(orders_mutex_);
    
    auto it = split_orders_.find(parent_id);
    if (it == split_orders_.end() || it->second.completed) return;
    
    auto& split_order = it->second;
    
    // Calculate unfilled quantity
    uint32_t unfilled = split_order.original.total_quantity - 
                       split_order.total_filled.load();
    
    if (unfilled == 0) {
        split_order.completed = true;
        return;
    }
    
    // Cancel slow venues and reallocate
    std::vector<uint64_t> orders_to_cancel;
    std::unordered_map<std::string, uint32_t> venue_performance;
    
    for (const auto& [order_id, venue_id] : split_order.child_orders) {
        auto order_it = active_orders_.find(order_id);
        if (order_it != active_orders_.end()) {
            const auto& order = order_it->second;
            if (order.filled_quantity == 0) {
                orders_to_cancel.push_back(order_id);
            } else {
                venue_performance[venue_id] = order.filled_quantity;
            }
        }
    }
    
    // Cancel underperforming orders
    for (uint64_t order_id : orders_to_cancel) {
        cancel_order(order_id);
    }
    
    // Reallocate to best performing venues
    if (!venue_performance.empty() && unfilled > 0) {
        // Find best venue
        auto best_venue = std::max_element(venue_performance.begin(), 
                                         venue_performance.end(),
                                         [](const auto& a, const auto& b) {
                                             return a.second < b.second;
                                         });
        
        // Send remaining to best venue
        route_order_slice(split_order, best_venue->first, unfilled);
    }
}

bool CrossVenueOptimizer::detect_latency_opportunity(const std::string& fast_venue,
                                                    const std::string& slow_venue) noexcept {
    // Simplified latency arbitrage detection
    // In production, would track actual quote timestamps from each venue
    
    std::lock_guard<std::mutex> lock(venues_mutex_);
    
    auto fast_it = venues_.find(fast_venue);
    auto slow_it = venues_.find(slow_venue);
    
    if (fast_it == venues_.end() || slow_it == venues_.end()) return false;
    
    int64_t latency_diff = slow_it->second->characteristics.avg_latency_us - 
                          fast_it->second->characteristics.avg_latency_us;
    
    // Need at least 100us difference for profitable arbitrage
    return latency_diff > 100;
}

void CrossVenueOptimizer::execute_latency_arbitrage(const std::string& fast_venue,
                                                   const std::string& slow_venue,
                                                   uint32_t quantity, Side side) noexcept {
    // Send aggressive order to fast venue
    ExecutionOrder fast_order;
    fast_order.symbol = config_.symbol;
    fast_order.quantity = quantity;
    fast_order.side = side;
    fast_order.type = OrderType::MARKET;
    fast_order.venue_id = fast_venue;
    fast_order.timestamp = std::chrono::high_resolution_clock::now()
                          .time_since_epoch().count();
    
    send_order(fast_order);
    
    // Send opposite side to slow venue (simplified)
    // In production, would implement proper arbitrage logic
}

void CrossVenueOptimizer::update_venue_model(const std::string& venue_id,
                                            const Trade& fill,
                                            const ExecutionOrder& order) noexcept {
    std::lock_guard<std::mutex> lock(venues_mutex_);
    
    auto it = venues_.find(venue_id);
    if (it == venues_.end()) return;
    
    auto& venue = *it->second;
    
    // Update fill tracking
    venue.orders_filled.fetch_add(1);
    venue.total_filled_quantity.fetch_add(fill.quantity);
    
    // Update latency estimate
    int64_t fill_latency = fill.timestamp - order.timestamp;
    int64_t current_avg = venue.total_latency_us.load() / 
                         (venue.orders_filled.load() + 1);
    venue.total_latency_us.fetch_add(fill_latency);
    
    // Update recent performance
    double fill_rate = static_cast<double>(fill.quantity) / order.quantity;
    venue.recent_fill_rates.push_back(fill_rate);
    if (venue.recent_fill_rates.size() > config_.learning_window) {
        venue.recent_fill_rates.pop_front();
    }
    
    // Update spread tracking
    double spread = (market_cache_.last_quote.ask_price - 
                    market_cache_.last_quote.bid_price) / 10000.0;
    venue.recent_spreads.push_back(spread);
    if (venue.recent_spreads.size() > config_.learning_window) {
        venue.recent_spreads.pop_front();
    }
    
    // Update cost tracking
    double cost_bps = 0.0;
    if (order.side == Side::BUY) {
        cost_bps = (fill.price - market_cache_.last_quote.bid_price) / 
                  market_cache_.last_quote.bid_price * 10000;
    } else {
        cost_bps = (market_cache_.last_quote.ask_price - fill.price) / 
                  market_cache_.last_quote.ask_price * 10000;
    }
    venue.total_cost.fetch_add(cost_bps * fill.quantity);
    
    // Update venue score using exponential moving average
    double new_score = fill_rate * 0.6 + (1.0 - cost_bps / 10.0) * 0.4;
    venue.score = venue.score * (1.0 - config_.learning_rate) + 
                 new_score * config_.learning_rate;
    
    // Update queue position model
    update_queue_position_model(venue_id, fill);
}

void CrossVenueOptimizer::update_queue_position_model(const std::string& venue_id,
                                                     const Trade& fill) noexcept {
    // Estimate queue position based on fill time
    // Faster fills = better queue position
    
    auto& venue = *venues_[venue_id];
    
    int64_t fill_time = fill.timestamp - 
                       active_orders_[fill.exchange_order_id].timestamp;
    
    // Normalize to [0, 1] where 0 = front of queue
    double position_estimate = 1.0 - std::exp(-fill_time / 1e9); // 1 second scale
    
    // Update running average
    uint32_t updates = venue.queue_model_updates.fetch_add(1) + 1;
    double current_avg = venue.avg_queue_position.load();
    double new_avg = (current_avg * (updates - 1) + position_estimate) / updates;
    venue.avg_queue_position.store(new_avg);
}

std::vector<CrossVenueOptimizer::VenueAnalytics> 
CrossVenueOptimizer::get_venue_analytics() const noexcept {
    std::vector<VenueAnalytics> analytics;
    
    std::lock_guard<std::mutex> lock(venues_mutex_);
    
    for (const auto& [venue_id, venue] : venues_) {
        VenueAnalytics va;
        va.venue_id = venue_id;
        
        uint64_t filled = venue->orders_filled.load();
        if (filled > 0) {
            va.effective_fill_rate = static_cast<double>(filled) / 
                                   venue->orders_sent.load();
            va.avg_time_to_fill = std::chrono::milliseconds(
                venue->total_latency_us.load() / filled / 1000
            );
            va.total_cost_bps = venue->total_cost.load() / 
                               venue->total_filled_quantity.load();
        }
        
        va.avg_queue_position = venue->avg_queue_position.load();
        va.total_volume = venue->total_filled_quantity.load();
        
        // Calculate realized spread
        if (!venue->recent_spreads.empty()) {
            va.realized_spread_bps = std::accumulate(venue->recent_spreads.begin(),
                                                    venue->recent_spreads.end(), 0.0) /
                                   venue->recent_spreads.size();
        }
        
        analytics.push_back(va);
    }
    
    return analytics;
}

// CrossVenueMarketMaker implementation

CrossVenueMarketMaker::CrossVenueMarketMaker(const MMConfig& config)
    : CrossVenueOptimizer(config) {
}

void CrossVenueMarketMaker::update_quotes() noexcept {
    std::unordered_map<std::string, Quote> quotes;
    calculate_optimal_quotes(quotes);
    
    // Send quotes to each venue
    for (const auto& [venue_id, quote] : quotes) {
        // In production, would send actual quote updates
        // Here we simulate with orders
        
        ExecutionOrder bid_order;
        bid_order.symbol = config_.symbol;
        bid_order.side = Side::BUY;
        bid_order.type = OrderType::LIMIT;
        bid_order.price = quote.bid_price;
        bid_order.quantity = quote.bid_size;
        bid_order.venue_id = venue_id;
        
        ExecutionOrder ask_order;
        ask_order.symbol = config_.symbol;
        ask_order.side = Side::SELL;
        ask_order.type = OrderType::LIMIT;
        ask_order.price = quote.ask_price;
        ask_order.quantity = quote.ask_size;
        ask_order.venue_id = venue_id;
        
        send_order(bid_order);
        send_order(ask_order);
    }
}

void CrossVenueMarketMaker::calculate_optimal_quotes(
    std::unordered_map<std::string, Quote>& quotes) noexcept {
    
    // Get current market
    int64_t mid_price = (market_cache_.last_quote.bid_price + 
                        market_cache_.last_quote.ask_price) / 2;
    
    // Calculate inventory skew
    int32_t position = get_position();
    double skew = position / static_cast<MMConfig*>(&config_)->inventory_limit;
    
    // Base spread
    int64_t half_spread = static_cast<MMConfig*>(&config_)->target_spread_bps * 
                         mid_price / 20000;
    
    // Adjust for inventory
    int64_t bid_offset = half_spread * (1.0 + skew * 
                        static_cast<MMConfig*>(&config_)->skew_multiplier);
    int64_t ask_offset = half_spread * (1.0 - skew * 
                        static_cast<MMConfig*>(&config_)->skew_multiplier);
    
    // Create quotes for each venue
    for (const auto& [venue_id, venue] : venues_) {
        if (!venue->enabled) continue;
        
        Quote q;
        q.symbol = config_.symbol;
        q.bid_price = mid_price - bid_offset;
        q.ask_price = mid_price + ask_offset;
        q.bid_size = 1000; // Simplified
        q.ask_size = 1000;
        q.timestamp = std::chrono::high_resolution_clock::now()
                     .time_since_epoch().count();
        
        quotes[venue_id] = q;
    }
}

} 
