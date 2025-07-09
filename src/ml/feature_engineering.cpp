#include "darkpool/ml/feature_engineering.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <immintrin.h>

namespace darkpool::ml {

FeatureEngineering::FeatureEngineering(uint32_t symbol, const Config& config)
    : symbol_(symbol)
    , config_(config)
    , price_buffer_(MAX_LOOKBACK)
    , volume_buffer_(MAX_LOOKBACK)
    , trade_size_buffer_(MAX_LOOKBACK)
    , spread_buffer_(MAX_LOOKBACK)
    , imbalance_buffer_(MAX_LOOKBACK) {
    
    // Pre-allocate minute bars
    minute_bars_.reserve(1440); // 24h of minute bars
    
    // Initialize cross-sectional data structures
    if (config_.enable_cross_sectional) {
        market_returns_.reserve(config_.correlation_symbols);
        sector_returns_.reserve(config_.correlation_symbols);
    }
}

void FeatureEngineering::on_quote(const Quote& quote) noexcept {
    if (quote.symbol != symbol_) return;
    
    // Update book state
    book_state_.levels[0].bid_price = quote.bid_price;
    book_state_.levels[0].ask_price = quote.ask_price;
    book_state_.levels[0].bid_size = quote.bid_size;
    book_state_.levels[0].ask_size = quote.ask_size;
    book_state_.last_update = quote.timestamp;
    
    // Calculate and store spread
    if (quote.bid_price > 0 && quote.ask_price > 0) {
        double spread = (quote.ask_price - quote.bid_price) / 10000.0;
        spread_buffer_.push(spread);
        
        // Calculate imbalance
        double total_size = quote.bid_size + quote.ask_size;
        if (total_size > 0) {
            double imbalance = (quote.bid_size - quote.ask_size) / total_size;
            imbalance_buffer_.push(imbalance);
        }
    }
    
    last_quote_time_.store(quote.timestamp, std::memory_order_relaxed);
    quote_count_.fetch_add(1, std::memory_order_relaxed);
}

void FeatureEngineering::on_trade(const Trade& trade) noexcept {
    if (trade.symbol != symbol_) return;
    
    // Update price and volume buffers
    double price = trade.price / 10000.0;
    price_buffer_.push(price);
    volume_buffer_.push(trade.quantity);
    trade_size_buffer_.push(trade.quantity);
    
    // Update volume aggregates
    total_volume_.fetch_add(trade.quantity, std::memory_order_relaxed);
    if (trade.side == Side::BUY) {
        buy_volume_.fetch_add(trade.quantity, std::memory_order_relaxed);
    } else if (trade.side == Side::SELL) {
        sell_volume_.fetch_add(trade.quantity, std::memory_order_relaxed);
    }
    
    // Update time bars
    update_time_bars(trade.timestamp, trade.price, trade.quantity);
    
    last_trade_time_.store(trade.timestamp, std::memory_order_relaxed);
    trade_count_.fetch_add(1, std::memory_order_relaxed);
}

void FeatureEngineering::on_order(const Order& order) noexcept {
    // Update book depth if we maintain full book
    // For now, handled by quote updates
}

void FeatureEngineering::on_cancel(const Cancel& cancel) noexcept {
    // Update book state if tracking orders
    // For now, handled by quote updates
}

FeatureEngineering::Features FeatureEngineering::get_features() const noexcept {
    Features features{};
    features.symbol = symbol_;
    features.timestamp = std::max(last_trade_time_.load(), last_quote_time_.load());
    features.valid = has_sufficient_data();
    
    if (!features.valid) {
        return features;
    }
    
    // Calculate all feature groups
    calculate_microstructure_features(features);
    calculate_orderflow_features(features);
    calculate_price_features(features);
    calculate_volume_features(features);
    calculate_temporal_features(features);
    calculate_statistical_features(features);
    calculate_derived_features(features);
    
    // Pack features into array
    size_t idx = 0;
    
    // Microstructure features (8)
    features.values[idx++] = features.microstructure.bid_ask_spread;
    features.values[idx++] = features.microstructure.bid_ask_spread_bps;
    features.values[idx++] = features.microstructure.effective_spread;
    features.values[idx++] = features.microstructure.quoted_depth;
    features.values[idx++] = features.microstructure.depth_imbalance;
    features.values[idx++] = features.microstructure.weighted_mid_price;
    features.values[idx++] = features.microstructure.micro_price;
    features.values[idx++] = features.microstructure.quote_intensity;
    
    // Order flow features (8)
    features.values[idx++] = features.order_flow.trade_intensity;
    features.values[idx++] = features.order_flow.avg_trade_size;
    features.values[idx++] = features.order_flow.large_trade_ratio;
    features.values[idx++] = features.order_flow.buy_sell_imbalance;
    features.values[idx++] = features.order_flow.signed_volume;
    features.values[idx++] = features.order_flow.order_flow_toxicity;
    features.values[idx++] = features.order_flow.kyle_lambda;
    features.values[idx++] = features.order_flow.arrival_rate;
    
    // Price dynamics (8)
    features.values[idx++] = features.price_dynamics.returns_1m;
    features.values[idx++] = features.price_dynamics.returns_5m;
    features.values[idx++] = features.price_dynamics.returns_30m;
    features.values[idx++] = features.price_dynamics.volatility_1m;
    features.values[idx++] = features.price_dynamics.volatility_5m;
    features.values[idx++] = features.price_dynamics.realized_variance;
    features.values[idx++] = features.price_dynamics.price_momentum;
    features.values[idx++] = features.price_dynamics.price_acceleration;
    
    // Volume features (8)
    features.values[idx++] = features.volume.volume_rate;
    features.values[idx++] = features.volume.vwap;
    features.values[idx++] = features.volume.twap;
    features.values[idx++] = features.volume.volume_profile_skew;
    features.values[idx++] = features.volume.participation_rate;
    features.values[idx++] = features.volume.relative_volume;
    features.values[idx++] = features.volume.volume_concentration;
    features.values[idx++] = features.volume.cumulative_delta;
    
    // Temporal features (8)
    features.values[idx++] = features.temporal.time_of_day_sin;
    features.values[idx++] = features.temporal.time_of_day_cos;
    features.values[idx++] = features.temporal.day_of_week;
    features.values[idx++] = features.temporal.time_since_open;
    features.values[idx++] = features.temporal.time_to_close;
    features.values[idx++] = features.temporal.intraday_pattern;
    features.values[idx++] = features.temporal.seasonal_component;
    features.values[idx++] = features.temporal.time_since_event;
    
    // Cross-sectional features (8)
    features.values[idx++] = features.cross_sectional.market_correlation;
    features.values[idx++] = features.cross_sectional.sector_beta;
    features.values[idx++] = features.cross_sectional.relative_strength;
    features.values[idx++] = features.cross_sectional.cross_asset_flow;
    features.values[idx++] = features.cross_sectional.market_impact;
    features.values[idx++] = features.cross_sectional.systematic_component;
    features.values[idx++] = features.cross_sectional.idiosyncratic_vol;
    features.values[idx++] = features.cross_sectional.contagion_score;
    
    // Derived features (8)
    features.values[idx++] = features.derived.tqr_ratio;
    features.values[idx++] = features.derived.vpin_toxicity;
    features.values[idx++] = features.derived.pin_probability;
    features.values[idx++] = features.derived.hawkes_intensity;
    features.values[idx++] = features.derived.hmm_state;
    features.values[idx++] = features.derived.drift_magnitude;
    features.values[idx++] = features.derived.refill_probability;
    features.values[idx++] = features.derived.clustering_score;
    
    // Statistical features (8)
    features.values[idx++] = features.statistical.kurtosis;
    features.values[idx++] = features.statistical.skewness;
    features.values[idx++] = features.statistical.jarque_bera;
    features.values[idx++] = features.statistical.hurst_exponent;
    features.values[idx++] = features.statistical.entropy;
    features.values[idx++] = features.statistical.fractal_dimension;
    features.values[idx++] = features.statistical.lyapunov_exponent;
    features.values[idx++] = features.statistical.autocorrelation;
    
    // Additional features to reach 128
    for (size_t period : config_.lookback_periods) {
        if (idx >= NUM_FEATURES) break;
        
        // Rolling statistics for each lookback
        auto [mean, std] = price_buffer_.get_mean_std(period);
        features.values[idx++] = mean;
        features.values[idx++] = std;
        features.values[idx++] = price_buffer_.get_percentile(0.95, period);
        features.values[idx++] = price_buffer_.get_percentile(0.05, period);
        
        // Volume statistics
        auto [vol_mean, vol_std] = volume_buffer_.get_mean_std(period);
        features.values[idx++] = vol_mean;
        features.values[idx++] = vol_std;
        
        // Spread statistics
        auto [spread_mean, spread_std] = spread_buffer_.get_mean_std(period);
        features.values[idx++] = spread_mean;
        features.values[idx++] = spread_std;
    }
    
    // Normalize remaining features
    while (idx < NUM_FEATURES) {
        features.values[idx++] = 0.0;
    }
    
    return features;
}

void FeatureEngineering::calculate_microstructure_features(Features& features) const noexcept {
    // Bid-ask spread
    if (book_state_.levels[0].bid_price > 0 && book_state_.levels[0].ask_price > 0) {
        features.microstructure.bid_ask_spread = 
            (book_state_.levels[0].ask_price - book_state_.levels[0].bid_price) / 10000.0;
        
        double mid_price = (book_state_.levels[0].bid_price + 
                           book_state_.levels[0].ask_price) / 2.0;
        features.microstructure.bid_ask_spread_bps = 
            features.microstructure.bid_ask_spread * 10000.0 / mid_price;
    }
    
    // Effective spread (recent average)
    features.microstructure.effective_spread = spread_buffer_.get_ema(0.1, 10);
    
    // Quoted depth
    features.microstructure.quoted_depth = 
        book_state_.levels[0].bid_size + book_state_.levels[0].ask_size;
    
    // Depth imbalance
    double total_depth = features.microstructure.quoted_depth;
    if (total_depth > 0) {
        features.microstructure.depth_imbalance = 
            (book_state_.levels[0].bid_size - book_state_.levels[0].ask_size) / total_depth;
    }
    
    // Weighted mid price
    if (total_depth > 0) {
        features.microstructure.weighted_mid_price = 
            (book_state_.levels[0].bid_price * book_state_.levels[0].ask_size +
             book_state_.levels[0].ask_price * book_state_.levels[0].bid_size) / 
            (total_depth * 10000.0);
    }
    
    // Micro price (using multiple levels if available)
    double micro_price = 0.0;
    double weight_sum = 0.0;
    for (size_t i = 0; i < std::min(config_.book_levels, size_t(5)); ++i) {
        if (book_state_.levels[i].bid_price > 0 && book_state_.levels[i].ask_price > 0) {
            double level_weight = 1.0 / (i + 1.0);
            double mid = (book_state_.levels[i].bid_price + 
                         book_state_.levels[i].ask_price) / 2.0;
            micro_price += mid * level_weight;
            weight_sum += level_weight;
        }
    }
    if (weight_sum > 0) {
        features.microstructure.micro_price = micro_price / (weight_sum * 10000.0);
    }
    
    // Quote intensity
    features.microstructure.quote_intensity = 
        quote_count_.load() / (last_quote_time_.load() / 1e9 + 1.0);
}

void FeatureEngineering::calculate_orderflow_features(Features& features) const noexcept {
    // Trade intensity
    double time_elapsed = last_trade_time_.load() / 1e9;
    features.order_flow.trade_intensity = trade_count_.load() / (time_elapsed + 1.0);
    
    // Average trade size
    features.order_flow.avg_trade_size = 
        total_volume_.load() / static_cast<double>(trade_count_.load() + 1);
    
    // Large trade ratio
    double large_threshold = features.order_flow.avg_trade_size * 2.0;
    size_t large_trades = 0;
    for (size_t i = 0; i < std::min(size_t(100), trade_count_.load()); ++i) {
        if (trade_size_buffer_.get_last(i) > large_threshold) {
            large_trades++;
        }
    }
    features.order_flow.large_trade_ratio = large_trades / 100.0;
    
    // Buy-sell imbalance
    double total_vol = total_volume_.load();
    if (total_vol > 0) {
        features.order_flow.buy_sell_imbalance = 
            (buy_volume_.load() - sell_volume_.load()) / total_vol;
    }
    
    // Signed volume (with decay)
    double signed_vol = 0.0;
    double decay = 0.95;
    for (size_t i = 0; i < 50; ++i) {
        double vol = volume_buffer_.get_last(i);
        double sign = (i % 2 == 0) ? 1.0 : -1.0; // Simplified
        signed_vol += sign * vol * std::pow(decay, i);
    }
    features.order_flow.signed_volume = signed_vol;
    
    // Order flow toxicity (simplified VPIN)
    features.order_flow.order_flow_toxicity = 
        std::abs(features.order_flow.buy_sell_imbalance) * 
        features.order_flow.trade_intensity / 100.0;
    
    // Kyle's lambda
    features.order_flow.kyle_lambda = calculate_kyle_lambda();
    
    // Arrival rate (Poisson intensity)
    features.order_flow.arrival_rate = features.order_flow.trade_intensity;
}

void FeatureEngineering::calculate_price_features(Features& features) const noexcept {
    // Returns over different horizons
    double current_price = price_buffer_.get_last(0);
    
    if (trade_count_.load() > 10) {
        features.price_dynamics.returns_1m = 
            (current_price - price_buffer_.get_last(10)) / price_buffer_.get_last(10);
    }
    
    if (trade_count_.load() > 50) {
        features.price_dynamics.returns_5m = 
            (current_price - price_buffer_.get_last(50)) / price_buffer_.get_last(50);
    }
    
    if (trade_count_.load() > 300) {
        features.price_dynamics.returns_30m = 
            (current_price - price_buffer_.get_last(300)) / price_buffer_.get_last(300);
    }
    
    // Volatility estimates
    features.price_dynamics.volatility_1m = std::sqrt(calculate_realized_variance(10));
    features.price_dynamics.volatility_5m = std::sqrt(calculate_realized_variance(50));
    features.price_dynamics.realized_variance = calculate_realized_variance(100);
    
    // Price momentum
    double momentum = 0.0;
    for (size_t period : {10, 20, 50}) {
        if (trade_count_.load() > period) {
            momentum += (current_price - price_buffer_.get_last(period)) / 
                       price_buffer_.get_last(period);
        }
    }
    features.price_dynamics.price_momentum = momentum / 3.0;
    
    // Price acceleration
    if (trade_count_.load() > 20) {
        double vel1 = current_price - price_buffer_.get_last(10);
        double vel2 = price_buffer_.get_last(10) - price_buffer_.get_last(20);
        features.price_dynamics.price_acceleration = (vel1 - vel2) / current_price;
    }
}

void FeatureEngineering::calculate_volume_features(Features& features) const noexcept {
    // Volume rate
    double time_elapsed = last_trade_time_.load() / 1e9;
    features.volume.volume_rate = total_volume_.load() / (time_elapsed + 1.0);
    
    // VWAP calculation
    double vwap_num = 0.0;
    double vwap_den = 0.0;
    for (size_t i = 0; i < std::min(size_t(100), trade_count_.load()); ++i) {
        double price = price_buffer_.get_last(i);
        double volume = volume_buffer_.get_last(i);
        vwap_num += price * volume;
        vwap_den += volume;
    }
    if (vwap_den > 0) {
        features.volume.vwap = vwap_num / vwap_den;
    }
    
    // TWAP
    double twap_sum = 0.0;
    size_t twap_count = std::min(size_t(100), trade_count_.load());
    for (size_t i = 0; i < twap_count; ++i) {
        twap_sum += price_buffer_.get_last(i);
    }
    if (twap_count > 0) {
        features.volume.twap = twap_sum / twap_count;
    }
    
    // Volume profile skew
    auto [vol_mean, vol_std] = volume_buffer_.get_mean_std(100);
    if (vol_std > 0) {
        double skew = 0.0;
        for (size_t i = 0; i < 100; ++i) {
            double z = (volume_buffer_.get_last(i) - vol_mean) / vol_std;
            skew += z * z * z;
        }
        features.volume.volume_profile_skew = skew / 100.0;
    }
    
    // Participation rate (recent volume / total volume)
    double recent_volume = 0.0;
    for (size_t i = 0; i < 20; ++i) {
        recent_volume += volume_buffer_.get_last(i);
    }
    features.volume.participation_rate = recent_volume / (total_volume_.load() + 1.0);
    
    // Relative volume
    features.volume.relative_volume = features.volume.volume_rate / 
                                     (features.order_flow.avg_trade_size + 1.0);
    
    // Volume concentration (Herfindahl index)
    std::vector<double> volumes;
    for (size_t i = 0; i < 50; ++i) {
        volumes.push_back(volume_buffer_.get_last(i));
    }
    double total = std::accumulate(volumes.begin(), volumes.end(), 0.0);
    double hhi = 0.0;
    if (total > 0) {
        for (double vol : volumes) {
            double share = vol / total;
            hhi += share * share;
        }
    }
    features.volume.volume_concentration = hhi;
    
    // Cumulative delta
    features.volume.cumulative_delta = buy_volume_.load() - sell_volume_.load();
}

void FeatureEngineering::calculate_temporal_features(Features& features) const noexcept {
    int64_t current_time = last_trade_time_.load();
    
    // Time of day encoding (sine/cosine for cyclical)
    int64_t seconds_in_day = (current_time / 1000000000) % 86400;
    double time_of_day_ratio = seconds_in_day / 86400.0;
    features.temporal.time_of_day_sin = std::sin(2 * M_PI * time_of_day_ratio);
    features.temporal.time_of_day_cos = std::cos(2 * M_PI * time_of_day_ratio);
    
    // Day of week
    int64_t days_since_epoch = current_time / (1000000000LL * 86400);
    features.temporal.day_of_week = (days_since_epoch + 4) % 7; // Thursday = 0
    
    // Time since market open (9:30 AM)
    int64_t market_open_seconds = 9 * 3600 + 30 * 60;
    features.temporal.time_since_open = 
        std::max(0LL, seconds_in_day - market_open_seconds) / 3600.0;
    
    // Time to market close (4:00 PM)
    int64_t market_close_seconds = 16 * 3600;
    features.temporal.time_to_close = 
        std::max(0LL, market_close_seconds - seconds_in_day) / 3600.0;
    
    // Intraday pattern (U-shape volume)
    double morning_factor = std::exp(-features.temporal.time_since_open / 2.0);
    double afternoon_factor = std::exp(-features.temporal.time_to_close / 2.0);
    features.temporal.intraday_pattern = morning_factor + afternoon_factor;
    
    // Seasonal component (simplified)
    int64_t day_of_year = (days_since_epoch % 365);
    features.temporal.seasonal_component = 
        std::sin(2 * M_PI * day_of_year / 365.0);
    
    // Time since last significant event
    features.temporal.time_since_event = 0.0; // Placeholder
}

void FeatureEngineering::calculate_statistical_features(Features& features) const noexcept {
    std::vector<double> returns;
    for (size_t i = 1; i < std::min(size_t(100), trade_count_.load()); ++i) {
        double r = (price_buffer_.get_last(i-1) - price_buffer_.get_last(i)) / 
                   price_buffer_.get_last(i);
        returns.push_back(r);
    }
    
    if (returns.size() < 4) return;
    
    // Calculate moments
    double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    
    double variance = 0.0;
    double skew_num = 0.0;
    double kurt_num = 0.0;
    
    for (double r : returns) {
        double diff = r - mean;
        variance += diff * diff;
        skew_num += diff * diff * diff;
        kurt_num += diff * diff * diff * diff;
    }
    
    variance /= returns.size();
    double std_dev = std::sqrt(variance);
    
    if (std_dev > 0) {
        features.statistical.skewness = 
            (skew_num / returns.size()) / (std_dev * std_dev * std_dev);
        features.statistical.kurtosis = 
            (kurt_num / returns.size()) / (variance * variance) - 3.0;
        
        // Jarque-Bera test statistic
        features.statistical.jarque_bera = 
            returns.size() / 6.0 * (features.statistical.skewness * features.statistical.skewness +
                                   0.25 * features.statistical.kurtosis * features.statistical.kurtosis);
    }
    
    // Hurst exponent
    features.statistical.hurst_exponent = calculate_hurst_exponent(100);
    
    // Shannon entropy
    features.statistical.entropy = calculate_entropy(returns);
    
    // Fractal dimension (simplified box-counting)
    features.statistical.fractal_dimension = 2.0 - features.statistical.hurst_exponent;
    
    // Lyapunov exponent (simplified)
    double lyap = 0.0;
    for (size_t i = 1; i < returns.size(); ++i) {
        if (std::abs(returns[i-1]) > 1e-10) {
            lyap += std::log(std::abs((returns[i] - returns[i-1]) / returns[i-1]));
        }
    }
    features.statistical.lyapunov_exponent = lyap / returns.size();
    
    // Autocorrelation at lag 1
    double auto_corr = 0.0;
    for (size_t i = 1; i < returns.size(); ++i) {
        auto_corr += (returns[i] - mean) * (returns[i-1] - mean);
    }
    features.statistical.autocorrelation = auto_corr / ((returns.size() - 1) * variance);
}

void FeatureEngineering::calculate_derived_features(Features& features) const noexcept {
    // These would come from the actual detectors in production
    // Here we calculate simplified versions
    
    // TQR ratio
    features.derived.tqr_ratio = features.order_flow.trade_intensity / 
                                (features.microstructure.quote_intensity + 1.0);
    
    // VPIN toxicity (simplified)
    features.derived.vpin_toxicity = features.order_flow.order_flow_toxicity;
    
    // PIN probability (simplified)
    double informed_ratio = features.order_flow.large_trade_ratio;
    features.derived.pin_probability = 1.0 / (1.0 + std::exp(-5.0 * (informed_ratio - 0.2)));
    
    // Hawkes intensity (simplified)
    features.derived.hawkes_intensity = features.order_flow.arrival_rate * 
                                       (1.0 + 0.5 * features.volume.volume_concentration);
    
    // HMM state (simplified to 3 states)
    if (features.price_dynamics.volatility_1m > 2.0 * features.price_dynamics.volatility_5m) {
        features.derived.hmm_state = 2.0; // High volatility state
    } else if (features.order_flow.buy_sell_imbalance > 0.5) {
        features.derived.hmm_state = 1.0; // Buying pressure state
    } else {
        features.derived.hmm_state = 0.0; // Normal state
    }
    
    // Drift magnitude
    features.derived.drift_magnitude = std::abs(features.price_dynamics.price_momentum) * 
                                      features.volume.volume_rate;
    
    // Refill probability (iceberg detection)
    features.derived.refill_probability = 
        features.order_flow.large_trade_ratio * features.volume.volume_concentration;
    
    // Clustering score
    features.derived.clustering_score = 
        features.statistical.autocorrelation * features.volume.participation_rate;
}

// Helper method implementations

double FeatureEngineering::calculate_kyle_lambda() const noexcept {
    // Simplified Kyle's lambda: price impact per unit volume
    std::vector<double> price_changes;
    std::vector<double> volumes;
    
    for (size_t i = 1; i < std::min(size_t(50), trade_count_.load()); ++i) {
        double dp = std::abs(price_buffer_.get_last(i-1) - price_buffer_.get_last(i));
        double vol = volume_buffer_.get_last(i);
        if (vol > 0) {
            price_changes.push_back(dp);
            volumes.push_back(vol);
        }
    }
    
    if (price_changes.size() < 10) return 0.0;
    
    // Simple regression
    double sum_pv = 0.0, sum_v = 0.0, sum_vv = 0.0;
    for (size_t i = 0; i < price_changes.size(); ++i) {
        sum_pv += price_changes[i] * volumes[i];
        sum_v += volumes[i];
        sum_vv += volumes[i] * volumes[i];
    }
    
    double n = price_changes.size();
    double lambda = (n * sum_pv - sum_v * sum_v) / (n * sum_vv - sum_v * sum_v);
    
    return std::max(0.0, lambda);
}

double FeatureEngineering::calculate_realized_variance(size_t lookback) const noexcept {
    if (trade_count_.load() < lookback + 1) return 0.0;
    
    double variance = 0.0;
    for (size_t i = 1; i <= lookback; ++i) {
        double r = std::log(price_buffer_.get_last(i-1) / price_buffer_.get_last(i));
        variance += r * r;
    }
    
    return variance / lookback;
}

double FeatureEngineering::calculate_hurst_exponent(size_t lookback) const noexcept {
    if (trade_count_.load() < lookback) return 0.5;
    
    std::vector<double> log_prices;
    for (size_t i = 0; i < lookback; ++i) {
        log_prices.push_back(std::log(price_buffer_.get_last(i)));
    }
    
    // R/S analysis
    std::vector<double> rs_values;
    std::vector<double> log_n_values;
    
    for (size_t n = 4; n <= lookback/4; n *= 2) {
        double rs_sum = 0.0;
        size_t count = 0;
        
        for (size_t start = 0; start + n <= lookback; start += n) {
            // Calculate mean
            double mean = 0.0;
            for (size_t i = start; i < start + n; ++i) {
                mean += log_prices[i];
            }
            mean /= n;
            
            // Calculate cumulative deviations and range
            double max_y = -1e10, min_y = 1e10;
            double cum_dev = 0.0;
            
            for (size_t i = start; i < start + n; ++i) {
                cum_dev += log_prices[i] - mean;
                max_y = std::max(max_y, cum_dev);
                min_y = std::min(min_y, cum_dev);
            }
            
            double range = max_y - min_y;
            
            // Calculate standard deviation
            double std_dev = 0.0;
            for (size_t i = start; i < start + n; ++i) {
                double diff = log_prices[i] - mean;
                std_dev += diff * diff;
            }
            std_dev = std::sqrt(std_dev / n);
            
            if (std_dev > 0) {
                rs_sum += range / std_dev;
                count++;
            }
        }
        
        if (count > 0) {
            rs_values.push_back(std::log(rs_sum / count));
            log_n_values.push_back(std::log(n));
        }
    }
    
    // Linear regression to find Hurst exponent
    if (rs_values.size() < 2) return 0.5;
    
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_xx = 0.0;
    for (size_t i = 0; i < rs_values.size(); ++i) {
        sum_x += log_n_values[i];
        sum_y += rs_values[i];
        sum_xy += log_n_values[i] * rs_values[i];
        sum_xx += log_n_values[i] * log_n_values[i];
    }
    
    double n = rs_values.size();
    double hurst = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    
    return std::max(0.0, std::min(1.0, hurst));
}

double FeatureEngineering::calculate_entropy(const std::vector<double>& data) const noexcept {
    if (data.empty()) return 0.0;
    
    // Discretize data into bins
    const size_t num_bins = 10;
    auto [min_it, max_it] = std::minmax_element(data.begin(), data.end());
    double min_val = *min_it;
    double max_val = *max_it;
    double range = max_val - min_val;
    
    if (range < 1e-10) return 0.0;
    
    std::vector<size_t> bins(num_bins, 0);
    for (double val : data) {
        size_t bin = std::min(num_bins - 1, 
                             static_cast<size_t>((val - min_val) / range * num_bins));
        bins[bin]++;
    }
    
    // Calculate entropy
    double entropy = 0.0;
    for (size_t count : bins) {
        if (count > 0) {
            double p = static_cast<double>(count) / data.size();
            entropy -= p * std::log2(p);
        }
    }
    
    return entropy;
}

void FeatureEngineering::update_time_bars(int64_t timestamp, int64_t price, 
                                         uint64_t volume) noexcept {
    // Convert timestamp to minute boundary
    int64_t minute = timestamp / 60000000000LL;
    
    // Check if we need a new bar
    if (minute_bars_.empty() || minute_bars_.back().timestamp != minute) {
        TimeBar new_bar;
        new_bar.timestamp = minute;
        new_bar.open = price;
        new_bar.high = price;
        new_bar.low = price;
        new_bar.close = price;
        new_bar.volume = volume;
        new_bar.trades = 1;
        minute_bars_.push_back(new_bar);
        
        // Limit history
        if (minute_bars_.size() > 1440) {
            minute_bars_.erase(minute_bars_.begin());
        }
    } else {
        // Update current bar
        auto& bar = minute_bars_.back();
        bar.high = std::max(bar.high, price);
        bar.low = std::min(bar.low, price);
        bar.close = price;
        bar.volume += volume;
        bar.trades++;
    }
}

std::vector<double> FeatureEngineering::get_feature_importance() const noexcept {
    // Would be loaded from trained model
    // For now, return uniform importance
    return std::vector<double>(NUM_FEATURES, 1.0 / NUM_FEATURES);
}

std::vector<std::string> FeatureEngineering::get_feature_names() noexcept {
    std::vector<std::string> names;
    names.reserve(NUM_FEATURES);
    
    // Microstructure
    names.push_back("bid_ask_spread");
    names.push_back("bid_ask_spread_bps");
    names.push_back("effective_spread");
    names.push_back("quoted_depth");
    names.push_back("depth_imbalance");
    names.push_back("weighted_mid_price");
    names.push_back("micro_price");
    names.push_back("quote_intensity");
    
    // Order flow
    names.push_back("trade_intensity");
    names.push_back("avg_trade_size");
    names.push_back("large_trade_ratio");
    names.push_back("buy_sell_imbalance");
    names.push_back("signed_volume");
    names.push_back("order_flow_toxicity");
    names.push_back("kyle_lambda");
    names.push_back("arrival_rate");
    
    // Price dynamics
    names.push_back("returns_1m");
    names.push_back("returns_5m");
    names.push_back("returns_30m");
    names.push_back("volatility_1m");
    names.push_back("volatility_5m");
    names.push_back("realized_variance");
    names.push_back("price_momentum");
    names.push_back("price_acceleration");
    
    // Continue for all feature groups...
    // Fill remaining with generic names
    while (names.size() < NUM_FEATURES) {
        names.push_back("feature_" + std::to_string(names.size()));
    }
    
    return names;
}

// CircularBuffer implementations

std::pair<double, double> FeatureEngineering::CircularBuffer::get_mean_std(
    size_t lookback) const noexcept {
    
    if (lookback == 0) return {0.0, 0.0};
    
    double sum = 0.0;
    double sum_sq = 0.0;
    size_t count = 0;
    
    size_t pos = write_pos.load();
    for (size_t i = 0; i < std::min(lookback, capacity); ++i) {
        size_t idx = (pos - 1 - i + capacity) % capacity;
        double val = data[idx];
        sum += val;
        sum_sq += val * val;
        count++;
    }
    
    if (count == 0) return {0.0, 0.0};
    
    double mean = sum / count;
    double variance = (sum_sq / count) - (mean * mean);
    double std_dev = std::sqrt(std::max(0.0, variance));
    
    return {mean, std_dev};
}

double FeatureEngineering::CircularBuffer::get_percentile(
    double pct, size_t lookback) const noexcept {
    
    if (lookback == 0 || pct < 0.0 || pct > 1.0) return 0.0;
    
    std::vector<double> values;
    values.reserve(lookback);
    
    size_t pos = write_pos.load();
    for (size_t i = 0; i < std::min(lookback, capacity); ++i) {
        size_t idx = (pos - 1 - i + capacity) % capacity;
        values.push_back(data[idx]);
    }
    
    if (values.empty()) return 0.0;
    
    std::sort(values.begin(), values.end());
    size_t idx = static_cast<size_t>(pct * (values.size() - 1));
    return values[idx];
}

double FeatureEngineering::CircularBuffer::get_ema(
    double alpha, size_t lookback) const noexcept {
    
    if (lookback == 0 || alpha <= 0.0 || alpha >= 1.0) return 0.0;
    
    double ema = 0.0;
    double weight_sum = 0.0;
    
    size_t pos = write_pos.load();
    for (size_t i = 0; i < std::min(lookback, capacity); ++i) {
        size_t idx = (pos - 1 - i + capacity) % capacity;
        double weight = std::pow(1.0 - alpha, i);
        ema += data[idx] * weight;
        weight_sum += weight;
    }
    
    return weight_sum > 0 ? ema / weight_sum : 0.0;
}

}
