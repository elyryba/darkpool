#include "darkpool/detector.hpp"
#include "darkpool/protocols/protocol_normalizer.hpp"
#include "darkpool/utils/ring_buffer.hpp"
#include "darkpool/utils/cpu_affinity.hpp"

// Include all detection algorithms
#include "darkpool/core/trade_to_quote_ratio.hpp"
#include "darkpool/core/slippage_tracker.hpp"
#include "darkpool/core/order_book_imbalance.hpp"
#include "darkpool/core/hidden_refill_detector.hpp"
#include "darkpool/core/trade_clustering.hpp"
#include "darkpool/core/hawkes_process.hpp"
#include "darkpool/core/hidden_markov_model.hpp"
#include "darkpool/core/vpin_calculator.hpp"
#include "darkpool/core/pin_model.hpp"
#include "darkpool/core/post_trade_drift.hpp"
#include "darkpool/core/execution_heatmap.hpp"

#include <thread>
#include <chrono>

namespace darkpool {

// Forward declaration
class DetectorImpl : public MessageHandler {
public:
    explicit DetectorImpl(const Config& config);
    ~DetectorImpl();
    
    void start();
    void stop();
    bool is_running() const { return running_.load(); }
    
    // MessageHandler interface
    void on_message(const MarketMessage& message) override;
    void on_error(const std::string& error) override;
    
    // Process message (must be <500ns)
    void process_message(const MarketMessage& message);
    
    // Callbacks
    void on_anomaly(Detector::AnomalyCallback callback) { anomaly_callback_ = callback; }
    void on_message_processed(Detector::MessageCallback callback) { message_callback_ = callback; }
    void on_error_occurred(Detector::ErrorCallback callback) { error_callback_ = callback; }
    
    // Get performance metrics
    PerformanceMetrics get_metrics() const;
    
    // Get anomaly history
    std::vector<Anomaly> get_anomaly_history(Symbol symbol, size_t max_count) const;
    
    // Runtime updates
    void update_tqr_threshold(double threshold);
    void update_ml_model(const std::string& model_path);
    
    // Symbol management
    void add_symbol(const std::string& symbol);
    void remove_symbol(const std::string& symbol);
    std::vector<std::string> get_active_symbols() const;

private:
    // Detection worker thread
    class DetectionWorker {
    public:
        DetectionWorker(DetectorImpl* detector, int cpu_id, size_t start_symbol, size_t end_symbol);
        ~DetectionWorker();
        
        void start();
        void stop();
        void process_batch();
        
    private:
        DetectorImpl* detector_;
        int cpu_id_;
        size_t start_symbol_;
        size_t end_symbol_;
        std::thread thread_;
        std::atomic<bool> running_{false};
    };
    
    // Initialize all components
    void initialize_algorithms();
    void initialize_workers();
    
    // Route message to algorithms
    void route_to_algorithms(const MarketMessage& message);
    
    // Check for anomalies
    void check_anomalies(Symbol symbol);
    
    // Aggregate anomalies from all algorithms
    void aggregate_anomalies(Symbol symbol, std::vector<Anomaly>& anomalies);
    
    Config config_;
    
    // Market data buffer (Disruptor pattern)
    std::unique_ptr<RingBuffer<MarketMessage>> market_data_buffer_;
    std::unique_ptr<RingBuffer<MarketMessage>::BatchEventProcessor<DetectorImpl>> event_processor_;
    
    // Protocol normalizer
    std::unique_ptr<ProtocolNormalizer> protocol_normalizer_;
    
    // Detection algorithms
    std::unique_ptr<TradeToQuoteRatio> tqr_;
    std::unique_ptr<SlippageTracker> slippage_;
    std::unique_ptr<OrderBookImbalance> order_book_;
    std::unique_ptr<HiddenRefillDetector> hidden_refill_;
    std::unique_ptr<TradeClustering> clustering_;
    std::unique_ptr<HawkesProcess> hawkes_;
    std::unique_ptr<HiddenMarkovModel> hmm_;
    std::unique_ptr<VPINCalculator> vpin_;
    std::unique_ptr<PINModel> pin_;
    std::unique_ptr<PostTradeDrift> drift_;
    std::unique_ptr<ExecutionHeatmap> heatmap_;
    
    // Worker threads
    std::vector<std::unique_ptr<DetectionWorker>> workers_;
    
    // Output queue
    std::unique_ptr<LockFreeQueue<Anomaly, 65536>> anomaly_queue_;
    
    // Metrics
    std::unique_ptr<MetricsCollector> metrics_;
    mutable PerformanceMetrics perf_metrics_;
    
    // Symbol mapping
    std::unordered_map<std::string, Symbol> symbol_map_;
    std::unordered_map<Symbol, std::string> reverse_symbol_map_;
    std::atomic<Symbol> next_symbol_id_{1};
    
    // Last anomaly check time per symbol
    std::unordered_map<Symbol, Timestamp> last_anomaly_check_;
    
    // Callbacks
    Detector::AnomalyCallback anomaly_callback_;
    Detector::MessageCallback message_callback_;
    Detector::ErrorCallback error_callback_;
    
    // State
    std::atomic<bool> running_{false};
    mutable std::shared_mutex mutex_;
};

// Main Detector implementation
Detector::Detector(const Config& config) 
    : impl_(std::make_unique<DetectorImpl>(config)) {
}

Detector::~Detector() = default;

void Detector::start() {
    impl_->start();
}

void Detector::stop() {
    impl_->stop();
}

bool Detector::is_running() const {
    return impl_->is_running();
}

void Detector::on_anomaly(AnomalyCallback callback) {
    impl_->on_anomaly(callback);
}

void Detector::on_message(MessageCallback callback) {
    impl_->on_message_processed(callback);
}

void Detector::on_error(ErrorCallback callback) {
    impl_->on_error_occurred(callback);
}

void Detector::process_message(const MarketMessage& message) {
    impl_->process_message(message);
}

PerformanceMetrics Detector::get_metrics() const {
    return impl_->get_metrics();
}

std::vector<Anomaly> Detector::get_anomaly_history(Symbol symbol, size_t max_count) const {
    return impl_->get_anomaly_history(symbol, max_count);
}

void Detector::update_tqr_threshold(double threshold) {
    impl_->update_tqr_threshold(threshold);
}

void Detector::update_ml_model(const std::string& model_path) {
    impl_->update_ml_model(model_path);
}

void Detector::add_symbol(const std::string& symbol) {
    impl_->add_symbol(symbol);
}

void Detector::remove_symbol(const std::string& symbol) {
    impl_->remove_symbol(symbol);
}

std::vector<std::string> Detector::get_active_symbols() const {
    return impl_->get_active_symbols();
}

// DetectorImpl implementation
DetectorImpl::DetectorImpl(const Config& config) : config_(config) {
    // Initialize components
    market_data_buffer_ = std::make_unique<RingBuffer<MarketMessage>>(
        config.performance.ring_buffer_size
    );
    
    anomaly_queue_ = std::make_unique<LockFreeQueue<Anomaly, 65536>>();
    
    metrics_ = std::make_unique<MetricsCollector>(config.monitoring.prometheus_port);
    
    protocol_normalizer_ = std::make_unique<ProtocolNormalizer>();
    
    initialize_algorithms();
}

DetectorImpl::~DetectorImpl() {
    stop();
}

void DetectorImpl::initialize_algorithms() {
    // Initialize all detection algorithms with configs
    tqr_ = std::make_unique<TradeToQuoteRatio>(
        TradeToQuoteRatio::Config{
            config_.tqr.window_size,
            config_.tqr.threshold,
            config_.tqr.min_trades,
            config_.tqr.adaptive_threshold
        }
    );
    
    slippage_ = std::make_unique<SlippageTracker>(
        SlippageTracker::Config{
            config_.slippage.lookback_trades,
            config_.slippage.impact_decay,
            config_.slippage.use_vwap,
            config_.slippage.outlier_threshold
        }
    );
    
    order_book_ = std::make_unique<OrderBookImbalance>();
    hidden_refill_ = std::make_unique<HiddenRefillDetector>();
    clustering_ = std::make_unique<TradeClustering>();
    
    hawkes_ = std::make_unique<HawkesProcess>(
        HawkesProcess::Config{
            config_.hawkes.baseline_intensity,
            config_.hawkes.decay_rate,
            0.3, // self_excitation
            0.1, // cross_excitation
            config_.hawkes.max_history,
            config_.hawkes.kernel_bandwidth
        }
    );
    
    hmm_ = std::make_unique<HiddenMarkovModel>(
        HiddenMarkovModel::Config{
            static_cast<size_t>(config_.hmm.states),
            config_.hmm.observation_window,
            config_.hmm.transition_asymmetry,
            config_.hmm.convergence_threshold,
            config_.hmm.max_iterations
        }
    );
    
    vpin_ = std::make_unique<VPINCalculator>();
    pin_ = std::make_unique<PINModel>();
    drift_ = std::make_unique<PostTradeDrift>();
    heatmap_ = std::make_unique<ExecutionHeatmap>();
}

void DetectorImpl::initialize_workers() {
    size_t num_workers = config_.performance.cpu_affinity.size();
    if (num_workers == 0) {
        num_workers = std::thread::hardware_concurrency();
    }
    
    // Shard symbols across workers
    size_t symbols_per_worker = 1000 / num_workers; // Assume max 1000 symbols
    
    for (size_t i = 0; i < num_workers; ++i) {
        int cpu_id = i < config_.performance.cpu_affinity.size() ? 
                     config_.performance.cpu_affinity[i] : -1;
        
        size_t start_symbol = i * symbols_per_worker;
        size_t end_symbol = (i + 1) * symbols_per_worker;
        
        workers_.emplace_back(std::make_unique<DetectionWorker>(
            this, cpu_id, start_symbol, end_symbol
        ));
    }
}

void DetectorImpl::start() {
    if (running_.load()) return;
    
    running_.store(true);
    
    // Set up protocol normalizer
    protocol_normalizer_->set_handler(shared_from_this());
    
    // Add data sources from config
    for (const auto& source : config_.market_data_sources) {
        if (source.type == "FIX") {
            protocol_normalizer_->add_fix_source(source.host, source.port, source.symbols);
        } else if (source.type == "ITCH") {
            protocol_normalizer_->add_itch_source(source.host, source.port, source.symbols);
        } else if (source.type == "OUCH") {
            protocol_normalizer_->add_ouch_source(source.host, source.port, source.symbols);
        }
    }
    
    // Start components
    protocol_normalizer_->start();
    
    // Initialize workers
    initialize_workers();
    
    // Start workers
    for (auto& worker : workers_) {
        worker->start();
    }
    
    // Start event processor
    event_processor_ = std::make_unique<RingBuffer<MarketMessage>::BatchEventProcessor<DetectorImpl>>(
        *market_data_buffer_, *this
    );
    event_processor_->start();
}

void DetectorImpl::stop() {
    if (!running_.load()) return;
    
    running_.store(false);
    
    // Stop components
    if (event_processor_) {
        event_processor_->stop();
    }
    
    for (auto& worker : workers_) {
        worker->stop();
    }
    
    if (protocol_normalizer_) {
        protocol_normalizer_->stop();
    }
}

void DetectorImpl::on_message(const MarketMessage& message) {
    PerfCounter perf;
    
    // Add to ring buffer (zero copy)
    int64_t sequence = market_data_buffer_->claim_next();
    if (sequence >= 0) {
        (*market_data_buffer_)[sequence] = message;
        market_data_buffer_->publish(sequence);
        
        // Update metrics
        perf_metrics_.messages_processed.fetch_add(1, std::memory_order_relaxed);
        metrics_->fast_metrics().messages.fetch_add(1, std::memory_order_relaxed);
    } else {
        // Buffer full - this should never happen with proper sizing
        if (error_callback_) {
            error_callback_("Market data buffer full");
        }
    }
    
    // Track latency
    uint64_t latency_ns = perf.elapsed_ns();
    perf_metrics_.total_latency_ns.fetch_add(latency_ns, std::memory_order_relaxed);
    
    // Update max latency
    uint64_t current_max = perf_metrics_.max_latency_ns.load(std::memory_order_relaxed);
    while (latency_ns > current_max && 
           !perf_metrics_.max_latency_ns.compare_exchange_weak(current_max, latency_ns)) {}
}

void DetectorImpl::on_error(const std::string& error) {
    if (error_callback_) {
        error_callback_(error);
    }
}

void DetectorImpl::process_message(const MarketMessage& message) {
    PerfCounter perf;
    
    // Route to algorithms (visit pattern for std::variant)
    route_to_algorithms(message);
    
    // Periodically check for anomalies (not on every message for performance)
    std::visit([this](auto&& msg) {
        using T = std::decay_t<decltype(msg)>;
        if constexpr (std::is_same_v<T, Trade>) {
            Symbol symbol = msg.symbol;
            
            auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            auto& last_check = last_anomaly_check_[symbol];
            
            // Check every 100ms
            if (now - last_check > 100000000) {
                check_anomalies(symbol);
                last_check = now;
            }
        }
    }, message);
    
    // Call message callback if set
    if (message_callback_) {
        message_callback_(message);
    }
    
    // Track performance
    uint64_t latency_ns = perf.elapsed_ns();
    if (latency_ns > 500) {
        // Log slow processing
        metrics_->fast_metrics().messages.fetch_add(1, std::memory_order_relaxed);
    }
}

void DetectorImpl::route_to_algorithms(const MarketMessage& message) {
    // Use index-based dispatch for performance
    switch (message.index()) {
        case 0: { // Order
            const auto& order = std::get<Order>(message);
            hawkes_->on_order(order);
            hidden_refill_->on_order(order);
            pin_->on_daily_close(order.symbol, order.timestamp); // Simplified
            break;
        }
        
        case 1: { // Trade
            const auto& trade = std::get<Trade>(message);
            tqr_->on_trade(trade);
            slippage_->on_trade(trade);
            order_book_->on_trade(trade);
            hidden_refill_->on_trade(trade);
            clustering_->on_trade(trade);
            hawkes_->on_trade(trade);
            hmm_->on_trade(trade);
            vpin_->on_trade(trade);
            pin_->on_trade(trade);
            drift_->on_trade(trade);
            heatmap_->on_trade(trade);
            break;
        }
        
        case 2: { // Quote
            const auto& quote = std::get<Quote>(message);
            tqr_->on_quote(quote);
            slippage_->on_quote(quote);
            clustering_->on_quote(quote);
            hmm_->on_quote(quote);
            vpin_->on_quote(quote);
            drift_->on_quote(quote);
            break;
        }
        
        case 3: { // OrderBookSnapshot
            const auto& book = std::get<OrderBookSnapshot>(message);
            order_book_->on_order_book(book);
            slippage_->on_order_book(book);
            hmm_->on_order_book(book);
            break;
        }
    }
}

void DetectorImpl::check_anomalies(Symbol symbol) {
    std::vector<Anomaly> anomalies;
    aggregate_anomalies(symbol, anomalies);
    
    // Process detected anomalies
    for (const auto& anomaly : anomalies) {
        // Add to queue
        if (!anomaly_queue_->enqueue(anomaly)) {
            // Queue full - shouldn't happen
            metrics_->fast_metrics().anomalies.fetch_add(1, std::memory_order_relaxed);
        }
        
        // Update metrics
        perf_metrics_.anomalies_detected.fetch_add(1, std::memory_order_relaxed);
        metrics_->fast_metrics().anomalies.fetch_add(1, std::memory_order_relaxed);
        
        // Call callback
        if (anomaly_callback_) {
            anomaly_callback_(anomaly);
        }
        
        // Send to heatmap
        heatmap_->on_anomaly(anomaly);
    }
}

void DetectorImpl::aggregate_anomalies(Symbol symbol, std::vector<Anomaly>& anomalies) {
    // Check each algorithm for anomalies
    auto check_algorithm = [&](auto& algo, const char* name) {
        if (auto anomaly = algo->check_anomaly(symbol)) {
            anomalies.push_back(*anomaly);
        }
    };
    
    check_algorithm(tqr_, "TQR");
    check_algorithm(slippage_, "Slippage");
    check_algorithm(order_book_, "OrderBook");
    check_algorithm(hidden_refill_, "HiddenRefill");
    check_algorithm(clustering_, "Clustering");
    check_algorithm(hawkes_, "Hawkes");
    check_algorithm(hmm_, "HMM");
    check_algorithm(vpin_, "VPIN");
    check_algorithm(pin_, "PIN");
    check_algorithm(drift_, "Drift");
    
    // TODO: Aggregate confidence scores if multiple algorithms detect same anomaly
}

PerformanceMetrics DetectorImpl::get_metrics() const {
    return perf_metrics_;
}

std::vector<Anomaly> DetectorImpl::get_anomaly_history(Symbol symbol, size_t max_count) const {
    std::vector<Anomaly> history;
    
    // TODO: Implement anomaly history storage
    
    return history;
}

void DetectorImpl::update_tqr_threshold(double threshold) {
    // TODO: Update TQR config
}

void DetectorImpl::update_ml_model(const std::string& model_path) {
    // TODO: Reload ML model
}

void DetectorImpl::add_symbol(const std::string& symbol) {
    std::unique_lock lock(mutex_);
    
    if (symbol_map_.find(symbol) == symbol_map_.end()) {
        Symbol id = next_symbol_id_.fetch_add(1);
        symbol_map_[symbol] = id;
        reverse_symbol_map_[id] = symbol;
    }
}

void DetectorImpl::remove_symbol(const std::string& symbol) {
    std::unique_lock lock(mutex_);
    
    auto it = symbol_map_.find(symbol);
    if (it != symbol_map_.end()) {
        reverse_symbol_map_.erase(it->second);
        symbol_map_.erase(it);
    }
}

std::vector<std::string> DetectorImpl::get_active_symbols() const {
    std::shared_lock lock(mutex_);
    
    std::vector<std::string> symbols;
    for (const auto& [symbol, id] : symbol_map_) {
        symbols.push_back(symbol);
    }
    
    return symbols;
}

// DetectionWorker implementation
DetectorImpl::DetectionWorker::DetectionWorker(DetectorImpl* detector, int cpu_id, 
                                              size_t start_symbol, size_t end_symbol)
    : detector_(detector), cpu_id_(cpu_id), 
      start_symbol_(start_symbol), end_symbol_(end_symbol) {
}

DetectorImpl::DetectionWorker::~DetectionWorker() {
    stop();
}

void DetectorImpl::DetectionWorker::start() {
    running_.store(true);
    thread_ = std::thread([this] {
        // Set CPU affinity
        if (cpu_id_ >= 0) {
            CPUAffinity::set_thread_affinity(cpu_id_);
            CPUAffinity::set_thread_name("darkpool_detect_" + std::to_string(cpu_id_));
            CPUAffinity::set_scheduler_fifo(90);
        }
        
        while (running_.load()) {
            process_batch();
            
            // Yield to prevent spinning
            std::this_thread::yield();
        }
    });
}

void DetectorImpl::DetectionWorker::stop() {
    running_.store(false);
    if (thread_.joinable()) {
        thread_.join();
    }
}

void DetectorImpl::DetectionWorker::process_batch() {
    // Process anomaly checks for assigned symbols
    // This runs periodically, not on every message
    
    // TODO: Implement batch anomaly checking for assigned symbol range
}

// Factory methods
std::unique_ptr<Detector> DetectorFactory::create_with_config(const std::filesystem::path& config_path) {
    auto config = Config::load(config_path);
    return std::make_unique<Detector>(config);
}

std::unique_ptr<Detector> DetectorFactory::create_with_strategy(
    const Config& config, std::unique_ptr<Strategy> strategy) {
    
    auto detector = std::make_unique<Detector>(config);
    
    // Set up strategy callbacks
    detector->on_anomaly([&strategy](const Anomaly& anomaly) {
        strategy->on_anomaly(anomaly);
    });
    
    detector->on_message([&strategy](const MarketMessage& message) {
        strategy->on_market_update(message);
    });
    
    return detector;
}

std::unique_ptr<Detector> DetectorFactory::create_for_backtesting(
    const Config& config, const std::filesystem::path& data_path) {
    
    // TODO: Implement backtesting mode
    return std::make_unique<Detector>(config);
}

} 
