#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "darkpool/types.hpp"
#include "darkpool/config.hpp"

namespace darkpool {

// Forward declarations
class DetectorImpl;
class RealTimeStream;
class MetricsCollector;

class Detector {
public:
    using AnomalyCallback = std::function<void(const Anomaly&)>;
    using MessageCallback = std::function<void(const MarketMessage&)>;
    using ErrorCallback = std::function<void(const std::string&)>;
    
    explicit Detector(const Config& config);
    ~Detector();
    
    // Non-copyable, movable
    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;
    Detector(Detector&&) = default;
    Detector& operator=(Detector&&) = default;
    
    // Start/stop detector
    void start();
    void stop();
    bool is_running() const;
    
    // Callbacks
    void on_anomaly(AnomalyCallback callback);
    void on_message(MessageCallback callback);
    void on_error(ErrorCallback callback);
    
    // Manual message injection (for testing)
    void process_message(const MarketMessage& message);
    
    // Get performance metrics
    PerformanceMetrics get_metrics() const;
    
    // Get detected anomalies history
    std::vector<Anomaly> get_anomaly_history(Symbol symbol, size_t max_count = 100) const;
    
    // Configuration updates (some parameters can be updated at runtime)
    void update_tqr_threshold(double threshold);
    void update_ml_model(const std::string& model_path);
    
    // Symbol management
    void add_symbol(const std::string& symbol);
    void remove_symbol(const std::string& symbol);
    std::vector<std::string> get_active_symbols() const;
    
private:
    std::unique_ptr<DetectorImpl> impl_;
};

// Strategy interface for custom trading logic
class Strategy {
public:
    virtual ~Strategy() = default;
    
    // Called when anomaly is detected
    virtual void on_anomaly(const Anomaly& anomaly) = 0;
    
    // Called for each market message
    virtual void on_market_update(const MarketMessage& message) {}
    
    // Called when hidden liquidity is detected
    struct HiddenLiquiditySignal {
        Symbol symbol;
        Side side;
        Quantity size_estimate;
        Price expected_price;
        double confidence;
        Venue venue;
        Timestamp timestamp;
    };
    
    virtual void on_hidden_liquidity(const HiddenLiquiditySignal& signal) {}
    
    // Risk management
    virtual bool should_trade(const Anomaly& anomaly) { return true; }
    
    // Get strategy name
    virtual std::string name() const = 0;
};

// Factory for creating detector with custom components
class DetectorFactory {
public:
    static std::unique_ptr<Detector> create_with_config(const std::filesystem::path& config_path);
    
    static std::unique_ptr<Detector> create_with_strategy(
        const Config& config,
        std::unique_ptr<Strategy> strategy
    );
    
    static std::unique_ptr<Detector> create_for_backtesting(
        const Config& config,
        const std::filesystem::path& data_path
    );
};

} 
