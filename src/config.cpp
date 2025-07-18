#include "darkpool/config.hpp"
#include <unordered_set>
#include <fstream>
#include <iostream>

namespace darkpool {

Config Config::load(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        throw ConfigException("Configuration file not found: " + path.string());
    }
    

    Config config;
    YAML::Node root = YAML::LoadFile(path.string());
    
    // Load sections
    if (root["market_data"]) {
        config.load_market_data(root["market_data"]);
    }
    

    if (root["detection"]) {
        config.load_detection(root["detection"]);
    }
    

    if (root["performance"]) {
        config.load_performance(root["performance"]);
    }
    

    if (root["visualization"]) {
        config.load_visualization(root["visualization"]);
    }
    

    if (root["monitoring"]) {
        config.load_monitoring(root["monitoring"]);
    }
    

    // Global settings
    if (root["logging"]) {
        config.enable_logging = root["logging"]["enabled"].as<bool>(true);
        config.log_level = root["logging"]["level"].as<std::string>("INFO");
        config.log_file = root["logging"]["file"].as<std::string>("darkpool.log");
    }
    

    config.dry_run = root["dry_run"].as<bool>(false);
    
    config.validate();
    return config;
}


void Config::load_market_data(const YAML::Node& node) {
    if (node["sources"]) {
        for (const auto& source : node["sources"]) {
            MarketDataSource mds;
            mds.type = source["type"].as<std::string>();
            mds.host = source["host"].as<std::string>();
            mds.port = source["port"].as<uint16_t>();
            
            if (source["symbols"]) {
                for (const auto& sym : source["symbols"]) {
                    mds.symbols.push_back(sym.as<std::string>());
                }

            }
            

            if (source["username"]) {
                mds.username = source["username"].as<std::string>();
            }
            

            if (source["password"]) {
                mds.password = source["password"].as<std::string>();
            }
            

            mds.use_ssl = source["use_ssl"].as<bool>(false);
            mds.buffer_size = source["buffer_size"].as<size_t>(65536);
            
            market_data_sources.push_back(std::move(mds));
        }

    }

}

void Config::load_detection(const YAML::Node& node) {
    // ML configuration
    if (node["enable_ml"]) {
        ml.enabled = node["enable_ml"].as<bool>(true);
    }
    

    if (node["ml_model"]) {
        ml.model_path = node["ml_model"].as<std::string>();
    }
    

    if (node["ml"]) {
        const auto& ml_node = node["ml"];
        ml.batch_size = ml_node["batch_size"].as<size_t>(1);
        ml.feature_window = ml_node["feature_window"].as<size_t>(100);
        ml.inference_timeout_ms = ml_node["inference_timeout_ms"].as<double>(5.0);
        ml.use_gpu = ml_node["use_gpu"].as<bool>(false);
        ml.gpu_device_id = ml_node["gpu_device_id"].as<int>(0);
    }
    

    // Algorithm configurations
    if (node["algorithms"]) {
        const auto& algos = node["algorithms"];
        
        if (algos["tqr"]) {
            const auto& tqr_node = algos["tqr"];
            tqr.window_size = tqr_node["window_size"].as<size_t>(1000);
            tqr.threshold = tqr_node["threshold"].as<double>(2.5);
            tqr.min_trades = tqr_node["min_trades"].as<size_t>(10);
            tqr.adaptive_threshold = tqr_node["adaptive_threshold"].as<bool>(true);
        }
        

        if (algos["hawkes"]) {
            const auto& hawkes_node = algos["hawkes"];
            hawkes.decay_rate = hawkes_node["decay_rate"].as<double>(0.1);
            hawkes.baseline_intensity = hawkes_node["baseline_intensity"].as<double>(0.5);
            hawkes.max_history = hawkes_node["max_history"].as<size_t>(1000);
            hawkes.kernel_bandwidth = hawkes_node["kernel_bandwidth"].as<double>(0.01);
        }
        

        if (algos["hmm"]) {
            const auto& hmm_node = algos["hmm"];
            hmm.states = hmm_node["states"].as<size_t>(3);
            hmm.transition_asymmetry = hmm_node["transition_asymmetry"].as<double>(0.2);
            hmm.observation_window = hmm_node["observation_window"].as<size_t>(500);
            hmm.convergence_threshold = hmm_node["convergence_threshold"].as<double>(1e-6);
            hmm.max_iterations = hmm_node["max_iterations"].as<size_t>(100);
        }
        

        if (algos["slippage"]) {
            const auto& slip_node = algos["slippage"];
            slippage.lookback_trades = slip_node["lookback_trades"].as<size_t>(100);
            slippage.impact_decay = slip_node["impact_decay"].as<double>(0.95);
            slippage.use_vwap = slip_node["use_vwap"].as<bool>(true);
            slippage.outlier_threshold = slip_node["outlier_threshold"].as<double>(3.0);
        }

    }

}

void Config::load_performance(const YAML::Node& node) {
    if (node["cpu_affinity"]) {
        for (const auto& cpu : node["cpu_affinity"]) {
            performance.cpu_affinity.push_back(cpu.as<int>());
        }

    }
    

    performance.numa_node = node["numa_node"].as<int>(-1);
    performance.use_huge_pages = node["huge_pages"].as<bool>(true);
    performance.memory_pool_size_mb = node["memory_pool_size_mb"].as<size_t>(1024);
    performance.ring_buffer_size = node["ring_buffer_size"].as<size_t>(1048576);
    performance.enable_prefetch = node["enable_prefetch"].as<bool>(true);
    performance.kernel_bypass = node["kernel_bypass"].as<bool>(false);
}


void Config::load_visualization(const YAML::Node& node) {
    visualization.enabled = node["enabled"].as<bool>(true);
    visualization.websocket_port = node["port"].as<uint16_t>(8080);
    visualization.update_frequency_ms = node["update_frequency"].as<size_t>(100);
    visualization.max_clients = node["max_clients"].as<size_t>(10);
    
    if (node["static_path"]) {
        visualization.static_path = node["static_path"].as<std::string>();
    }

}

void Config::load_monitoring(const YAML::Node& node) {
    monitoring.prometheus_enabled = node["prometheus_enabled"].as<bool>(true);
    monitoring.prometheus_port = node["prometheus_port"].as<uint16_t>(9090);
    monitoring.metrics_prefix = node["metrics_prefix"].as<std::string>("darkpool_");
    monitoring.metrics_flush_interval_ms = node["metrics_flush_interval_ms"].as<size_t>(1000);
}


void Config::validate() const {
    // Validate market data sources
    if (market_data_sources.empty()) {
        throw ConfigException("No market data sources configured");
    }
    

    for (const auto& source : market_data_sources) {
        if (source.type != "FIX" && source.type != "ITCH" && source.type != "OUCH") {
            throw ConfigException("Invalid market data source type: " + source.type);
        }
        

        if (source.symbols.empty()) {
            throw ConfigException("No symbols configured for " + source.type + " source");
        }

    }

    

    // Validate ML configuration
    if (ml.enabled && ml.model_path.empty()) {
        throw ConfigException("ML enabled but no model path specified");
    }

    

    if (ml.enabled && !std::filesystem::exists(ml.model_path)) {
        throw ConfigException("ML model file not found: " + ml.model_path);
    }

    

    // Validate performance settings
    if (performance.ring_buffer_size & (performance.ring_buffer_size - 1)) {
        throw ConfigException("Ring buffer size must be a power of 2");
    }

    

    // Validate detection parameters
    if (tqr.window_size == 0) {
        throw ConfigException("TQR window size must be greater than 0");
    }

    

    if (hawkes.decay_rate <= 0 || hawkes.decay_rate >= 1) {
        throw ConfigException("Hawkes decay rate must be between 0 and 1");
    }

    

    if (hmm.states < 2) {
        throw ConfigException("HMM must have at least 2 states");
    }
    

}

std::unordered_set<std::string> Config::get_all_symbols() const {
    std::unordered_set<std::string> symbols;
    for (const auto& source : market_data_sources) {
        symbols.insert(source.symbols.begin(), source.symbols.end());
    }

    return symbols;
}


}  


