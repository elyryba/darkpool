#include <iostream>
#include <signal.h>
#include <atomic>
#include <filesystem>
#include <cxxopts.hpp>
#include "darkpool/detector.hpp"
#include "darkpool/utils/cpu_affinity.hpp"

std::atomic<bool> g_running{true};

void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        g_running.store(false);
    }
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    cxxopts::Options options("darkpool_detector", "Dark Pool Detection System");
    
    options.add_options()
        ("c,config", "Configuration file", cxxopts::value<std::string>()->default_value("config/production.yaml"))
        ("d,dry-run", "Dry run mode (no trades)", cxxopts::value<bool>()->default_value("false"))
        ("v,verbose", "Verbose logging", cxxopts::value<bool>()->default_value("false"))
        ("cpu", "CPU affinity list (e.g., 0,2,4)", cxxopts::value<std::string>())
        ("h,help", "Print help");
    
    auto result = options.parse(argc, argv);
    
    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }
    
    try {
        // Set up signal handlers
        signal(SIGINT, signal_handler);
        signal(SIGTERM, signal_handler);
        
        // Load configuration
        std::filesystem::path config_path = result["config"].as<std::string>();
        if (!std::filesystem::exists(config_path)) {
            std::cerr << "Configuration file not found: " << config_path << std::endl;
            return 1;
        }
        
        auto config = darkpool::Config::load(config_path);
        
        // Override config with command line options
        if (result.count("dry-run")) {
            config.dry_run = result["dry-run"].as<bool>();
        }
        
        // Set CPU affinity if specified
        if (result.count("cpu")) {
            std::string cpu_list = result["cpu"].as<std::string>();
            std::vector<int> cpus;
            
            std::stringstream ss(cpu_list);
            std::string cpu;
            while (std::getline(ss, cpu, ',')) {
                cpus.push_back(std::stoi(cpu));
            }
            
            darkpool::CPUAffinity::set_thread_affinity(cpus);
            std::cout << "Set CPU affinity to: " << cpu_list << std::endl;
        }
        
        // Optimize system settings
        darkpool::CPUAffinity::disable_cpu_frequency_scaling();
        darkpool::CPUAffinity::set_scheduler_fifo(95);
        
        // Create and configure detector
        auto detector = std::make_unique<darkpool::Detector>(config);
        
        // Set up anomaly callback
        detector->on_anomaly([](const darkpool::Anomaly& anomaly) {
            std::cout << "[ANOMALY] "
                     << "Symbol: " << anomaly.symbol << " "
                     << "Type: " << darkpool::to_string(anomaly.type) << " "
                     << "Confidence: " << anomaly.confidence << " "
                     << "Magnitude: " << anomaly.magnitude << " "
                     << "Hidden Size: " << anomaly.estimated_hidden_size << " "
                     << "Description: " << anomaly.description.data() << std::endl;
        });
        
        // Set up error callback
        detector->on_error([](const std::string& error) {
            std::cerr << "[ERROR] " << error << std::endl;
        });
        
        // Start detector
        std::cout << "Starting Dark Pool Detector..." << std::endl;
        std::cout << "Configuration: " << config_path << std::endl;
        std::cout << "Dry run: " << (config.dry_run ? "enabled" : "disabled") << std::endl;
        std::cout << "Press Ctrl+C to stop" << std::endl;
        
        detector->start();
        
        // Main loop
        while (g_running.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            
            // Print periodic stats
            auto metrics = detector->get_metrics();
            uint64_t messages = metrics.messages_processed.load();
            uint64_t anomalies = metrics.anomalies_detected.load();
            uint64_t avg_latency = messages > 0 ? 
                metrics.total_latency_ns.load() / messages : 0;
            
            std::cout << "\r[STATS] Messages: " << messages 
                     << " | Anomalies: " << anomalies
                     << " | Avg Latency: " << avg_latency << "ns" 
                     << std::flush;
        }
        
        std::cout << "\nShutting down..." << std::endl;
        detector->stop();
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
