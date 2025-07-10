#include <gtest/gtest.h>
#include "darkpool/core/hawkes_process.hpp"
#include <random>
#include <chrono>

using namespace darkpool;
using namespace darkpool::core;

class HawkesProcessTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.kernel_type = HawkesProcess::KernelType::EXPONENTIAL;
        config_.decay_rate = 0.5;
        config_.baseline_intensity = 1.0;
        config_.self_excitation = 0.3;
        config_.max_history = 1000;
        
        hawkes_ = std::make_unique<HawkesProcess>(12345, config_); // Symbol 12345
    }
    
    HawkesProcess::Config config_;
    std::unique_ptr<HawkesProcess> hawkes_;
    
    // Generate synthetic events
    std::vector<int64_t> generate_poisson_events(double lambda, int64_t duration_ns, int seed = 42) {
        std::mt19937 gen(seed);
        std::exponential_distribution<> dist(lambda);
        
        std::vector<int64_t> events;
        int64_t current_time = 0;
        
        while (current_time < duration_ns) {
            int64_t interval = static_cast<int64_t>(dist(gen) * 1e9); // Convert to nanoseconds
            current_time += interval;
            if (current_time < duration_ns) {
                events.push_back(current_time);
            }
        }
        
        return events;
    }
};

TEST_F(HawkesProcessTest, InitialState) {
    EXPECT_DOUBLE_EQ(hawkes_->get_current_intensity(), config_.baseline_intensity);
    EXPECT_EQ(hawkes_->get_event_count(), 0);
}

TEST_F(HawkesProcessTest, SingleEventExcitation) {
    Trade trade;
    trade.symbol = 12345;
    trade.timestamp = 1000000000; // 1 second
    trade.price = 10000;
    trade.quantity = 100;
    
    hawkes_->on_trade(trade);
    
    // Intensity should increase immediately after event
    double intensity_after = hawkes_->get_current_intensity();
    EXPECT_GT(intensity_after, config_.baseline_intensity);
    EXPECT_NEAR(intensity_after, config_.baseline_intensity + config_.self_excitation, 0.01);
}

TEST_F(HawkesProcessTest, IntensityDecay) {
    Trade trade;
    trade.symbol = 12345;
    trade.timestamp = 1000000000;
    trade.price = 10000;
    trade.quantity = 100;
    
    hawkes_->on_trade(trade);
    double initial_intensity = hawkes_->get_current_intensity();
    
    // Update with a later timestamp to trigger decay
    trade.timestamp = 2000000000; // 1 second later
    hawkes_->on_trade(trade);
    
    // Calculate expected decay
    double expected_decay = config_.self_excitation * std::exp(-config_.decay_rate * 1.0);
    double expected_intensity = config_.baseline_intensity + expected_decay + config_.self_excitation;
    
    EXPECT_NEAR(hawkes_->get_current_intensity(), expected_intensity, 0.01);
}

TEST_F(HawkesProcessTest, MultipleEvents) {
    // Generate a burst of events
    std::vector<Trade> trades;
    for (int i = 0; i < 10; ++i) {
        Trade trade;
        trade.symbol = 12345;
        trade.timestamp = i * 100000000; // 100ms intervals
        trade.price = 10000 + i;
        trade.quantity = 100;
        trades.push_back(trade);
    }
    
    // Process all trades
    for (const auto& trade : trades) {
        hawkes_->on_trade(trade);
    }
    
    // Intensity should be significantly elevated
    EXPECT_GT(hawkes_->get_current_intensity(), config_.baseline_intensity * 2);
}

TEST_F(HawkesProcessTest, MLECalibration) {
    // Generate synthetic Hawkes process data
    std::vector<int64_t> event_times;
    
    // Simple simulation
    std::mt19937 gen(12345);
    std::uniform_real_distribution<> unif(0, 1);
    
    double mu = 1.0;    // Baseline intensity
    double alpha = 0.5; // Self-excitation
    double beta = 1.0;  // Decay rate
    
    double t = 0;
    double t_max = 10.0; // 10 seconds
    
    while (t < t_max) {
        double lambda_star = mu;
        
        // Calculate intensity based on history
        for (double ti : event_times) {
            if (ti < t * 1e9) {
                lambda_star += alpha * std::exp(-beta * (t - ti / 1e9));
            }
        }
        
        // Next event time
        double u = unif(gen);
        double dt = -std::log(u) / lambda_star;
        t += dt;
        
        if (t < t_max) {
            event_times.push_back(static_cast<int64_t>(t * 1e9));
        }
    }
    
    // Feed events to Hawkes process
    for (auto time : event_times) {
        Trade trade;
        trade.symbol = 12345;
        trade.timestamp = time;
        trade.price = 10000;
        trade.quantity = 100;
        hawkes_->on_trade(trade);
    }
    
    // Calibrate parameters
    auto calibrated = hawkes_->calibrate_parameters();
    
    // Check if calibration is reasonable
    EXPECT_NEAR(calibrated.baseline_intensity, mu, mu * 0.3); // Within 30%
    EXPECT_NEAR(calibrated.self_excitation, alpha, alpha * 0.3);
    EXPECT_NEAR(calibrated.decay_rate, beta, beta * 0.3);
}

TEST_F(HawkesProcessTest, AnomalyDetection) {
    // Normal trading pattern
    for (int i = 0; i < 100; ++i) {
        Trade trade;
        trade.symbol = 12345;
        trade.timestamp = i * 100000000; // 100ms intervals
        trade.price = 10000;
        trade.quantity = 100;
        hawkes_->on_trade(trade);
    }
    
    // Check no anomaly during normal pattern
    Anomaly anomaly1 = hawkes_->check_anomaly();
    EXPECT_EQ(anomaly1.type, AnomalyType::NONE);
    
    // Sudden burst of activity
    int64_t burst_start = 100 * 100000000;
    for (int i = 0; i < 50; ++i) {
        Trade trade;
        trade.symbol = 12345;
        trade.timestamp = burst_start + i * 1000000; // 1ms intervals
        trade.price = 10000;
        trade.quantity = 1000; // Larger size
        hawkes_->on_trade(trade);
    }
    
    // Should detect anomaly
    Anomaly anomaly2 = hawkes_->check_anomaly();
    EXPECT_NE(anomaly2.type, AnomalyType::NONE);
    EXPECT_GT(anomaly2.confidence, 0.7);
}

TEST_F(HawkesProcessTest, DifferentKernels) {
    // Test power law kernel
    config_.kernel_type = HawkesProcess::KernelType::POWER_LAW;
    config_.power_law_exponent = 1.5;
    auto hawkes_power = std::make_unique<HawkesProcess>(12345, config_);
    
    Trade trade;
    trade.symbol = 12345;
    trade.timestamp = 1000000000;
    trade.price = 10000;
    trade.quantity = 100;
    
    hawkes_power->on_trade(trade);
    EXPECT_GT(hawkes_power->get_current_intensity(), config_.baseline_intensity);
    
    // Test sum of exponentials
    config_.kernel_type = HawkesProcess::KernelType::SUM_EXPONENTIALS;
    config_.decay_rates = {0.1, 1.0, 10.0};
    config_.excitation_weights = {0.5, 0.3, 0.2};
    auto hawkes_sum = std::make_unique<HawkesProcess>(12345, config_);
    
    hawkes_sum->on_trade(trade);
    EXPECT_GT(hawkes_sum->get_current_intensity(), config_.baseline_intensity);
}

TEST_F(HawkesProcessTest, PerformanceTest) {
    const int num_events = 10000;
    
    // Generate events
    std::vector<Trade> trades;
    for (int i = 0; i < num_events; ++i) {
        Trade trade;
        trade.symbol = 12345;
        trade.timestamp = i * 1000000; // 1ms intervals
        trade.price = 10000 + (i % 10);
        trade.quantity = 100;
        trades.push_back(trade);
    }
    
    // Measure processing time
    auto start = std::chrono::high_resolution_clock::now();
    
    for (const auto& trade : trades) {
        hawkes_->on_trade(trade);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    double avg_latency_us = static_cast<double>(duration) / num_events;
    std::cout << "Hawkes process average latency: " << avg_latency_us << " µs per event" << std::endl;
    
    // Should process each event in under 1 microsecond
    EXPECT_LT(avg_latency_us, 1.0);
}

TEST_F(HawkesProcessTest, MemoryStability) {
    // Test that memory usage is bounded with continuous events
    for (int i = 0; i < 100000; ++i) {
        Trade trade;
        trade.symbol = 12345;
        trade.timestamp = i * 1000000;
        trade.price = 10000;
        trade.quantity = 100;
        hawkes_->on_trade(trade);
    }
    
    // Event count should be capped at max_history
    EXPECT_LE(hawkes_->get_event_count(), config_.max_history);
}

TEST_F(HawkesProcessTest, IntensityBounds) {
    // Test that intensity remains bounded even with extreme inputs
    for (int burst = 0; burst < 10; ++burst) {
        // Create burst of events
        for (int i = 0; i < 1000; ++i) {
            Trade trade;
            trade.symbol = 12345;
            trade.timestamp = burst * 1000000000 + i * 1000; // 1µs intervals
            trade.price = 10000;
            trade.quantity = 10000; // Large trades
            hawkes_->on_trade(trade);
        }
        
        // Intensity should remain finite and reasonable
        double intensity = hawkes_->get_current_intensity();
        EXPECT_GT(intensity, 0);
        EXPECT_LT(intensity, 1000000); // Some reasonable upper bound
        EXPECT_FALSE(std::isnan(intensity));
        EXPECT_FALSE(std::isinf(intensity));
    }
}

// Benchmark test
TEST_F(HawkesProcessTest, BenchmarkIntensityCalculation) {
    // Pre-fill with events
    for (int i = 0; i < 1000; ++i) {
        Trade trade;
        trade.symbol = 12345;
        trade.timestamp = i * 1000000;
        trade.price = 10000;
        trade.quantity = 100;
        hawkes_->on_trade(trade);
    }
    
    const int iterations = 100000;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        volatile double intensity = hawkes_->get_current_intensity();
        (void)intensity; // Prevent optimization
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    double avg_latency_ns = static_cast<double>(duration) / iterations;
    std::cout << "Intensity calculation latency: " << avg_latency_ns << " ns" << std::endl;
    
    // Should be very fast
    EXPECT_LT(avg_latency_ns, 100);
}
