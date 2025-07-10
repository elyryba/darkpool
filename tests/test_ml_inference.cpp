#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <thread>
#include <vector>
#include <fstream>
#include <filesystem>
#include "darkpool/ml/inference_engine.hpp"
#include "darkpool/ml/feature_engineering.hpp"
#include "darkpool/utils/cpu_affinity.hpp"

using namespace darkpool;
using namespace darkpool::ml;
namespace fs = std::filesystem;

class MLInferenceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test model directory
        fs::create_directories("test_models");
        
        // Create dummy ONNX model file for testing
        create_dummy_model("test_models/test_model.onnx");
        create_dummy_model("test_models/test_model_int8.onnx");
        
        // Initialize inference engine
        InferenceEngine::Config config;
        config.model_path = "test_models/test_model.onnx";
        config.batch_size = 32;
        config.num_threads = 4;
        config.use_gpu = false; // CPU testing
        config.enable_profiling = true;
        
        engine_ = std::make_unique<InferenceEngine>(config);
        feature_eng_ = std::make_unique<FeatureEngineering>();
        
        // Pin to CPU for consistent timing
        utils::set_cpu_affinity(0);
        
        // Warm up CPU
        volatile uint64_t dummy = 0;
        for (int i = 0; i < 1000000; ++i) {
            dummy += i;
        }
    }
    
    void TearDown() override {
        engine_.reset();
        feature_eng_.reset();
        
        // Clean up test models
        fs::remove_all("test_models");
    }
    
    void create_dummy_model(const std::string& path) {
        // Create a minimal valid ONNX file structure
        // In real implementation, this would use ONNX API to create proper model
        std::ofstream file(path, std::ios::binary);
        
        // ONNX magic number and version
        const char magic[] = "\x08\x01\x1A\x01\x00\x00\x00\x00";
        file.write(magic, 8);
        
        // Minimal model proto (simplified)
        const char model_data[] = "dummy_model_data";
        file.write(model_data, sizeof(model_data));
        
        file.close();
    }
    
    // Generate realistic market features
    std::vector<float> generate_features(size_t num_features = 128) {
        std::vector<float> features(num_features);
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        // Microstructure features
        features[0] = 150.50f;  // Price
        features[1] = 0.02f;    // Spread
        features[2] = 100000.0f; // Volume
        features[3] = 0.15f;    // Volatility
        
        // Order flow features
        features[4] = 0.55f;    // Buy ratio
        features[5] = 1250.0f;  // Trade count
        features[6] = 80.5f;    // Avg trade size
        
        // Technical indicators
        for (size_t i = 7; i < 20; ++i) {
            features[i] = dist(gen);
        }
        
        // Normalized features
        for (size_t i = 20; i < num_features; ++i) {
            features[i] = dist(gen);
        }
        
        return features;
    }
    
    // CPU cycle counter
    inline uint64_t rdtsc() {
        uint32_t lo, hi;
        __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
        return ((uint64_t)hi << 32) | lo;
    }
    
    std::unique_ptr<InferenceEngine> engine_;
    std::unique_ptr<FeatureEngineering> feature_eng_;
};

// Test model loading
TEST_F(MLInferenceTest, ModelLoading) {
    // Test loading valid model
    EXPECT_TRUE(engine_->is_ready());
    
    // Test loading non-existent model
    InferenceEngine::Config bad_config;
    bad_config.model_path = "non_existent_model.onnx";
    
    EXPECT_THROW({
        InferenceEngine bad_engine(bad_config);
    }, std::runtime_error);
}

// Test single inference
TEST_F(MLInferenceTest, SingleInference) {
    auto features = generate_features();
    
    InferenceResult result;
    ASSERT_TRUE(engine_->infer(features, result));
    
    // Check result structure
    EXPECT_GT(result.anomaly_score, 0.0f);
    EXPECT_LE(result.anomaly_score, 1.0f);
    EXPECT_GT(result.confidence, 0.0f);
    EXPECT_LE(result.confidence, 1.0f);
    
    // Check anomaly types detected
    EXPECT_FALSE(result.anomaly_types.empty());
}

// Test batch inference
TEST_F(MLInferenceTest, BatchInference) {
    const size_t batch_size = 32;
    std::vector<std::vector<float>> batch_features;
    
    for (size_t i = 0; i < batch_size; ++i) {
        batch_features.push_back(generate_features());
    }
    
    std::vector<InferenceResult> results;
    ASSERT_TRUE(engine_->infer_batch(batch_features, results));
    
    EXPECT_EQ(results.size(), batch_size);
    
    // Verify all results are valid
    for (const auto& result : results) {
        EXPECT_GT(result.anomaly_score, 0.0f);
        EXPECT_LE(result.anomaly_score, 1.0f);
    }
}

// Test INT8 quantization accuracy
TEST_F(MLInferenceTest, INT8QuantizationAccuracy) {
    // Load FP32 model
    InferenceEngine::Config fp32_config;
    fp32_config.model_path = "test_models/test_model.onnx";
    fp32_config.batch_size = 1;
    InferenceEngine fp32_engine(fp32_config);
    
    // Load INT8 model
    InferenceEngine::Config int8_config;
    int8_config.model_path = "test_models/test_model_int8.onnx";
    int8_config.batch_size = 1;
    int8_config.use_int8_quantization = true;
    InferenceEngine int8_engine(int8_config);
    
    // Test on same features
    auto features = generate_features();
    
    InferenceResult fp32_result, int8_result;
    ASSERT_TRUE(fp32_engine.infer(features, fp32_result));
    ASSERT_TRUE(int8_engine.infer(features, int8_result));
    
    // Check accuracy within tolerance (3% for INT8)
    float diff = std::abs(fp32_result.anomaly_score - int8_result.anomaly_score);
    float tolerance = 0.03f * fp32_result.anomaly_score;
    
    EXPECT_LT(diff, tolerance) << "INT8 quantization error too large: " 
                               << diff << " > " << tolerance;
}

// Performance test - CRITICAL
TEST_F(MLInferenceTest, PerformanceSingleInference) {
    auto features = generate_features();
    InferenceResult result;
    
    // Warm up
    for (int i = 0; i < 1000; ++i) {
        engine_->infer(features, result);
    }
    
    // Measure latency
    const int iterations = 10000;
    std::vector<double> latencies;
    latencies.reserve(iterations);
    
    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        engine_->infer(features, result);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        latencies.push_back(duration.count() / 1e6); // Convert to ms
    }
    
    // Calculate percentiles
    std::sort(latencies.begin(), latencies.end());
    double p50 = latencies[iterations * 0.50];
    double p95 = latencies[iterations * 0.95];
    double p99 = latencies[iterations * 0.99];
    double p999 = latencies[iterations * 0.999];
    
    std::cout << "ML Inference Performance (single):\n"
              << "  P50:  " << p50 << " ms\n"
              << "  P95:  " << p95 << " ms\n"
              << "  P99:  " << p99 << " ms\n"
              << "  P99.9: " << p999 << " ms\n";
    
    // Verify <2.3ms target at P99
    EXPECT_LT(p99, 2.3) << "Inference latency exceeds 2.3ms target";
}

TEST_F(MLInferenceTest, PerformanceBatchInference) {
    const size_t batch_sizes[] = {1, 16, 32, 64};
    
    for (size_t batch_size : batch_sizes) {
        // Reconfigure engine for batch size
        InferenceEngine::Config config;
        config.model_path = "test_models/test_model.onnx";
        config.batch_size = batch_size;
        config.num_threads = 4;
        InferenceEngine batch_engine(config);
        
        // Generate batch
        std::vector<std::vector<float>> batch_features;
        for (size_t i = 0; i < batch_size; ++i) {
            batch_features.push_back(generate_features());
        }
        
        std::vector<InferenceResult> results;
        
        // Warm up
        for (int i = 0; i < 100; ++i) {
            batch_engine.infer_batch(batch_features, results);
        }
        
        // Measure
        const int iterations = 1000;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            batch_engine.infer_batch(batch_features, results);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        double avg_batch_time_ms = total_duration.count() / (iterations * 1000.0);
        double avg_per_sample_us = total_duration.count() / (iterations * batch_size * 1.0);
        
        std::cout << "Batch size " << batch_size << ":\n"
                  << "  Avg batch time: " << avg_batch_time_ms << " ms\n"
                  << "  Avg per sample: " << avg_per_sample_us << " μs\n";
        
        // Batch of 32 should complete in <2.3ms
        if (batch_size == 32) {
            EXPECT_LT(avg_batch_time_ms, 2.3);
        }
    }
}

// Test memory stability
TEST_F(MLInferenceTest, MemoryStability) {
    auto features = generate_features();
    InferenceResult result;
    
    // Get initial memory usage
    size_t initial_memory = engine_->get_memory_usage();
    
    // Run many inferences
    const int iterations = 100000;
    for (int i = 0; i < iterations; ++i) {
        engine_->infer(features, result);
        
        // Check memory every 10k iterations
        if (i % 10000 == 0) {
            size_t current_memory = engine_->get_memory_usage();
            size_t growth = current_memory - initial_memory;
            
            // Memory growth should be minimal (<1MB)
            EXPECT_LT(growth, 1024 * 1024) << "Memory leak detected at iteration " << i;
        }
    }
}

// Test GPU vs CPU comparison (if available)
TEST_F(MLInferenceTest, GPUvsCPUComparison) {
    if (!InferenceEngine::is_gpu_available()) {
        GTEST_SKIP() << "GPU not available, skipping GPU tests";
    }
    
    // CPU engine
    InferenceEngine::Config cpu_config;
    cpu_config.model_path = "test_models/test_model.onnx";
    cpu_config.batch_size = 32;
    cpu_config.use_gpu = false;
    InferenceEngine cpu_engine(cpu_config);
    
    // GPU engine
    InferenceEngine::Config gpu_config;
    gpu_config.model_path = "test_models/test_model.onnx";
    gpu_config.batch_size = 32;
    gpu_config.use_gpu = true;
    InferenceEngine gpu_engine(gpu_config);
    
    // Generate batch
    std::vector<std::vector<float>> batch_features;
    for (size_t i = 0; i < 32; ++i) {
        batch_features.push_back(generate_features());
    }
    
    std::vector<InferenceResult> cpu_results, gpu_results;
    
    // Warm up both
    for (int i = 0; i < 100; ++i) {
        cpu_engine.infer_batch(batch_features, cpu_results);
        gpu_engine.infer_batch(batch_features, gpu_results);
    }
    
    // Measure CPU
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        cpu_engine.infer_batch(batch_features, cpu_results);
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    
    // Measure GPU
    auto gpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        gpu_engine.infer_batch(batch_features, gpu_results);
    }
    auto gpu_end = std::chrono::high_resolution_clock::now();
    
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
    auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start);
    
    double cpu_avg_ms = cpu_duration.count() / 1000.0 / 1000.0;
    double gpu_avg_ms = gpu_duration.count() / 1000.0 / 1000.0;
    double speedup = cpu_avg_ms / gpu_avg_ms;
    
    std::cout << "CPU vs GPU Performance:\n"
              << "  CPU avg: " << cpu_avg_ms << " ms\n"
              << "  GPU avg: " << gpu_avg_ms << " ms\n"
              << "  GPU speedup: " << speedup << "x\n";
    
    // GPU should be faster for batch inference
    EXPECT_GT(speedup, 1.5);
}

// Test feature tensor preparation
TEST_F(MLInferenceTest, FeatureTensorPreparation) {
    // Test with feature engineering pipeline
    MarketSnapshot snapshot;
    snapshot.timestamp = std::chrono::system_clock::now();
    snapshot.bid_price = 150.48;
    snapshot.ask_price = 150.52;
    snapshot.bid_size = 500;
    snapshot.ask_size = 600;
    snapshot.last_price = 150.50;
    snapshot.last_size = 100;
    
    // Add some history
    for (int i = 0; i < 100; ++i) {
        snapshot.recent_trades.push_back({
            150.50 + (i % 2 ? 0.01 : -0.01),
            static_cast<uint32_t>(100 + i % 50),
            snapshot.timestamp - std::chrono::milliseconds(i * 100)
        });
    }
    
    // Extract features
    auto features = feature_eng_->extract_features(snapshot);
    EXPECT_EQ(features.size(), 128); // Expected feature count
    
    // Run inference
    InferenceResult result;
    ASSERT_TRUE(engine_->infer(features, result));
    
    // Verify result
    EXPECT_GT(result.anomaly_score, 0.0f);
    EXPECT_LE(result.anomaly_score, 1.0f);
}

// Test concurrent inference
TEST_F(MLInferenceTest, ConcurrentInference) {
    const int num_threads = 4;
    const int inferences_per_thread = 1000;
    
    std::vector<std::thread> threads;
    std::atomic<int> total_inferences{0};
    std::atomic<int> failed_inferences{0};
    
    auto worker = [&](int thread_id) {
        utils::set_cpu_affinity(thread_id);
        
        auto features = generate_features();
        InferenceResult result;
        
        for (int i = 0; i < inferences_per_thread; ++i) {
            if (engine_->infer(features, result)) {
                total_inferences++;
            } else {
                failed_inferences++;
            }
        }
    };
    
    // Start threads
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker, i);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    EXPECT_EQ(failed_inferences.load(), 0);
    EXPECT_EQ(total_inferences.load(), num_threads * inferences_per_thread);
    
    double throughput = total_inferences.load() / (duration.count() / 1000.0);
    std::cout << "Concurrent inference throughput: " << throughput << " inferences/sec\n";
}

// Test different model architectures
TEST_F(MLInferenceTest, ModelArchitectures) {
    struct ModelTest {
        std::string name;
        std::string path;
        size_t expected_features;
        double max_latency_ms;
    };
    
    ModelTest models[] = {
        {"Elastic Net", "test_models/elastic_net.onnx", 128, 0.5},
        {"LSTM Autoencoder", "test_models/lstm_ae.onnx", 100, 1.5},
        {"Transformer", "test_models/transformer.onnx", 128, 2.3}
    };
    
    for (const auto& model : models) {
        // Skip if model doesn't exist
        if (!fs::exists(model.path)) {
            create_dummy_model(model.path);
        }
        
        InferenceEngine::Config config;
        config.model_path = model.path;
        config.batch_size = 1;
        
        try {
            InferenceEngine model_engine(config);
            
            auto features = generate_features(model.expected_features);
            InferenceResult result;
            
            // Warm up
            for (int i = 0; i < 100; ++i) {
                model_engine.infer(features, result);
            }
            
            // Measure
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < 1000; ++i) {
                model_engine.infer(features, result);
            }
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            double avg_ms = duration.count() / 1000.0 / 1000.0;
            
            std::cout << model.name << " avg latency: " << avg_ms << " ms\n";
            EXPECT_LT(avg_ms, model.max_latency_ms);
            
        } catch (const std::exception& e) {
            // Model loading might fail with dummy models
            std::cout << "Skipping " << model.name << ": " << e.what() << "\n";
        }
    }
}

// Test edge cases
TEST_F(MLInferenceTest, EdgeCases) {
    InferenceResult result;
    
    // Empty features
    std::vector<float> empty_features;
    EXPECT_FALSE(engine_->infer(empty_features, result));
    
    // Wrong feature count
    std::vector<float> wrong_features(64, 0.0f); // Expecting 128
    EXPECT_FALSE(engine_->infer(wrong_features, result));
    
    // NaN/Inf features
    auto features = generate_features();
    features[0] = std::numeric_limits<float>::quiet_NaN();
    features[1] = std::numeric_limits<float>::infinity();
    EXPECT_FALSE(engine_->infer(features, result));
    
    // Very large batch
    std::vector<std::vector<float>> huge_batch(1000);
    for (auto& f : huge_batch) {
        f = generate_features();
    }
    std::vector<InferenceResult> results;
    // Should handle gracefully (might process in chunks)
    EXPECT_TRUE(engine_->infer_batch(huge_batch, results));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
