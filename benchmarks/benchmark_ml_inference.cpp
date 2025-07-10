#include <benchmark/benchmark.h>
#include <chrono>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <memory>

#include "darkpool/ml/inference_engine.hpp"
#include "darkpool/ml/feature_engineering.hpp"
#include "darkpool/ml/elastic_net.hpp"
#include "darkpool/ml/lstm_autoencoder.hpp"
#include "darkpool/ml/transformer_model.hpp"
#include "darkpool/utils/cpu_affinity.hpp"

using namespace darkpool;
using namespace darkpool::ml;

// Latency statistics
class LatencyStats {
public:
    void add(double ms) {
        samples_.push_back(ms);
    }
    
    void analyze(const std::string& name, double target_ms = 2.3) {
        if (samples_.empty()) return;
        
        std::sort(samples_.begin(), samples_.end());
        
        double p50 = percentile(0.50);
        double p90 = percentile(0.90);
        double p95 = percentile(0.95);
        double p99 = percentile(0.99);
        double p999 = percentile(0.999);
        
        double mean = std::accumulate(samples_.begin(), samples_.end(), 0.0) / samples_.size();
        
        std::cout << "\n" << name << " Latency Analysis:\n";
        std::cout << "Samples: " << samples_.size() << "\n";
        std::cout << "Mean:    " << std::fixed << std::setprecision(3) << mean << " ms\n";
        std::cout << "P50:     " << p50 << " ms\n";
        std::cout << "P90:     " << p90 << " ms\n";
        std::cout << "P95:     " << p95 << " ms\n";
        std::cout << "P99:     " << p99 << " ms\n";
        std::cout << "P99.9:   " << p999 << " ms\n";
        
        if (p99 > target_ms) {
            std::cout << "WARNING: P99 latency " << p99 
                      << " ms exceeds " << target_ms << "ms target!\n";
        } else {
            std::cout << "SUCCESS: P99 latency " << p99 
                      << " ms meets <" << target_ms << "ms target!\n";
        }
    }
    
private:
    double percentile(double p) {
        size_t idx = static_cast<size_t>(samples_.size() * p);
        return samples_[std::min(idx, samples_.size() - 1)];
    }
    
    std::vector<double> samples_;
};

// Feature generator
std::vector<float> generate_features(size_t count = 128) {
    static std::mt19937 gen(42);
    static std::normal_distribution<float> dist(0.0f, 1.0f);
    
    std::vector<float> features(count);
    
    // Microstructure features
    features[0] = 150.50f;  // Price
    features[1] = 0.02f;    // Spread
    features[2] = 100000.0f; // Volume
    features[3] = 0.15f;    // Volatility
    features[4] = 0.55f;    // Buy ratio
    
    // Fill rest with random values
    for (size_t i = 5; i < count; ++i) {
        features[i] = dist(gen);
    }
    
    return features;
}

// Benchmark single inference
static void BM_SingleInference(benchmark::State& state) {
    InferenceEngine::Config config;
    config.batch_size = 1;
    config.num_threads = 1;
    config.use_gpu = false;
    config.enable_profiling = true;
    
    auto engine = std::make_unique<InferenceEngine>(config);
    auto features = generate_features();
    
    LatencyStats stats;
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        InferenceResult result;
        engine->infer(features, result);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end - start);
        
        stats.add(duration.count());
        benchmark::DoNotOptimize(result);
    }
    
    stats.analyze("Single Inference");
    state.SetLabel("Batch size: 1");
}

// Benchmark batch inference
static void BM_BatchInference(benchmark::State& state) {
    const size_t batch_size = state.range(0);
    
    InferenceEngine::Config config;
    config.batch_size = batch_size;
    config.num_threads = 4;
    config.use_gpu = false;
    
    auto engine = std::make_unique<InferenceEngine>(config);
    
    // Generate batch
    std::vector<std::vector<float>> batch;
    for (size_t i = 0; i < batch_size; ++i) {
        batch.push_back(generate_features());
    }
    
    LatencyStats stats;
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<InferenceResult> results;
        engine->infer_batch(batch, results);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end - start);
        
        stats.add(duration.count());
        benchmark::DoNotOptimize(results);
    }
    
    stats.analyze("Batch Inference (size " + std::to_string(batch_size) + ")");
    state.SetLabel("Batch size: " + std::to_string(batch_size));
}

// Benchmark INT8 quantization
static void BM_INT8Quantization(benchmark::State& state) {
    const size_t batch_size = state.range(0);
    
    // FP32 engine
    InferenceEngine::Config fp32_config;
    fp32_config.batch_size = batch_size;
    fp32_config.use_int8_quantization = false;
    auto fp32_engine = std::make_unique<InferenceEngine>(fp32_config);
    
    // INT8 engine
    InferenceEngine::Config int8_config;
    int8_config.batch_size = batch_size;
    int8_config.use_int8_quantization = true;
    auto int8_engine = std::make_unique<InferenceEngine>(int8_config);
    
    // Generate batch
    std::vector<std::vector<float>> batch;
    for (size_t i = 0; i < batch_size; ++i) {
        batch.push_back(generate_features());
    }
    
    LatencyStats fp32_stats, int8_stats;
    
    for (auto _ : state) {
        // Benchmark FP32
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<InferenceResult> fp32_results;
        fp32_engine->infer_batch(batch, fp32_results);
        auto end = std::chrono::high_resolution_clock::now();
        fp32_stats.add(std::chrono::duration<double, std::milli>(end - start).count());
        
        // Benchmark INT8
        start = std::chrono::high_resolution_clock::now();
        std::vector<InferenceResult> int8_results;
        int8_engine->infer_batch(batch, int8_results);
        end = std::chrono::high_resolution_clock::now();
        int8_stats.add(std::chrono::duration<double, std::milli>(end - start).count());
        
        benchmark::DoNotOptimize(fp32_results);
        benchmark::DoNotOptimize(int8_results);
    }
    
    fp32_stats.analyze("FP32 Inference");
    int8_stats.analyze("INT8 Inference");
}

// Benchmark feature extraction overhead
static void BM_FeatureExtraction(benchmark::State& state) {
    auto feature_eng = std::make_unique<FeatureEngineering>();
    
    // Create market snapshot
    MarketSnapshot snapshot;
    snapshot.timestamp = std::chrono::system_clock::now();
    snapshot.bid_price = 150.48;
    snapshot.ask_price = 150.52;
    snapshot.bid_size = 500;
    snapshot.ask_size = 600;
    snapshot.last_price = 150.50;
    snapshot.last_size = 100;
    
    // Add trade history
    for (int i = 0; i < 1000; ++i) {
        snapshot.recent_trades.push_back({
            150.50 + (i % 2 ? 0.01 : -0.01),
            static_cast<uint32_t>(100 + i % 50),
            snapshot.timestamp - std::chrono::milliseconds(i * 100)
        });
    }
    
    LatencyStats stats;
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto features = feature_eng->extract_features(snapshot);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::micro>(end - start);
        
        stats.add(duration.count() / 1000.0); // Convert to ms
        benchmark::DoNotOptimize(features);
    }
    
    stats.analyze("Feature Extraction", 0.1); // 100μs target
}

// Benchmark different model architectures
static void BM_ModelArchitectures(benchmark::State& state) {
    const std::string model_name = state.label();
    
    std::unique_ptr<ModelBase> model;
    
    if (model_name == "ElasticNet") {
        model = std::make_unique<ElasticNet>();
    } else if (model_name == "LSTM") {
        model = std::make_unique<LSTMAutoencoder>(100, 50, 2);
    } else if (model_name == "Transformer") {
        model = std::make_unique<TransformerModel>(128, 8, 4, 512);
    }
    
    auto features = generate_features(model->get_input_size());
    LatencyStats stats;
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto output = model->forward(features);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end - start);
        
        stats.add(duration.count());
        benchmark::DoNotOptimize(output);
    }
    
    stats.analyze(model_name + " Model");
}

// Benchmark GPU vs CPU (if available)
static void BM_GPUvsCPU(benchmark::State& state) {
    if (!InferenceEngine::is_gpu_available()) {
        state.SkipWithError("GPU not available");
        return;
    }
    
    const size_t batch_size = state.range(0);
    
    // CPU engine
    InferenceEngine::Config cpu_config;
    cpu_config.batch_size = batch_size;
    cpu_config.use_gpu = false;
    auto cpu_engine = std::make_unique<InferenceEngine>(cpu_config);
    
    // GPU engine
    InferenceEngine::Config gpu_config;
    gpu_config.batch_size = batch_size;
    gpu_config.use_gpu = true;
    auto gpu_engine = std::make_unique<InferenceEngine>(gpu_config);
    
    // Generate batch
    std::vector<std::vector<float>> batch;
    for (size_t i = 0; i < batch_size; ++i) {
        batch.push_back(generate_features());
    }
    
    LatencyStats cpu_stats, gpu_stats;
    
    for (auto _ : state) {
        // CPU inference
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<InferenceResult> cpu_results;
        cpu_engine->infer_batch(batch, cpu_results);
        auto end = std::chrono::high_resolution_clock::now();
        cpu_stats.add(std::chrono::duration<double, std::milli>(end - start).count());
        
        // GPU inference
        start = std::chrono::high_resolution_clock::now();
        std::vector<InferenceResult> gpu_results;
        gpu_engine->infer_batch(batch, gpu_results);
        end = std::chrono::high_resolution_clock::now();
        gpu_stats.add(std::chrono::duration<double, std::milli>(end - start).count());
        
        benchmark::DoNotOptimize(cpu_results);
        benchmark::DoNotOptimize(gpu_results);
    }
    
    cpu_stats.analyze("CPU Inference");
    gpu_stats.analyze("GPU Inference");
}

// Benchmark memory allocation
static void BM_MemoryAllocation(benchmark::State& state) {
    const size_t batch_size = state.range(0);
    
    InferenceEngine::Config config;
    config.batch_size = batch_size;
    config.preallocate_buffers = true;
    
    auto engine = std::make_unique<InferenceEngine>(config);
    
    // Track memory usage
    size_t initial_memory = engine->get_memory_usage();
    
    std::vector<std::vector<float>> batch;
    for (size_t i = 0; i < batch_size; ++i) {
        batch.push_back(generate_features());
    }
    
    size_t iterations = 0;
    
    for (auto _ : state) {
        std::vector<InferenceResult> results;
        engine->infer_batch(batch, results);
        
        iterations++;
        
        // Check memory growth every 1000 iterations
        if (iterations % 1000 == 0) {
            size_t current_memory = engine->get_memory_usage();
            size_t growth = current_memory - initial_memory;
            
            if (growth > 1024 * 1024) { // 1MB growth
                state.SkipWithError("Memory leak detected");
                return;
            }
        }
        
        benchmark::DoNotOptimize(results);
    }
    
    state.SetLabel("Batch size: " + std::to_string(batch_size));
}

// Benchmark model loading time
static void BM_ModelLoading(benchmark::State& state) {
    const std::string model_path = "models/darkpool_detector.onnx";
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        InferenceEngine::Config config;
        config.model_path = model_path;
        auto engine = std::make_unique<InferenceEngine>(config);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end - start);
        
        state.SetIterationTime(duration.count() / 1000.0); // Convert to seconds
        benchmark::DoNotOptimize(engine);
    }
    
    state.SetLabel("Model loading time");
}

// End-to-end ML pipeline benchmark
static void BM_EndToEndMLPipeline(benchmark::State& state) {
    auto feature_eng = std::make_unique<FeatureEngineering>();
    
    InferenceEngine::Config config;
    config.batch_size = 32;
    auto engine = std::make_unique<InferenceEngine>(config);
    
    // Create market snapshots
    std::vector<MarketSnapshot> snapshots;
    for (int i = 0; i < 32; ++i) {
        MarketSnapshot snapshot;
        snapshot.timestamp = std::chrono::system_clock::now();
        snapshot.bid_price = 150.00 + i * 0.01;
        snapshot.ask_price = 150.02 + i * 0.01;
        snapshot.bid_size = 1000;
        snapshot.ask_size = 1000;
        snapshot.last_price = 150.01 + i * 0.01;
        snapshot.last_size = 100;
        snapshots.push_back(snapshot);
    }
    
    LatencyStats stats;
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Extract features
        std::vector<std::vector<float>> features;
        for (const auto& snapshot : snapshots) {
            features.push_back(feature_eng->extract_features(snapshot));
        }
        
        // Run inference
        std::vector<InferenceResult> results;
        engine->infer_batch(features, results);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end - start);
        
        stats.add(duration.count());
        benchmark::DoNotOptimize(results);
    }
    
    stats.analyze("End-to-End ML Pipeline", 2.3);
}

// Custom main
int main(int argc, char** argv) {
    std::cout << "Dark Pool Detector - ML Inference Benchmark\n";
    std::cout << "===========================================\n";
    std::cout << "Target: <2.3ms inference latency at P99\n\n";
    
    // Pin to CPU
    utils::set_cpu_affinity(0);
    
    // Initialize benchmark
    ::benchmark::Initialize(&argc, argv);
    
    // Register benchmarks
    BENCHMARK(BM_SingleInference)->Iterations(1000);
    BENCHMARK(BM_BatchInference)->Range(1, 128)->Iterations(100);
    BENCHMARK(BM_INT8Quantization)->Range(16, 64)->Iterations(100);
    BENCHMARK(BM_FeatureExtraction)->Iterations(10000);
    
    // Model architecture benchmarks
    benchmark::RegisterBenchmark("BM_ModelArchitectures", BM_ModelArchitectures)
        ->Iterations(1000)->SetLabel("ElasticNet");
    benchmark::RegisterBenchmark("BM_ModelArchitectures", BM_ModelArchitectures)
        ->Iterations(100)->SetLabel("LSTM");
    benchmark::RegisterBenchmark("BM_ModelArchitectures", BM_ModelArchitectures)
        ->Iterations(100)->SetLabel("Transformer");
    
    BENCHMARK(BM_GPUvsCPU)->Range(1, 64)->Iterations(100);
    BENCHMARK(BM_MemoryAllocation)->Range(1, 128)->Iterations(1000);
    BENCHMARK(BM_ModelLoading)->Iterations(10)->UseManualTime();
    BENCHMARK(BM_EndToEndMLPipeline)->Iterations(1000);
    
    // Run benchmarks
    ::benchmark::RunSpecifiedBenchmarks();
    
    std::cout << "\n===========================================\n";
    std::cout << "ML INFERENCE BENCHMARK SUMMARY\n";
    std::cout << "===========================================\n";
    std::cout << "Target: <2.3ms inference latency\n";
    std::cout << "Check analysis above for PASS/FAIL status\n";
    std::cout << "===========================================\n";
    
    return 0;
}
