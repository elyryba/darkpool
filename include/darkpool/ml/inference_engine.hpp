#pragma once

#include "darkpool/types.hpp"
#include "darkpool/ml/feature_engineering.hpp"
#include <memory>
#include <vector>
#include <string>
#include <atomic>
#include <unordered_map>

// Forward declarations for ONNX Runtime
namespace Ort {
    class Session;
    class Env;
    class SessionOptions;
    class MemoryInfo;
    class Value;
    class AllocatorWithDefaultOptions;
    class RunOptions;
}

namespace darkpool::ml {

class InferenceEngine {
public:
    static constexpr size_t MAX_BATCH_SIZE = 256;
    static constexpr size_t TENSOR_CACHE_SIZE = 16;
    
    struct Config {
        std::string model_path;
        size_t batch_size = 32;
        size_t max_batch_wait_us = 100;
        bool enable_gpu = false;
        int gpu_device_id = 0;
        bool enable_tensorrt = false;
        bool enable_quantization = true; // INT8
        size_t num_threads = 4;
        size_t num_inter_threads = 2;
        bool enable_profiling = false;
        bool enable_memory_pattern = true;
        bool enable_cpu_mem_arena = true;
        size_t tensor_cache_size = TENSOR_CACHE_SIZE;
        double confidence_threshold = 0.7;
    };
    
    struct Prediction {
        alignas(64) struct {
            double dark_pool_probability;
            double confidence;
            double hidden_size_estimate;
            std::array<double, 5> venue_probabilities; // Top 5 venues
            AnomalyType detected_type;
            double execution_urgency;
            double market_impact_bps;
            uint32_t symbol;
            int64_t timestamp;
        } data;
        
        // Model metadata
        std::string model_version;
        int64_t inference_time_us;
        size_t batch_position;
        
        bool is_dark_pool() const noexcept { 
            return data.dark_pool_probability > 0.5; 
        }
        
        bool is_high_confidence() const noexcept {
            return data.confidence > 0.8;
        }
    };
    
    struct BatchRequest {
        alignas(64) std::vector<FeatureEngineering::Features> features;
        std::vector<uint32_t> symbols;
        std::vector<int64_t> timestamps;
        size_t count = 0;
        int64_t oldest_timestamp = 0;
        std::atomic<bool> ready{false};
    };
    
    struct ModelStats {
        std::atomic<uint64_t> total_inferences{0};
        std::atomic<uint64_t> total_batches{0};
        std::atomic<uint64_t> total_latency_us{0};
        std::atomic<uint64_t> max_latency_us{0};
        std::atomic<uint64_t> cache_hits{0};
        std::atomic<uint64_t> cache_misses{0};
        std::atomic<double> avg_batch_size{0.0};
        std::atomic<double> avg_confidence{0.0};
        std::atomic<uint64_t> high_confidence_predictions{0};
        std::atomic<uint64_t> dark_pool_detections{0};
    };
    
    explicit InferenceEngine(const Config& config);
    ~InferenceEngine();
    
    // Single inference (may batch internally)
    Prediction infer(const FeatureEngineering::Features& features) noexcept;
    
    // Batch inference (more efficient)
    std::vector<Prediction> infer_batch(
        const std::vector<FeatureEngineering::Features>& features) noexcept;
    
    // Async inference with callback
    using InferenceCallback = std::function<void(const Prediction&)>;
    void infer_async(const FeatureEngineering::Features& features,
                    InferenceCallback callback) noexcept;
    
    // Model management
    bool load_model(const std::string& path) noexcept;
    bool is_ready() const noexcept;
    std::string get_model_version() const noexcept;
    
    // Performance monitoring
    ModelStats get_stats() const noexcept;
    void reset_stats() noexcept;
    
    // Warm up model with dummy data
    void warmup(size_t iterations = 100) noexcept;
    
private:
    struct TensorCache {
        alignas(64) struct CacheEntry {
            std::unique_ptr<Ort::Value> tensor;
            size_t size;
            std::atomic<bool> in_use{false};
            int64_t last_used_ns;
        };
        
        std::array<CacheEntry, TENSOR_CACHE_SIZE> entries;
        std::atomic<size_t> next_slot{0};
        
        Ort::Value* acquire(size_t size) noexcept;
        void release(Ort::Value* tensor) noexcept;
    };
    
    struct InferenceContext {
        // ONNX Runtime objects
        std::unique_ptr<Ort::Env> env;
        std::unique_ptr<Ort::SessionOptions> session_options;
        std::unique_ptr<Ort::Session> session;
        std::unique_ptr<Ort::MemoryInfo> memory_info;
        std::unique_ptr<Ort::AllocatorWithDefaultOptions> allocator;
        std::unique_ptr<Ort::RunOptions> run_options;
        
        // Model I/O info
        std::vector<std::string> input_names;
        std::vector<std::vector<int64_t>> input_shapes;
        std::vector<std::string> output_names;
        std::vector<std::vector<int64_t>> output_shapes;
        
        // Pre-allocated tensors
        TensorCache tensor_cache;
        
        // Batch accumulator
        std::unique_ptr<BatchRequest> current_batch;
        std::mutex batch_mutex;
        std::condition_variable batch_cv;
        
        bool initialize(const Config& config) noexcept;
        void cleanup() noexcept;
    };
    
    // Configuration
    Config config_;
    
    // Model context
    std::unique_ptr<InferenceContext> context_;
    
    // Threading
    std::atomic<bool> running_{false};
    std::thread batch_thread_;
    
    // Statistics
    mutable ModelStats stats_;
    
    // Model metadata
    std::string model_version_;
    std::atomic<bool> model_loaded_{false};
    
    // Internal methods
    void batch_processing_loop() noexcept;
    std::vector<Prediction> run_inference(const BatchRequest& batch) noexcept;
    Prediction process_output(const float* output_data, size_t idx,
                            const BatchRequest& batch,
                            int64_t inference_time_us) noexcept;
    
    // Tensor preparation
    std::unique_ptr<Ort::Value> create_input_tensor(
        const std::vector<FeatureEngineering::Features>& features) noexcept;
    
    // INT8 quantization
    void quantize_features(const float* input, int8_t* output, 
                          size_t count) noexcept;
    void dequantize_output(const int8_t* input, float* output,
                          size_t count) noexcept;
    
    // Performance optimization
    void optimize_session_options(Ort::SessionOptions* options) noexcept;
    void setup_tensorrt_provider(Ort::SessionOptions* options) noexcept;
    void setup_cuda_provider(Ort::SessionOptions* options) noexcept;
    
    // Error handling
    void handle_onnx_error(const std::exception& e) noexcept;
};

// Global inference engine instance (singleton pattern)
InferenceEngine& get_inference_engine();

// Helper to check if a model file exists and is valid
bool validate_model_file(const std::string& path) noexcept;

} 
