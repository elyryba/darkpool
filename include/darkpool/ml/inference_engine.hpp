#pragma once

#include <memory>
#include <vector>
#include <string>
#include <atomic>
#include <optional>
#include <chrono>

// Stub ONNX types for CI
namespace Ort {
    class Env {};
    class Session {};
    class SessionOptions {};
    class MemoryInfo {};
    class Value {};
    class RunOptions {};
}

namespace darkpool::ml {

class InferenceEngine {
public:
    struct Config {
        std::string model_path;
        size_t batch_size = 1;
        size_t max_sequence_length = 1000;
        bool use_gpu = false;
        int gpu_device_id = 0;
        size_t num_threads = 1;
        bool enable_profiling = false;
        bool use_int8_quantization = false;
        std::string cache_dir = "/tmp/darkpool_ml_cache";
    };

    struct PredictionResult {
        std::vector<float> anomaly_scores;
        std::vector<float> confidence_scores;
        std::vector<int> classifications;
        std::chrono::microseconds inference_time;
        size_t batch_size;
    };

    explicit InferenceEngine(const Config& config);
    ~InferenceEngine();

    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;
    InferenceEngine(InferenceEngine&&) noexcept;
    InferenceEngine& operator=(InferenceEngine&&) noexcept;

    PredictionResult predict(const std::vector<std::vector<float>>& features);
    void warmup(size_t num_iterations = 10);
    bool is_ready() const noexcept;

private:
    struct InferenceContext;
    std::unique_ptr<InferenceContext> ctx_;
};

}  // namespace darkpool::ml
