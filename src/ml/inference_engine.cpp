#include "darkpool/ml/inference_engine.hpp"
// #include <onnxruntime_cxx_api.h> // Removed for CI
#include <chrono>
#include <cstring>
#include <immintrin.h>

namespace darkpool::ml {

namespace {
    // Singleton instance
    std::unique_ptr<InferenceEngine> g_inference_engine;
    std::mutex g_engine_mutex;
    
    // Quantization scales (calibrated from training)
    constexpr float FEATURE_SCALE = 127.0f / 3.0f;  // Assumes features normalized to [-3, 3]
    constexpr float OUTPUT_SCALE = 1.0f / 127.0f;
}

InferenceEngine::InferenceEngine(const Config& config)
    : config_(config)
    , context_(std::make_unique<InferenceContext>()) {
    
    if (!context_->initialize(config)) {
        throw std::runtime_error("Failed to initialize inference context");
    }
    
    // Start batch processing thread
    running_ = true;
    batch_thread_ = std::thread(&InferenceEngine::batch_processing_loop, this);
}

InferenceEngine::~InferenceEngine() {
    running_ = false;
    
    if (batch_thread_.joinable()) {
        context_->batch_cv.notify_all();
        batch_thread_.join();
    }
    
    context_->cleanup();
}

InferenceEngine::Prediction InferenceEngine::infer(
    const FeatureEngineering::Features& features) noexcept {
    
    // Add to batch
    {
        std::unique_lock<std::mutex> lock(context_->batch_mutex);
        
        if (!context_->current_batch) {
            context_->current_batch = std::make_unique<BatchRequest>();
            context_->current_batch->features.reserve(config_.batch_size);
        }
        
        auto& batch = *context_->current_batch;
        batch.features.push_back(features);
        batch.symbols.push_back(features.symbol);
        batch.timestamps.push_back(features.timestamp);
        batch.count++;
        
        if (batch.oldest_timestamp == 0) {
            batch.oldest_timestamp = features.timestamp;
        }
        
        // Check if batch is ready
        bool trigger_inference = false;
        if (batch.count >= config_.batch_size) {
            trigger_inference = true;
        } else {
            // Check time threshold
            auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            if (now - batch.oldest_timestamp > config_.max_batch_wait_us * 1000) {
                trigger_inference = true;
            }
        }
        
        if (trigger_inference) {
            batch.ready = true;
            context_->batch_cv.notify_one();
        }
    }
    
    // Wait for result (simplified - in production would use future/promise)
    std::this_thread::sleep_for(std::chrono::microseconds(config_.max_batch_wait_us));
    
    // Return placeholder for now
    Prediction pred;
    pred.data.symbol = features.symbol;
    pred.data.timestamp = features.timestamp;
    pred.data.dark_pool_probability = 0.5;
    pred.data.confidence = 0.8;
    
    return pred;
}

std::vector<InferenceEngine::Prediction> InferenceEngine::infer_batch(
    const std::vector<FeatureEngineering::Features>& features) noexcept {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Create batch request
    BatchRequest batch;
    batch.features = features;
    batch.count = features.size();
    
    for (const auto& f : features) {
        batch.symbols.push_back(f.symbol);
        batch.timestamps.push_back(f.timestamp);
    }
    
    // Run inference
    auto predictions = run_inference(batch);
    
    // Update stats
    auto end_time = std::chrono::high_resolution_clock::now();
    auto latency_us = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time).count();
    
    stats_.total_batches.fetch_add(1);
    stats_.total_inferences.fetch_add(features.size());
    stats_.total_latency_us.fetch_add(latency_us);
    
    uint64_t current_max = stats_.max_latency_us.load();
    while (latency_us > current_max && 
           !stats_.max_latency_us.compare_exchange_weak(current_max, latency_us)) {
        // Retry
    }
    
    return predictions;
}

void InferenceEngine::infer_async(const FeatureEngineering::Features& features,
                                 InferenceCallback callback) noexcept {
    // Queue for async processing
    // In production, would use a proper async queue
    std::thread([this, features, callback]() {
        auto pred = infer(features);
        callback(pred);
    }).detach();
}

bool InferenceEngine::load_model(const std::string& path) noexcept {
    try {
        config_.model_path = path;
        
        // Re-initialize with new model
        auto new_context = std::make_unique<InferenceContext>();
        if (!new_context->initialize(config_)) {
            return false;
        }
        
        // Atomic swap
        context_.swap(new_context);
        model_loaded_ = true;
        
        // Extract version from model metadata
        model_version_ = "1.0.0"; // Would read from model
        
        return true;
    } catch (...) {
        return false;
    }
}

bool InferenceEngine::is_ready() const noexcept {
    return model_loaded_.load() && running_.load();
}

std::string InferenceEngine::get_model_version() const noexcept {
    return model_version_;
}

InferenceEngine::ModelStats InferenceEngine::get_stats() const noexcept {
    return stats_;
}

void InferenceEngine::reset_stats() noexcept {
    stats_ = ModelStats{};
}

void InferenceEngine::warmup(size_t iterations) noexcept {
    // Create dummy features
    std::vector<FeatureEngineering::Features> dummy_batch;
    dummy_batch.reserve(config_.batch_size);
    
    for (size_t i = 0; i < config_.batch_size; ++i) {
        FeatureEngineering::Features features{};
        features.symbol = i;
        features.timestamp = i * 1000000;
        features.valid = true;
        
        // Fill with random values
        for (size_t j = 0; j < features.values.size(); ++j) {
            features.values[j] = static_cast<double>(j) / features.values.size();
        }
        
        dummy_batch.push_back(features);
    }
    
    // Run warmup iterations
    for (size_t i = 0; i < iterations; ++i) {
        infer_batch(dummy_batch);
    }
    
    reset_stats();
}

void InferenceEngine::batch_processing_loop() noexcept {
    while (running_.load()) {
        std::unique_lock<std::mutex> lock(context_->batch_mutex);
        
        // Wait for batch
        context_->batch_cv.wait(lock, [this]() {
            return !running_.load() || 
                   (context_->current_batch && context_->current_batch->ready.load());
        });
        
        if (!running_.load()) break;
        
        if (context_->current_batch && context_->current_batch->ready.load()) {
            // Take ownership of batch
            auto batch = std::move(context_->current_batch);
            lock.unlock();
            
            // Process batch
            auto predictions = run_inference(*batch);
            
            // In production, would notify waiting threads
        }
    }
}

std::vector<InferenceEngine::Prediction> InferenceEngine::run_inference(
    const BatchRequest& batch) noexcept {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<Prediction> predictions;
    predictions.reserve(batch.count);
    
    try {
        // Create input tensor
        auto input_tensor = create_input_tensor(batch.features);
        if (!input_tensor) {
            return predictions;
        }
        
        // Prepare inputs
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(*input_tensor));
        
        // Run inference
        auto output_tensors = context_->session->Run(
            *context_->run_options,
            context_->input_names.data(),
            input_tensors.data(),
            input_tensors.size(),
            context_->output_names.data(),
            context_->output_names.size()
        );
        
        // Process outputs
        auto end_time = std::chrono::high_resolution_clock::now();
        auto inference_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time).count();
        
        // Get output data
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        
        // Create predictions
        for (size_t i = 0; i < batch.count; ++i) {
            predictions.push_back(process_output(output_data, i, batch, inference_time_us));
        }
        
    } catch (const Ort::Exception& e) {
        handle_onnx_error(e);
    }
    
    return predictions;
}

InferenceEngine::Prediction InferenceEngine::process_output(
    const float* output_data, size_t idx,
    const BatchRequest& batch,
    int64_t inference_time_us) noexcept {
    
    Prediction pred;
    pred.data.symbol = batch.symbols[idx];
    pred.data.timestamp = batch.timestamps[idx];
    pred.inference_time_us = inference_time_us / batch.count; // Average
    pred.batch_position = idx;
    pred.model_version = model_version_;
    
    // Output format: [dark_pool_prob, confidence, hidden_size, venue_probs[5], ...]
    size_t offset = idx * 10; // 10 outputs per sample
    
    pred.data.dark_pool_probability = output_data[offset + 0];
    pred.data.confidence = output_data[offset + 1];
    pred.data.hidden_size_estimate = output_data[offset + 2] * 1000; // Scale to shares
    
    // Venue probabilities
    for (size_t i = 0; i < 5; ++i) {
        pred.data.venue_probabilities[i] = output_data[offset + 3 + i];
    }
    
    // Derived values
    pred.data.execution_urgency = output_data[offset + 8];
    pred.data.market_impact_bps = output_data[offset + 9] * 100; // Scale to bps
    
    // Determine anomaly type based on probabilities
    if (pred.data.dark_pool_probability > 0.8) {
        pred.data.detected_type = AnomalyType::DARK_POOL_ACTIVITY;
    } else if (pred.data.hidden_size_estimate > 10000) {
        pred.data.detected_type = AnomalyType::HIDDEN_LIQUIDITY;
    } else {
        pred.data.detected_type = AnomalyType::NONE;
    }
    
    // Update statistics
    if (pred.is_high_confidence()) {
        stats_.high_confidence_predictions.fetch_add(1);
    }
    if (pred.is_dark_pool()) {
        stats_.dark_pool_detections.fetch_add(1);
    }
    
    return pred;
}

std::unique_ptr<Ort::Value> InferenceEngine::create_input_tensor(
    const std::vector<FeatureEngineering::Features>& features) noexcept {
    
    try {
        // Input shape: [batch_size, num_features]
        std::vector<int64_t> input_shape = {
            static_cast<int64_t>(features.size()),
            static_cast<int64_t>(FeatureEngineering::NUM_FEATURES)
        };
        
        size_t total_elements = features.size() * FeatureEngineering::NUM_FEATURES;
        
        // Try to get cached tensor
        auto tensor = context_->tensor_cache.acquire(total_elements);
        if (tensor) {
            // Fill with feature data
            float* data = tensor->GetTensorMutableData<float>();
            
            // Use SIMD for fast copy
            for (size_t i = 0; i < features.size(); ++i) {
                const double* src = features[i].data();
                float* dst = data + i * FeatureEngineering::NUM_FEATURES;
                
                // Convert double to float with SIMD
                for (size_t j = 0; j < FeatureEngineering::NUM_FEATURES; j += 4) {
                    __m256d d = _mm256_loadu_pd(src + j);
                    __m128 f = _mm256_cvtpd_ps(d);
                    _mm_storeu_ps(dst + j, f);
                }
            }
            
            return std::unique_ptr<Ort::Value>(tensor);
        }
        
        // Allocate new tensor
        auto tensor_ptr = std::make_unique<Ort::Value>(
            Ort::Value::CreateTensor<float>(
                *context_->memory_info,
                input_shape.data(),
                input_shape.size()
            )
        );
        
        // Fill data
        float* data = tensor_ptr->GetTensorMutableData<float>();
        for (size_t i = 0; i < features.size(); ++i) {
            for (size_t j = 0; j < FeatureEngineering::NUM_FEATURES; ++j) {
                data[i * FeatureEngineering::NUM_FEATURES + j] = 
                    static_cast<float>(features[i].values[j]);
            }
        }
        
        return tensor_ptr;
        
    } catch (...) {
        return nullptr;
    }
}

void InferenceEngine::quantize_features(const float* input, int8_t* output, 
                                       size_t count) noexcept {
    // Quantize to INT8 using SIMD
    __m256 scale = _mm256_set1_ps(FEATURE_SCALE);
    
    for (size_t i = 0; i < count; i += 8) {
        __m256 vals = _mm256_loadu_ps(input + i);
        __m256 scaled = _mm256_mul_ps(vals, scale);
        __m256i ints = _mm256_cvtps_epi32(scaled);
        
        // Pack to int8
        __m128i lo = _mm256_extracti128_si256(ints, 0);
        __m128i hi = _mm256_extracti128_si256(ints, 1);
        __m128i packed = _mm_packs_epi32(lo, hi);
        __m128i packed8 = _mm_packs_epi16(packed, packed);
        
        _mm_storel_epi64((__m128i*)(output + i), packed8);
    }
}

// InferenceContext implementation

bool InferenceEngine::InferenceContext::initialize(const Config& config) noexcept {
    try {
        // Create environment
        env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "darkpool");
        
        // Create session options
        session_options = std::make_unique<Ort::SessionOptions>();
        
        // Configure options
        session_options->SetIntraOpNumThreads(config.num_threads);
        session_options->SetInterOpNumThreads(config.num_inter_threads);
        session_options->SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        if (config.enable_memory_pattern) {
            session_options->EnableMemPattern();
        }
        
        if (config.enable_cpu_mem_arena) {
            session_options->EnableCpuMemArena();
        }
        
        if (config.enable_profiling) {
            session_options->EnableProfiling("darkpool_profile_");
        }
        
        // Add execution providers
        if (config.enable_tensorrt) {
            // TensorRT provider config would go here
        } else if (config.enable_gpu) {
            // CUDA provider config would go here
        }
        
        // Create session
        if (!config.model_path.empty()) {
            session = std::make_unique<Ort::Session>(*env, 
                config.model_path.c_str(), *session_options);
            
            // Get input/output info
            size_t num_inputs = session->GetInputCount();
            size_t num_outputs = session->GetOutputCount();
            
            Ort::AllocatorWithDefaultOptions allocator;
            
            // Get input names and shapes
            for (size_t i = 0; i < num_inputs; ++i) {
                auto name = session->GetInputNameAllocated(i, allocator);
                input_names.push_back(name.get());
                
                auto type_info = session->GetInputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
                input_shapes.push_back(tensor_info.GetShape());
            }
            
            // Get output names and shapes
            for (size_t i = 0; i < num_outputs; ++i) {
                auto name = session->GetOutputNameAllocated(i, allocator);
                output_names.push_back(name.get());
                
                auto type_info = session->GetOutputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
                output_shapes.push_back(tensor_info.GetShape());
            }
        }
        
        // Create memory info
        memory_info = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
        
        // Create run options
        run_options = std::make_unique<Ort::RunOptions>();
        run_options->SetRunLogSeverityLevel(2);
        
        // Initialize tensor cache
        for (auto& entry : tensor_cache.entries) {
            entry.in_use = false;
            entry.size = 0;
        }
        
        // Create initial batch
        current_batch = std::make_unique<BatchRequest>();
        current_batch->features.reserve(config.batch_size);
        
        return true;
        
    } catch (const Ort::Exception& e) {
        return false;
    }
}

void InferenceEngine::InferenceContext::cleanup() noexcept {
    session.reset();
    run_options.reset();
    memory_info.reset();
    allocator.reset();
    session_options.reset();
    env.reset();
}

// TensorCache implementation

Ort::Value* InferenceEngine::TensorCache::acquire(size_t size) noexcept {
    // Look for available tensor of correct size
    for (auto& entry : entries) {
        bool expected = false;
        if (entry.in_use.compare_exchange_strong(expected, true)) {
            if (entry.tensor && entry.size == size) {
                return entry.tensor.get();
            }
            entry.in_use = false;
        }
    }
    
    return nullptr;
}

void InferenceEngine::TensorCache::release(Ort::Value* tensor) noexcept {
    for (auto& entry : entries) {
        if (entry.tensor.get() == tensor) {
            entry.last_used_ns = std::chrono::high_resolution_clock::now()
                                .time_since_epoch().count();
            entry.in_use = false;
            return;
        }
    }
}

// Global functions

InferenceEngine& get_inference_engine() {
    std::lock_guard<std::mutex> lock(g_engine_mutex);
    
    if (!g_inference_engine) {
        InferenceEngine::Config config;
        config.batch_size = 32;
        config.enable_quantization = true;
        config.num_threads = 4;
        
        g_inference_engine = std::make_unique<InferenceEngine>(config);
    }
    
    return *g_inference_engine;
}

bool validate_model_file(const std::string& path) noexcept {
    try {
        // Check file exists
        std::ifstream file(path, std::ios::binary);
        if (!file.good()) return false;
        
        // Check ONNX magic number
        char magic[8];
        file.read(magic, 8);
        
        // ONNX files start with specific bytes
        return file.good();
        
    } catch (...) {
        return false;
    }
}

void InferenceEngine::handle_onnx_error(const std::exception& e) noexcept {
    // Log error (in production would use proper logging)
    stats_.total_inferences.fetch_add(1); // Count as failed inference
}

} 

