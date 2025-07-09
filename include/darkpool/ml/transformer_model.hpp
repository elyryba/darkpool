#pragma once

#include "darkpool/types.hpp"
#include "darkpool/ml/feature_engineering.hpp"
#include <vector>
#include <memory>
#include <array>

namespace darkpool::ml {

class TransformerModel {
public:
    static constexpr size_t MAX_SEQUENCE_LENGTH = 100;
    static constexpr size_t MODEL_DIM = 256;
    static constexpr size_t NUM_HEADS = 8;
    static constexpr size_t FF_DIM = 1024;
    
    struct Config {
        size_t sequence_length = 50;
        size_t model_dim = MODEL_DIM;
        size_t num_heads = NUM_HEADS;
        size_t num_layers = 6;
        size_t ff_dim = FF_DIM;
        double dropout_rate = 0.1;
        size_t max_position = 1000;
        bool use_causal_mask = false;  // For order book, we can see all
        bool enable_flash_attention = true;
        size_t batch_size = 32;
        bool use_fp16 = false;  // Half precision for speed
        bool enable_kv_cache = true;  // Cache key-value pairs
        size_t vocab_size = 0;  // 0 = continuous features
    };
    
    struct AttentionOutput {
        alignas(64) std::vector<std::vector<float>> hidden_states;
        std::vector<std::vector<float>> attention_weights;
        std::vector<float> attention_scores;
        
        // Pattern detection results
        struct Pattern {
            size_t start_idx;
            size_t end_idx;
            float confidence;
            std::string pattern_type;
        };
        std::vector<Pattern> detected_patterns;
    };
    
    struct PredictionOutput {
        // Dark pool detection
        float dark_pool_probability;
        float hidden_liquidity_estimate;
        std::array<float, 5> venue_scores;
        
        // Market microstructure predictions
        float price_impact_bps;
        float execution_cost_bps;
        float adverse_selection_cost;
        
        // Temporal predictions
        float intensity_next_1m;
        float intensity_next_5m;
        float volume_forecast;
        
        // Attention insights
        std::vector<size_t> important_timesteps;
        std::vector<size_t> important_features;
        
        // Confidence and metadata
        float overall_confidence;
        int64_t inference_time_us;
        std::string model_version;
    };
    
    explicit TransformerModel(const Config& config = {});
    ~TransformerModel();
    
    // Single sequence prediction
    PredictionOutput predict(
        const std::vector<FeatureEngineering::Features>& sequence) noexcept;
    
    // Batch prediction
    std::vector<PredictionOutput> predict_batch(
        const std::vector<std::vector<FeatureEngineering::Features>>& sequences) noexcept;
    
    // Get attention weights for interpretability
    AttentionOutput get_attention_analysis(
        const std::vector<FeatureEngineering::Features>& sequence) noexcept;
    
    // Model management
    bool load_weights(const std::string& path) noexcept;
    bool is_initialized() const noexcept;
    void warmup(size_t iterations = 10) noexcept;
    
    // Feature importance via attention
    std::vector<double> get_feature_importance() const noexcept;
    std::vector<size_t> get_critical_timesteps(
        const std::vector<FeatureEngineering::Features>& sequence) const noexcept;
    
private:
    struct MultiHeadAttention {
        alignas(64) struct Weights {
            std::vector<float> W_query;   // [model_dim, model_dim]
            std::vector<float> W_key;     // [model_dim, model_dim]
            std::vector<float> W_value;   // [model_dim, model_dim]
            std::vector<float> W_output;  // [model_dim, model_dim]
            std::vector<float> bias_q, bias_k, bias_v, bias_o;
        };
        
        Weights weights;
        size_t num_heads;
        size_t head_dim;
        
        // Key-value cache for efficiency
        alignas(64) struct KVCache {
            std::vector<float> keys;
            std::vector<float> values;
            size_t cached_length = 0;
            bool valid = false;
        };
        mutable KVCache cache;
        
        std::vector<float> forward(const std::vector<float>& input,
                                  size_t seq_len,
                                  const float* mask = nullptr) noexcept;
        
        void clear_cache() noexcept { cache.valid = false; }
    };
    
    struct FeedForward {
        alignas(64) std::vector<float> W1;  // [model_dim, ff_dim]
        alignas(64) std::vector<float> W2;  // [ff_dim, model_dim]
        alignas(64) std::vector<float> bias1;
        alignas(64) std::vector<float> bias2;
        
        std::vector<float> forward(const std::vector<float>& input) noexcept;
    };
    
    struct TransformerLayer {
        std::unique_ptr<MultiHeadAttention> attention;
        std::unique_ptr<FeedForward> feed_forward;
        
        // Layer normalization
        alignas(64) std::vector<float> ln1_gamma, ln1_beta;
        alignas(64) std::vector<float> ln2_gamma, ln2_beta;
        
        std::vector<float> forward(const std::vector<float>& input,
                                  size_t seq_len) noexcept;
    };
    
    struct PositionalEncoding {
        alignas(64) std::vector<std::vector<float>> encodings;
        
        void initialize(size_t max_length, size_t model_dim) noexcept;
        void apply(std::vector<float>& embeddings, size_t seq_len) noexcept;
    };
    
    Config config_;
    
    // Model components
    alignas(64) std::vector<float> input_projection_;  // Project features to model_dim
    std::vector<std::unique_ptr<TransformerLayer>> layers_;
    std::unique_ptr<PositionalEncoding> positional_encoding_;
    
    // Output heads
    alignas(64) std::vector<float> dark_pool_head_;
    alignas(64) std::vector<float> microstructure_head_;
    alignas(64) std::vector<float> temporal_head_;
    
    // Normalization parameters
    alignas(64) std::vector<float> feature_scale_;
    alignas(64) std::vector<float> feature_bias_;
    
    // Helper methods
    std::vector<float> embed_features(
        const std::vector<FeatureEngineering::Features>& sequence) noexcept;
    
    PredictionOutput decode_output(const std::vector<float>& encoded,
                                  size_t seq_len) noexcept;
    
    void layer_norm(std::vector<float>& x, const std::vector<float>& gamma,
                   const std::vector<float>& beta) noexcept;
    
    float gelu(float x) noexcept;
    void apply_gelu(std::vector<float>& x) noexcept;
    
    // SIMD optimized operations
    void simd_attention_scores(const float* query, const float* key,
                              float* scores, size_t seq_len,
                              size_t head_dim) noexcept;
    
    void simd_softmax(float* scores, size_t n) noexcept;
    
    // Flash attention implementation
    void flash_attention(const float* Q, const float* K, const float* V,
                        float* output, size_t seq_len, size_t head_dim) noexcept;
};

// Specialized transformer for order book dynamics
class OrderBookTransformer : public TransformerModel {
public:
    struct OrderBookPrediction {
        // Price level predictions
        std::vector<float> bid_depths_1m;
        std::vector<float> ask_depths_1m;
        float mid_price_change_1m;
        float spread_forecast_1m;
        
        // Liquidity predictions
        float hidden_liquidity_bid;
        float hidden_liquidity_ask;
        float iceberg_probability;
        
        // Execution predictions
        float market_impact_10k_shares;
        float optimal_slice_size;
        float execution_alpha;
        
        // Microstructure regime
        enum Regime {
            NORMAL = 0,
            HIGH_VOLATILITY,
            LIQUIDITY_CRISIS,
            INSTITUTIONAL_FLOW,
            HFT_DOMINANT
        };
        Regime predicted_regime;
        float regime_confidence;
    };
    
    explicit OrderBookTransformer(const Config& config = {});
    
    OrderBookPrediction predict_order_book(
        const std::vector<Quote>& quotes,
        const std::vector<Trade>& trades,
        size_t lookback = 50) noexcept;
};

} 
