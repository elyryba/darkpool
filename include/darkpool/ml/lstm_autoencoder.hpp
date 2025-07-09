#pragma once

#include "darkpool/types.hpp"
#include "darkpool/ml/feature_engineering.hpp"
#include <vector>
#include <memory>
#include <deque>

namespace darkpool::ml {

class LSTMAutoencoder {
public:
    static constexpr size_t MAX_SEQUENCE_LENGTH = 100;
    static constexpr size_t HIDDEN_SIZE = 64;
    static constexpr size_t LATENT_DIM = 16;
    
    struct Config {
        size_t sequence_length = 50;
        size_t hidden_size = HIDDEN_SIZE;
        size_t latent_dim = LATENT_DIM;
        size_t num_layers = 2;
        double dropout_rate = 0.1;
        double learning_rate = 0.001;
        size_t batch_size = 32;
        double anomaly_threshold = 2.5; // Reconstruction error z-score
        bool bidirectional = true;
        bool use_attention = true;
        size_t attention_heads = 4;
        size_t window_stride = 1;
        bool normalize_sequences = true;
    };
    
    struct AnomalyScore {
        double reconstruction_error;
        double z_score;
        double percentile;
        std::vector<double> feature_errors; // Per-feature reconstruction errors
        std::vector<size_t> anomalous_timesteps;
        AnomalyType detected_type;
        double confidence;
        int64_t sequence_start_time;
        int64_t sequence_end_time;
        uint32_t symbol;
        
        bool is_anomaly() const noexcept {
            return z_score > 2.5 || percentile > 0.99;
        }
    };
    
    struct SequenceData {
        alignas(64) std::vector<std::vector<double>> sequences; // [time, features]
        std::vector<int64_t> timestamps;
        uint32_t symbol;
        bool normalized = false;
    };
    
    explicit LSTMAutoencoder(uint32_t symbol, const Config& config = {});
    ~LSTMAutoencoder();
    
    // Update with new features
    void update(const FeatureEngineering::Features& features) noexcept;
    
    // Detect anomalies in current sequence
    AnomalyScore detect_anomaly() const noexcept;
    
    // Batch anomaly detection
    std::vector<AnomalyScore> detect_anomalies_batch(
        const std::vector<SequenceData>& sequences) const noexcept;
    
    // Get reconstruction for analysis
    std::vector<std::vector<double>> reconstruct_sequence(
        const SequenceData& sequence) const noexcept;
    
    // Model management
    bool load_pretrained_weights(const std::string& path) noexcept;
    bool is_ready() const noexcept;
    size_t get_sequence_count() const noexcept;
    
    // Training interface (if online learning enabled)
    void train_step(const SequenceData& sequence) noexcept;
    
private:
    struct LSTMCell {
        alignas(64) struct Gates {
            std::vector<float> input_gate;
            std::vector<float> forget_gate;
            std::vector<float> output_gate;
            std::vector<float> cell_candidate;
        };
        
        // Weights (transposed for cache efficiency)
        alignas(64) std::vector<float> W_ii, W_if, W_ig, W_io;  // Input weights
        alignas(64) std::vector<float> W_hi, W_hf, W_hg, W_ho;  // Hidden weights
        alignas(64) std::vector<float> b_i, b_f, b_g, b_o;      // Biases
        
        // State
        std::vector<float> hidden_state;
        std::vector<float> cell_state;
        
        void forward(const float* input, size_t input_size) noexcept;
        void reset_state() noexcept;
    };
    
    struct AttentionLayer {
        alignas(64) std::vector<float> W_query;
        alignas(64) std::vector<float> W_key;
        alignas(64) std::vector<float> W_value;
        alignas(64) std::vector<float> W_output;
        
        std::vector<float> apply_attention(
            const std::vector<float>& sequence,
            size_t seq_len, size_t hidden_size) const noexcept;
    };
    
    struct EncoderDecoder {
        // Encoder LSTM layers
        std::vector<std::unique_ptr<LSTMCell>> encoder_layers;
        std::unique_ptr<AttentionLayer> encoder_attention;
        
        // Decoder LSTM layers
        std::vector<std::unique_ptr<LSTMCell>> decoder_layers;
        std::unique_ptr<AttentionLayer> decoder_attention;
        
        // Projection layers
        alignas(64) std::vector<float> encoder_projection;  // Hidden -> Latent
        alignas(64) std::vector<float> decoder_projection;  // Latent -> Hidden
        alignas(64) std::vector<float> output_projection;   // Hidden -> Features
        
        // Encode sequence to latent representation
        std::vector<float> encode(const SequenceData& sequence) noexcept;
        
        // Decode from latent representation
        std::vector<std::vector<float>> decode(
            const std::vector<float>& latent,
            size_t sequence_length) noexcept;
    };
    
    // Configuration
    uint32_t symbol_;
    Config config_;
    
    // Model
    std::unique_ptr<EncoderDecoder> model_;
    
    // Sequence buffer
    std::deque<std::vector<double>> feature_buffer_;
    std::deque<int64_t> timestamp_buffer_;
    
    // Normalization parameters
    alignas(64) std::vector<double> feature_means_;
    alignas(64) std::vector<double> feature_stds_;
    
    // Anomaly detection statistics
    mutable std::deque<double> error_history_;
    mutable double error_mean_ = 0.0;
    mutable double error_std_ = 1.0;
    mutable size_t total_sequences_ = 0;
    
    // Helper methods
    SequenceData prepare_sequence() const noexcept;
    void normalize_sequence(SequenceData& sequence) const noexcept;
    void denormalize_sequence(std::vector<std::vector<double>>& sequence) const noexcept;
    
    double compute_reconstruction_error(
        const SequenceData& original,
        const std::vector<std::vector<double>>& reconstructed) const noexcept;
    
    std::vector<double> compute_feature_errors(
        const SequenceData& original,
        const std::vector<std::vector<double>>& reconstructed) const noexcept;
    
    void update_error_statistics(double error) const noexcept;
    
    // SIMD optimized operations
    void simd_lstm_forward(float* gates, const float* input,
                          const float* hidden, const float* weights,
                          const float* bias, size_t size) const noexcept;
    
    float simd_dot_product(const float* a, const float* b, size_t n) const noexcept;
};

// Specialized anomaly patterns detected by LSTM
enum class LSTMPattern {
    NORMAL = 0,
    REGIME_CHANGE,       // Sudden shift in dynamics
    CYCLIC_ANOMALY,      // Disruption in periodic pattern
    TREND_BREAK,         // Unexpected trend reversal
    VOLATILITY_CLUSTER,  // Abnormal volatility clustering
    CORRELATION_BREAK,   // Cross-feature correlation anomaly
    SEQUENCE_NOVELTY     // Never-seen-before pattern
};

// Helper to identify specific pattern types
LSTMPattern classify_lstm_anomaly(const LSTMAutoencoder::AnomalyScore& score) noexcept;

} 
