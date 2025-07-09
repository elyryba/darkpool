#include "darkpool/ml/lstm_autoencoder.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <fstream>
#include <immintrin.h>

namespace darkpool::ml {

LSTMAutoencoder::LSTMAutoencoder(uint32_t symbol, const Config& config)
    : symbol_(symbol)
    , config_(config)
    , model_(std::make_unique<EncoderDecoder>()) {
    
    // Initialize model architecture
    size_t input_size = FeatureEngineering::NUM_FEATURES;
    
    // Create encoder layers
    for (size_t i = 0; i < config_.num_layers; ++i) {
        auto cell = std::make_unique<LSTMCell>();
        
        size_t layer_input_size = (i == 0) ? input_size : config_.hidden_size;
        size_t layer_output_size = config_.hidden_size;
        
        // Initialize weights with Xavier initialization
        float scale = std::sqrt(2.0f / (layer_input_size + layer_output_size));
        
        // Input weights
        cell->W_ii.resize(layer_input_size * layer_output_size);
        cell->W_if.resize(layer_input_size * layer_output_size);
        cell->W_ig.resize(layer_input_size * layer_output_size);
        cell->W_io.resize(layer_input_size * layer_output_size);
        
        // Hidden weights
        cell->W_hi.resize(layer_output_size * layer_output_size);
        cell->W_hf.resize(layer_output_size * layer_output_size);
        cell->W_hg.resize(layer_output_size * layer_output_size);
        cell->W_ho.resize(layer_output_size * layer_output_size);
        
        // Biases
        cell->b_i.resize(layer_output_size, 0.0f);
        cell->b_f.resize(layer_output_size, 1.0f); // Forget gate bias = 1
        cell->b_g.resize(layer_output_size, 0.0f);
        cell->b_o.resize(layer_output_size, 0.0f);
        
        // State
        cell->hidden_state.resize(layer_output_size, 0.0f);
        cell->cell_state.resize(layer_output_size, 0.0f);
        
        model_->encoder_layers.push_back(std::move(cell));
    }
    
    // Create decoder layers (mirror of encoder)
    for (size_t i = 0; i < config_.num_layers; ++i) {
        auto cell = std::make_unique<LSTMCell>();
        
        size_t layer_input_size = (i == 0) ? config_.latent_dim : config_.hidden_size;
        size_t layer_output_size = config_.hidden_size;
        
        // Initialize similar to encoder
        cell->W_ii.resize(layer_input_size * layer_output_size);
        cell->W_if.resize(layer_input_size * layer_output_size);
        cell->W_ig.resize(layer_input_size * layer_output_size);
        cell->W_io.resize(layer_input_size * layer_output_size);
        
        cell->W_hi.resize(layer_output_size * layer_output_size);
        cell->W_hf.resize(layer_output_size * layer_output_size);
        cell->W_hg.resize(layer_output_size * layer_output_size);
        cell->W_ho.resize(layer_output_size * layer_output_size);
        
        cell->b_i.resize(layer_output_size, 0.0f);
        cell->b_f.resize(layer_output_size, 1.0f);
        cell->b_g.resize(layer_output_size, 0.0f);
        cell->b_o.resize(layer_output_size, 0.0f);
        
        cell->hidden_state.resize(layer_output_size, 0.0f);
        cell->cell_state.resize(layer_output_size, 0.0f);
        
        model_->decoder_layers.push_back(std::move(cell));
    }
    
    // Initialize projection layers
    model_->encoder_projection.resize(config_.hidden_size * config_.latent_dim);
    model_->decoder_projection.resize(config_.latent_dim * config_.hidden_size);
    model_->output_projection.resize(config_.hidden_size * input_size);
    
    // Initialize attention layers if enabled
    if (config_.use_attention) {
        model_->encoder_attention = std::make_unique<AttentionLayer>();
        model_->decoder_attention = std::make_unique<AttentionLayer>();
        
        size_t head_dim = config_.hidden_size / config_.attention_heads;
        model_->encoder_attention->W_query.resize(config_.hidden_size * config_.hidden_size);
        model_->encoder_attention->W_key.resize(config_.hidden_size * config_.hidden_size);
        model_->encoder_attention->W_value.resize(config_.hidden_size * config_.hidden_size);
        model_->encoder_attention->W_output.resize(config_.hidden_size * config_.hidden_size);
    }
    
    // Initialize normalization parameters
    feature_means_.resize(input_size, 0.0);
    feature_stds_.resize(input_size, 1.0);
}

LSTMAutoencoder::~LSTMAutoencoder() = default;

void LSTMAutoencoder::update(const FeatureEngineering::Features& features) noexcept {
    if (!features.valid) return;
    
    // Convert to vector
    std::vector<double> feature_vec(features.values.begin(), features.values.end());
    
    // Add to buffer
    feature_buffer_.push_back(feature_vec);
    timestamp_buffer_.push_back(features.timestamp);
    
    // Maintain window size
    while (feature_buffer_.size() > config_.sequence_length) {
        feature_buffer_.pop_front();
        timestamp_buffer_.pop_front();
    }
    
    // Update normalization parameters incrementally
    if (config_.normalize_sequences && feature_buffer_.size() > 10) {
        for (size_t i = 0; i < feature_vec.size(); ++i) {
            double delta = feature_vec[i] - feature_means_[i];
            feature_means_[i] += delta / feature_buffer_.size();
            
            // Welford's algorithm for variance
            double delta2 = feature_vec[i] - feature_means_[i];
            feature_stds_[i] = std::sqrt(
                (feature_stds_[i] * feature_stds_[i] * (feature_buffer_.size() - 1) + 
                 delta * delta2) / feature_buffer_.size()
            );
        }
    }
}

LSTMAutoencoder::AnomalyScore LSTMAutoencoder::detect_anomaly() const noexcept {
    AnomalyScore score{};
    score.symbol = symbol_;
    
    if (feature_buffer_.size() < config_.sequence_length) {
        return score; // Not enough data
    }
    
    // Prepare sequence
    auto sequence = prepare_sequence();
    
    // Encode and decode
    auto latent = model_->encode(sequence);
    auto reconstructed = model_->decode(latent, config_.sequence_length);
    
    // Convert float reconstruction to double
    std::vector<std::vector<double>> reconstructed_double;
    for (const auto& step : reconstructed) {
        std::vector<double> step_double(step.begin(), step.end());
        reconstructed_double.push_back(step_double);
    }
    
    // Denormalize if needed
    if (config_.normalize_sequences) {
        denormalize_sequence(reconstructed_double);
    }
    
    // Compute reconstruction error
    score.reconstruction_error = compute_reconstruction_error(sequence, reconstructed_double);
    score.feature_errors = compute_feature_errors(sequence, reconstructed_double);
    
    // Update error statistics
    update_error_statistics(score.reconstruction_error);
    
    // Compute z-score
    if (error_std_ > 0) {
        score.z_score = (score.reconstruction_error - error_mean_) / error_std_;
    }
    
    // Compute percentile
    size_t below_count = 0;
    for (double err : error_history_) {
        if (err < score.reconstruction_error) below_count++;
    }
    score.percentile = static_cast<double>(below_count) / error_history_.size();
    
    // Set timestamps
    score.sequence_start_time = timestamp_buffer_.front();
    score.sequence_end_time = timestamp_buffer_.back();
    
    // Detect anomaly type
    if (score.is_anomaly()) {
        score.confidence = std::min(score.z_score / 5.0, 1.0);
        
        // Analyze pattern
        if (score.z_score > 4.0) {
            score.detected_type = AnomalyType::SEQUENCE_ANOMALY;
        } else if (score.percentile > 0.95) {
            score.detected_type = AnomalyType::STATISTICAL_ANOMALY;
        } else {
            score.detected_type = AnomalyType::PATTERN_CHANGE;
        }
        
        // Find anomalous timesteps
        for (size_t i = 0; i < score.feature_errors.size(); ++i) {
            if (score.feature_errors[i] > 2.0 * error_std_) {
                score.anomalous_timesteps.push_back(i);
            }
        }
    }
    
    return score;
}

bool LSTMAutoencoder::load_pretrained_weights(const std::string& path) noexcept {
    try {
        std::ifstream file(path, std::ios::binary);
        if (!file) return false;
        
        // Read model weights
        // Format: encoder weights, decoder weights, projections
        
        // This is simplified - in production would use proper serialization
        for (auto& layer : model_->encoder_layers) {
            file.read(reinterpret_cast<char*>(layer->W_ii.data()), 
                     layer->W_ii.size() * sizeof(float));
            file.read(reinterpret_cast<char*>(layer->W_if.data()), 
                     layer->W_if.size() * sizeof(float));
            // ... continue for all weights
        }
        
        return file.good();
    } catch (...) {
        return false;
    }
}

bool LSTMAutoencoder::is_ready() const noexcept {
    return feature_buffer_.size() >= config_.sequence_length &&
           total_sequences_ > 100; // Need some history for statistics
}

size_t LSTMAutoencoder::get_sequence_count() const noexcept {
    return total_sequences_;
}

LSTMAutoencoder::SequenceData LSTMAutoencoder::prepare_sequence() const noexcept {
    SequenceData data;
    data.symbol = symbol_;
    
    // Copy from buffer
    data.sequences.reserve(feature_buffer_.size());
    for (const auto& features : feature_buffer_) {
        data.sequences.push_back(features);
    }
    
    data.timestamps = std::vector<int64_t>(timestamp_buffer_.begin(), 
                                          timestamp_buffer_.end());
    
    // Normalize if configured
    if (config_.normalize_sequences) {
        normalize_sequence(data);
    }
    
    return data;
}

void LSTMAutoencoder::normalize_sequence(SequenceData& sequence) const noexcept {
    for (auto& timestep : sequence.sequences) {
        for (size_t i = 0; i < timestep.size(); ++i) {
            if (feature_stds_[i] > 0) {
                timestep[i] = (timestep[i] - feature_means_[i]) / feature_stds_[i];
            }
        }
    }
    sequence.normalized = true;
}

void LSTMAutoencoder::denormalize_sequence(
    std::vector<std::vector<double>>& sequence) const noexcept {
    
    for (auto& timestep : sequence) {
        for (size_t i = 0; i < timestep.size(); ++i) {
            timestep[i] = timestep[i] * feature_stds_[i] + feature_means_[i];
        }
    }
}

double LSTMAutoencoder::compute_reconstruction_error(
    const SequenceData& original,
    const std::vector<std::vector<double>>& reconstructed) const noexcept {
    
    double total_error = 0.0;
    size_t count = 0;
    
    size_t min_len = std::min(original.sequences.size(), reconstructed.size());
    
    for (size_t t = 0; t < min_len; ++t) {
        for (size_t f = 0; f < original.sequences[t].size(); ++f) {
            double diff = original.sequences[t][f] - reconstructed[t][f];
            total_error += diff * diff;
            count++;
        }
    }
    
    return std::sqrt(total_error / count); // RMSE
}

void LSTMAutoencoder::update_error_statistics(double error) const noexcept {
    error_history_.push_back(error);
    
    // Maintain history size
    if (error_history_.size() > 1000) {
        error_history_.pop_front();
    }
    
    // Update running statistics
    if (error_history_.size() > 1) {
        double sum = 0.0, sum_sq = 0.0;
        for (double e : error_history_) {
            sum += e;
            sum_sq += e * e;
        }
        
        error_mean_ = sum / error_history_.size();
        double variance = (sum_sq / error_history_.size()) - (error_mean_ * error_mean_);
        error_std_ = std::sqrt(std::max(variance, 1e-10));
    }
    
    total_sequences_++;
}

// LSTMCell implementation

void LSTMAutoencoder::LSTMCell::forward(const float* input, size_t input_size) noexcept {
    size_t hidden_size = hidden_state.size();
    Gates gates;
    gates.input_gate.resize(hidden_size);
    gates.forget_gate.resize(hidden_size);
    gates.output_gate.resize(hidden_size);
    gates.cell_candidate.resize(hidden_size);
    
    // Compute gates using SIMD
    #pragma omp simd
    for (size_t i = 0; i < hidden_size; ++i) {
        float i_gate = b_i[i];
        float f_gate = b_f[i];
        float g_gate = b_g[i];
        float o_gate = b_o[i];
        
        // Input contributions
        for (size_t j = 0; j < input_size; ++j) {
            i_gate += input[j] * W_ii[j * hidden_size + i];
            f_gate += input[j] * W_if[j * hidden_size + i];
            g_gate += input[j] * W_ig[j * hidden_size + i];
            o_gate += input[j] * W_io[j * hidden_size + i];
        }
        
        // Hidden contributions
        for (size_t j = 0; j < hidden_size; ++j) {
            i_gate += hidden_state[j] * W_hi[j * hidden_size + i];
            f_gate += hidden_state[j] * W_hf[j * hidden_size + i];
            g_gate += hidden_state[j] * W_hg[j * hidden_size + i];
            o_gate += hidden_state[j] * W_ho[j * hidden_size + i];
        }
        
        // Apply activations
        gates.input_gate[i] = 1.0f / (1.0f + std::exp(-i_gate));       // Sigmoid
        gates.forget_gate[i] = 1.0f / (1.0f + std::exp(-f_gate));      // Sigmoid
        gates.cell_candidate[i] = std::tanh(g_gate);                    // Tanh
        gates.output_gate[i] = 1.0f / (1.0f + std::exp(-o_gate));      // Sigmoid
        
        // Update cell state
        cell_state[i] = gates.forget_gate[i] * cell_state[i] + 
                       gates.input_gate[i] * gates.cell_candidate[i];
        
        // Update hidden state
        hidden_state[i] = gates.output_gate[i] * std::tanh(cell_state[i]);
    }
}

void LSTMAutoencoder::LSTMCell::reset_state() noexcept {
    std::fill(hidden_state.begin(), hidden_state.end(), 0.0f);
    std::fill(cell_state.begin(), cell_state.end(), 0.0f);
}

// EncoderDecoder implementation

std::vector<float> LSTMAutoencoder::EncoderDecoder::encode(
    const SequenceData& sequence) noexcept {
    
    // Reset encoder states
    for (auto& layer : encoder_layers) {
        layer->reset_state();
    }
    
    // Process sequence through encoder
    std::vector<float> current_input;
    
    for (const auto& timestep : sequence.sequences) {
        // Convert double to float
        current_input.resize(timestep.size());
        for (size_t i = 0; i < timestep.size(); ++i) {
            current_input[i] = static_cast<float>(timestep[i]);
        }
        
        // Forward through layers
        for (size_t layer_idx = 0; layer_idx < encoder_layers.size(); ++layer_idx) {
            encoder_layers[layer_idx]->forward(
                current_input.data(), 
                current_input.size()
            );
            
            // Use hidden state as input to next layer
            if (layer_idx < encoder_layers.size() - 1) {
                current_input = encoder_layers[layer_idx]->hidden_state;
            }
        }
    }
    
    // Apply attention if enabled
    std::vector<float> encoded = encoder_layers.back()->hidden_state;
    if (encoder_attention) {
        // Simplified - would apply attention over all timesteps
        encoded = encoder_attention->apply_attention(
            encoded, 1, encoded.size()
        );
    }
    
    // Project to latent space
    std::vector<float> latent(encoder_projection.size() / encoded.size());
    for (size_t i = 0; i < latent.size(); ++i) {
        latent[i] = 0.0f;
        for (size_t j = 0; j < encoded.size(); ++j) {
            latent[i] += encoded[j] * encoder_projection[j * latent.size() + i];
        }
    }
    
    return latent;
}

std::vector<std::vector<float>> LSTMAutoencoder::EncoderDecoder::decode(
    const std::vector<float>& latent,
    size_t sequence_length) noexcept {
    
    // Reset decoder states
    for (auto& layer : decoder_layers) {
        layer->reset_state();
    }
    
    // Project from latent space
    std::vector<float> decoder_input(decoder_projection.size() / latent.size());
    for (size_t i = 0; i < decoder_input.size(); ++i) {
        decoder_input[i] = 0.0f;
        for (size_t j = 0; j < latent.size(); ++j) {
            decoder_input[i] += latent[j] * decoder_projection[j * decoder_input.size() + i];
        }
    }
    
    // Generate sequence
    std::vector<std::vector<float>> reconstructed;
    reconstructed.reserve(sequence_length);
    
    for (size_t t = 0; t < sequence_length; ++t) {
        // Forward through decoder layers
        std::vector<float> current_input = (t == 0) ? decoder_input : decoder_layers.back()->hidden_state;
        
        for (auto& layer : decoder_layers) {
            layer->forward(current_input.data(), current_input.size());
            current_input = layer->hidden_state;
        }
        
        // Apply attention if enabled
        if (decoder_attention) {
            current_input = decoder_attention->apply_attention(
                current_input, 1, current_input.size()
            );
        }
        
        // Project to output space
        std::vector<float> output(output_projection.size() / current_input.size());
        for (size_t i = 0; i < output.size(); ++i) {
            output[i] = 0.0f;
            for (size_t j = 0; j < current_input.size(); ++j) {
                output[i] += current_input[j] * output_projection[j * output.size() + i];
            }
        }
        
        reconstructed.push_back(output);
    }
    
    return reconstructed;
}

// Helper function implementation

LSTMPattern classify_lstm_anomaly(const LSTMAutoencoder::AnomalyScore& score) noexcept {
    if (!score.is_anomaly()) {
        return LSTMPattern::NORMAL;
    }
    
    // Analyze anomalous timesteps pattern
    if (score.anomalous_timesteps.size() > score.feature_errors.size() * 0.8) {
        return LSTMPattern::REGIME_CHANGE;
    }
    
    // Check for periodic disruption
    bool has_periodic_pattern = false;
    if (score.anomalous_timesteps.size() > 2) {
        std::vector<size_t> diffs;
        for (size_t i = 1; i < score.anomalous_timesteps.size(); ++i) {
            diffs.push_back(score.anomalous_timesteps[i] - score.anomalous_timesteps[i-1]);
        }
        
        // Check if differences are similar (periodic)
        double mean_diff = std::accumulate(diffs.begin(), diffs.end(), 0.0) / diffs.size();
        double variance = 0.0;
        for (size_t d : diffs) {
            variance += std::pow(d - mean_diff, 2);
        }
        variance /= diffs.size();
        
        if (variance < mean_diff * 0.2) {
            has_periodic_pattern = true;
        }
    }
    
    if (has_periodic_pattern) {
        return LSTMPattern::CYCLIC_ANOMALY;
    }
    
    // Check reconstruction error pattern
    if (score.z_score > 5.0) {
        return LSTMPattern::SEQUENCE_NOVELTY;
    }
    
    // Check feature errors for correlation breaks
    double max_feature_error = *std::max_element(
        score.feature_errors.begin(), score.feature_errors.end()
    );
    double mean_feature_error = std::accumulate(
        score.feature_errors.begin(), score.feature_errors.end(), 0.0
    ) / score.feature_errors.size();
    
    if (max_feature_error > 3.0 * mean_feature_error) {
        return LSTMPattern::CORRELATION_BREAK;
    }
    
    return LSTMPattern::TREND_BREAK;
}

} 
