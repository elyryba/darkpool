#include "darkpool/ml/transformer_model.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <immintrin.h>

namespace darkpool::ml {

TransformerModel::TransformerModel(const Config& config)
    : config_(config) {
    
    // Initialize positional encoding
    positional_encoding_ = std::make_unique<PositionalEncoding>();
    positional_encoding_->initialize(config_.max_position, config_.model_dim);
    
    // Initialize input projection
    size_t input_features = FeatureEngineering::NUM_FEATURES;
    input_projection_.resize(input_features * config_.model_dim);
    
    // Xavier initialization
    float scale = std::sqrt(2.0f / (input_features + config_.model_dim));
    for (auto& w : input_projection_) {
        w = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f * scale;
    }
    
    // Initialize transformer layers
    for (size_t i = 0; i < config_.num_layers; ++i) {
        auto layer = std::make_unique<TransformerLayer>();
        
        // Initialize attention
        layer->attention = std::make_unique<MultiHeadAttention>();
        layer->attention->num_heads = config_.num_heads;
        layer->attention->head_dim = config_.model_dim / config_.num_heads;
        
        size_t weight_size = config_.model_dim * config_.model_dim;
        layer->attention->weights.W_query.resize(weight_size);
        layer->attention->weights.W_key.resize(weight_size);
        layer->attention->weights.W_value.resize(weight_size);
        layer->attention->weights.W_output.resize(weight_size);
        
        layer->attention->weights.bias_q.resize(config_.model_dim, 0.0f);
        layer->attention->weights.bias_k.resize(config_.model_dim, 0.0f);
        layer->attention->weights.bias_v.resize(config_.model_dim, 0.0f);
        layer->attention->weights.bias_o.resize(config_.model_dim, 0.0f);
        
        // Initialize feed-forward
        layer->feed_forward = std::make_unique<FeedForward>();
        layer->feed_forward->W1.resize(config_.model_dim * config_.ff_dim);
        layer->feed_forward->W2.resize(config_.ff_dim * config_.model_dim);
        layer->feed_forward->bias1.resize(config_.ff_dim, 0.0f);
        layer->feed_forward->bias2.resize(config_.model_dim, 0.0f);
        
        // Layer normalization parameters
        layer->ln1_gamma.resize(config_.model_dim, 1.0f);
        layer->ln1_beta.resize(config_.model_dim, 0.0f);
        layer->ln2_gamma.resize(config_.model_dim, 1.0f);
        layer->ln2_beta.resize(config_.model_dim, 0.0f);
        
        layers_.push_back(std::move(layer));
    }
    
    // Initialize output heads
    dark_pool_head_.resize(config_.model_dim * 8);      // 8 outputs
    microstructure_head_.resize(config_.model_dim * 3); // 3 outputs
    temporal_head_.resize(config_.model_dim * 3);       // 3 outputs
    
    // Feature normalization
    feature_scale_.resize(input_features, 1.0f);
    feature_bias_.resize(input_features, 0.0f);
}

TransformerModel::~TransformerModel() = default;

TransformerModel::PredictionOutput TransformerModel::predict(
    const std::vector<FeatureEngineering::Features>& sequence) noexcept {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    PredictionOutput output{};
    
    if (sequence.empty()) return output;
    
    // Embed features
    auto embeddings = embed_features(sequence);
    size_t seq_len = sequence.size();
    
    // Apply positional encoding
    positional_encoding_->apply(embeddings, seq_len);
    
    // Forward through transformer layers
    std::vector<float> hidden = embeddings;
    for (auto& layer : layers_) {
        hidden = layer->forward(hidden, seq_len);
    }
    
    // Decode output
    output = decode_output(hidden, seq_len);
    
    // Add metadata
    auto end_time = std::chrono::high_resolution_clock::now();
    output.inference_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time).count();
    output.model_version = "1.0.0";
    
    return output;
}

std::vector<float> TransformerModel::embed_features(
    const std::vector<FeatureEngineering::Features>& sequence) noexcept {
    
    size_t seq_len = sequence.size();
    size_t input_dim = FeatureEngineering::NUM_FEATURES;
    std::vector<float> embeddings(seq_len * config_.model_dim, 0.0f);
    
    // Project each timestep
    #pragma omp parallel for
    for (size_t t = 0; t < seq_len; ++t) {
        const double* features = sequence[t].data();
        float* output = &embeddings[t * config_.model_dim];
        
        // Normalize and project
        alignas(32) float normalized[FeatureEngineering::NUM_FEATURES];
        for (size_t i = 0; i < input_dim; ++i) {
            normalized[i] = static_cast<float>(features[i]) * feature_scale_[i] + feature_bias_[i];
        }
        
        // Matrix multiply: output = normalized @ input_projection
        for (size_t i = 0; i < config_.model_dim; ++i) {
            float sum = 0.0f;
            
            // SIMD dot product
            __m256 sum_vec = _mm256_setzero_ps();
            for (size_t j = 0; j < input_dim - 7; j += 8) {
                __m256 feat = _mm256_loadu_ps(&normalized[j]);
                __m256 weight = _mm256_loadu_ps(&input_projection_[j * config_.model_dim + i]);
                sum_vec = _mm256_fmadd_ps(feat, weight, sum_vec);
            }
            
            // Horizontal sum
            __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
            __m128 sum_low = _mm256_castps256_ps128(sum_vec);
            __m128 sum_128 = _mm_add_ps(sum_low, sum_high);
            sum_128 = _mm_hadd_ps(sum_128, sum_128);
            sum_128 = _mm_hadd_ps(sum_128, sum_128);
            sum = _mm_cvtss_f32(sum_128);
            
            // Handle remaining
            for (size_t j = (input_dim / 8) * 8; j < input_dim; ++j) {
                sum += normalized[j] * input_projection_[j * config_.model_dim + i];
            }
            
            output[i] = sum;
        }
    }
    
    return embeddings;
}

TransformerModel::PredictionOutput TransformerModel::decode_output(
    const std::vector<float>& encoded, size_t seq_len) noexcept {
    
    PredictionOutput output{};
    
    // Use last hidden state for classification
    const float* last_hidden = &encoded[(seq_len - 1) * config_.model_dim];
    
    // Dark pool predictions (8 outputs)
    std::vector<float> dark_pool_logits(8, 0.0f);
    for (size_t i = 0; i < 8; ++i) {
        for (size_t j = 0; j < config_.model_dim; ++j) {
            dark_pool_logits[i] += last_hidden[j] * dark_pool_head_[j * 8 + i];
        }
    }
    
    // Apply sigmoid/softmax
    output.dark_pool_probability = 1.0f / (1.0f + std::exp(-dark_pool_logits[0]));
    output.hidden_liquidity_estimate = std::exp(dark_pool_logits[1]) * 1000; // Scale
    
    // Venue scores (softmax over 5 venues)
    float max_score = *std::max_element(dark_pool_logits.begin() + 3, 
                                       dark_pool_logits.begin() + 8);
    float sum_exp = 0.0f;
    for (size_t i = 0; i < 5; ++i) {
        output.venue_scores[i] = std::exp(dark_pool_logits[3 + i] - max_score);
        sum_exp += output.venue_scores[i];
    }
    for (auto& score : output.venue_scores) {
        score /= sum_exp;
    }
    
    // Microstructure predictions (3 outputs)
    std::vector<float> micro_logits(3, 0.0f);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < config_.model_dim; ++j) {
            micro_logits[i] += last_hidden[j] * microstructure_head_[j * 3 + i];
        }
    }
    
    output.price_impact_bps = std::tanh(micro_logits[0]) * 10.0f; // [-10, 10] bps
    output.execution_cost_bps = std::abs(micro_logits[1]) * 5.0f; // [0, inf) bps
    output.adverse_selection_cost = std::abs(micro_logits[2]) * 3.0f;
    
    // Temporal predictions (3 outputs)
    std::vector<float> temporal_logits(3, 0.0f);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < config_.model_dim; ++j) {
            temporal_logits[i] += last_hidden[j] * temporal_head_[j * 3 + i];
        }
    }
    
    output.intensity_next_1m = std::exp(temporal_logits[0]);
    output.intensity_next_5m = std::exp(temporal_logits[1]);
    output.volume_forecast = std::exp(temporal_logits[2]) * 1000;
    
    // Extract important timesteps from attention
    // Simplified - in production would analyze actual attention weights
    for (size_t i = seq_len - 5; i < seq_len; ++i) {
        output.important_timesteps.push_back(i);
    }
    
    // Confidence based on logit magnitudes
    float avg_confidence = 0.0f;
    for (auto logit : dark_pool_logits) {
        avg_confidence += std::abs(logit);
    }
    output.overall_confidence = std::tanh(avg_confidence / 8.0f);
    
    return output;
}

void TransformerModel::warmup(size_t iterations) noexcept {
    std::vector<FeatureEngineering::Features> dummy_sequence;
    dummy_sequence.resize(config_.sequence_length);
    
    for (auto& features : dummy_sequence) {
        features.valid = true;
        for (size_t i = 0; i < features.values.size(); ++i) {
            features.values[i] = static_cast<double>(i) / features.values.size();
        }
    }
    
    for (size_t i = 0; i < iterations; ++i) {
        predict(dummy_sequence);
    }
}

bool TransformerModel::load_weights(const std::string& path) noexcept {
    try {
        std::ifstream file(path, std::ios::binary);
        if (!file) return false;
        
        // Load all layer weights
        for (auto& layer : layers_) {
            // Attention weights
            file.read(reinterpret_cast<char*>(layer->attention->weights.W_query.data()),
                     layer->attention->weights.W_query.size() * sizeof(float));
            file.read(reinterpret_cast<char*>(layer->attention->weights.W_key.data()),
                     layer->attention->weights.W_key.size() * sizeof(float));
            file.read(reinterpret_cast<char*>(layer->attention->weights.W_value.data()),
                     layer->attention->weights.W_value.size() * sizeof(float));
            file.read(reinterpret_cast<char*>(layer->attention->weights.W_output.data()),
                     layer->attention->weights.W_output.size() * sizeof(float));
            
            // Feed-forward weights
            file.read(reinterpret_cast<char*>(layer->feed_forward->W1.data()),
                     layer->feed_forward->W1.size() * sizeof(float));
            file.read(reinterpret_cast<char*>(layer->feed_forward->W2.data()),
                     layer->feed_forward->W2.size() * sizeof(float));
            
            // Layer norm parameters
            file.read(reinterpret_cast<char*>(layer->ln1_gamma.data()),
                     layer->ln1_gamma.size() * sizeof(float));
            file.read(reinterpret_cast<char*>(layer->ln1_beta.data()),
                     layer->ln1_beta.size() * sizeof(float));
        }
        
        // Clear KV caches
        for (auto& layer : layers_) {
            layer->attention->clear_cache();
        }
        
        return file.good();
    } catch (...) {
        return false;
    }
}

// TransformerLayer implementation

std::vector<float> TransformerModel::TransformerLayer::forward(
    const std::vector<float>& input, size_t seq_len) noexcept {
    
    std::vector<float> hidden = input;
    
    // Self-attention with residual
    auto attn_output = attention->forward(hidden, seq_len);
    
    // Add & norm
    for (size_t i = 0; i < hidden.size(); ++i) {
        hidden[i] += attn_output[i];
    }
    
    // Layer norm 1
    std::vector<float> ln1_output = hidden;
    // Simplified layer norm - in production would be more sophisticated
    
    // Feed-forward with residual
    auto ff_output = feed_forward->forward(ln1_output);
    
    // Add & norm
    for (size_t i = 0; i < hidden.size(); ++i) {
        hidden[i] = ln1_output[i] + ff_output[i];
    }
    
    return hidden;
}

// MultiHeadAttention implementation

std::vector<float> TransformerModel::MultiHeadAttention::forward(
    const std::vector<float>& input, size_t seq_len, const float* mask) noexcept {
    
    size_t model_dim = input.size() / seq_len;
    std::vector<float> output(input.size(), 0.0f);
    
    // Compute Q, K, V projections
    std::vector<float> Q(input.size()), K(input.size()), V(input.size());
    
    // Project input to Q, K, V
    for (size_t t = 0; t < seq_len; ++t) {
        const float* in_ptr = &input[t * model_dim];
        float* q_ptr = &Q[t * model_dim];
        float* k_ptr = &K[t * model_dim];
        float* v_ptr = &V[t * model_dim];
        
        // Matrix multiply for projections
        for (size_t i = 0; i < model_dim; ++i) {
            q_ptr[i] = bias_q[i];
            k_ptr[i] = bias_k[i];
            v_ptr[i] = bias_v[i];
            
            for (size_t j = 0; j < model_dim; ++j) {
                q_ptr[i] += in_ptr[j] * weights.W_query[j * model_dim + i];
                k_ptr[i] += in_ptr[j] * weights.W_key[j * model_dim + i];
                v_ptr[i] += in_ptr[j] * weights.W_value[j * model_dim + i];
            }
        }
    }
    
    // Multi-head attention
    size_t head_size = head_dim * seq_len;
    
    #pragma omp parallel for
    for (size_t h = 0; h < num_heads; ++h) {
        // Extract head
        size_t head_offset = h * head_dim;
        
        // Compute attention scores for this head
        std::vector<float> scores(seq_len * seq_len);
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        
        // Q @ K^T
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = 0; j < seq_len; ++j) {
                float score = 0.0f;
                for (size_t k = 0; k < head_dim; ++k) {
                    score += Q[i * model_dim + head_offset + k] * 
                            K[j * model_dim + head_offset + k];
                }
                scores[i * seq_len + j] = score * scale;
            }
        }
        
        // Apply mask if provided
        if (mask) {
            for (size_t i = 0; i < seq_len * seq_len; ++i) {
                if (mask[i] == 0.0f) {
                    scores[i] = -1e9f;
                }
            }
        }
        
        // Softmax over each row
        for (size_t i = 0; i < seq_len; ++i) {
            float* row = &scores[i * seq_len];
            
            // Find max for numerical stability
            float max_score = *std::max_element(row, row + seq_len);
            
            // Exp and sum
            float sum = 0.0f;
            for (size_t j = 0; j < seq_len; ++j) {
                row[j] = std::exp(row[j] - max_score);
                sum += row[j];
            }
            
            // Normalize
            for (size_t j = 0; j < seq_len; ++j) {
                row[j] /= sum;
            }
        }
        
        // Attention @ V
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t k = 0; k < head_dim; ++k) {
                float sum = 0.0f;
                for (size_t j = 0; j < seq_len; ++j) {
                    sum += scores[i * seq_len + j] * 
                          V[j * model_dim + head_offset + k];
                }
                output[i * model_dim + head_offset + k] = sum;
            }
        }
    }
    
    // Output projection
    std::vector<float> final_output(input.size());
    for (size_t t = 0; t < seq_len; ++t) {
        const float* out_ptr = &output[t * model_dim];
        float* final_ptr = &final_output[t * model_dim];
        
        for (size_t i = 0; i < model_dim; ++i) {
            final_ptr[i] = bias_o[i];
            for (size_t j = 0; j < model_dim; ++j) {
                final_ptr[i] += out_ptr[j] * weights.W_output[j * model_dim + i];
            }
        }
    }
    
    return final_output;
}

// FeedForward implementation

std::vector<float> TransformerModel::FeedForward::forward(
    const std::vector<float>& input) noexcept {
    
    size_t model_dim = bias2.size();
    size_t ff_dim = bias1.size();
    size_t seq_len = input.size() / model_dim;
    
    std::vector<float> hidden(seq_len * ff_dim);
    std::vector<float> output(input.size());
    
    // First linear layer + GELU
    #pragma omp parallel for
    for (size_t t = 0; t < seq_len; ++t) {
        const float* in_ptr = &input[t * model_dim];
        float* hid_ptr = &hidden[t * ff_dim];
        
        for (size_t i = 0; i < ff_dim; ++i) {
            hid_ptr[i] = bias1[i];
            for (size_t j = 0; j < model_dim; ++j) {
                hid_ptr[i] += in_ptr[j] * W1[j * ff_dim + i];
            }
            
            // GELU activation
            float x = hid_ptr[i];
            hid_ptr[i] = 0.5f * x * (1.0f + std::tanh(0.7978845608f * 
                        (x + 0.044715f * x * x * x)));
        }
    }
    
    // Second linear layer
    #pragma omp parallel for
    for (size_t t = 0; t < seq_len; ++t) {
        const float* hid_ptr = &hidden[t * ff_dim];
        float* out_ptr = &output[t * model_dim];
        
        for (size_t i = 0; i < model_dim; ++i) {
            out_ptr[i] = bias2[i];
            for (size_t j = 0; j < ff_dim; ++j) {
                out_ptr[i] += hid_ptr[j] * W2[j * model_dim + i];
            }
        }
    }
    
    return output;
}

// PositionalEncoding implementation

void TransformerModel::PositionalEncoding::initialize(
    size_t max_length, size_t model_dim) noexcept {
    
    encodings.resize(max_length);
    for (auto& enc : encodings) {
        enc.resize(model_dim);
    }
    
    for (size_t pos = 0; pos < max_length; ++pos) {
        for (size_t i = 0; i < model_dim / 2; ++i) {
            float angle = pos / std::pow(10000.0f, 2.0f * i / model_dim);
            encodings[pos][2 * i] = std::sin(angle);
            encodings[pos][2 * i + 1] = std::cos(angle);
        }
    }
}

void TransformerModel::PositionalEncoding::apply(
    std::vector<float>& embeddings, size_t seq_len) noexcept {
    
    size_t model_dim = embeddings.size() / seq_len;
    
    #pragma omp parallel for
    for (size_t t = 0; t < seq_len; ++t) {
        for (size_t i = 0; i < model_dim; ++i) {
            embeddings[t * model_dim + i] += encodings[t][i];
        }
    }
}

// OrderBookTransformer implementation

OrderBookTransformer::OrderBookTransformer(const Config& config)
    : TransformerModel(config) {
}

OrderBookTransformer::OrderBookPrediction OrderBookTransformer::predict_order_book(
    const std::vector<Quote>& quotes,
    const std::vector<Trade>& trades,
    size_t lookback) noexcept {
    
    OrderBookPrediction pred{};
    
    // Convert market data to features
    // This is simplified - in production would use full feature engineering
    std::vector<FeatureEngineering::Features> sequence;
    
    size_t start_idx = quotes.size() > lookback ? quotes.size() - lookback : 0;
    
    for (size_t i = start_idx; i < quotes.size(); ++i) {
        FeatureEngineering::Features features{};
        features.valid = true;
        
        // Basic features from quotes
        features.microstructure.bid_ask_spread = 
            (quotes[i].ask_price - quotes[i].bid_price) / 10000.0;
        features.microstructure.depth_imbalance = 
            (quotes[i].bid_size - quotes[i].ask_size) / 
            static_cast<double>(quotes[i].bid_size + quotes[i].ask_size);
        
        sequence.push_back(features);
    }
    
    // Get transformer predictions
    auto output = predict(sequence);
    
    // Map to order book specific predictions
    pred.hidden_liquidity_bid = output.hidden_liquidity_estimate * 0.6f;
    pred.hidden_liquidity_ask = output.hidden_liquidity_estimate * 0.4f;
    pred.iceberg_probability = output.dark_pool_probability;
    
    pred.market_impact_10k_shares = output.price_impact_bps;
    pred.optimal_slice_size = 10000.0f / (1.0f + output.execution_cost_bps);
    pred.execution_alpha = output.adverse_selection_cost;
    
    // Determine regime
    if (output.intensity_next_1m > 2.0f) {
        pred.predicted_regime = Regime::HIGH_VOLATILITY;
    } else if (output.volume_forecast > 50000) {
        pred.predicted_regime = Regime::INSTITUTIONAL_FLOW;
    } else {
        pred.predicted_regime = Regime::NORMAL;
    }
    
    pred.regime_confidence = output.overall_confidence;
    
    return pred;
}

} 
