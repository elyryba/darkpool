#include "darkpool/ml/elastic_net.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <fstream>
#include <thread>
#include <immintrin.h>

namespace darkpool::ml {

ElasticNet::ElasticNet(const Config& config)
    : config_(config)
    , state_(std::make_unique<ModelState>())
    , solver_(std::make_unique<CoordinateDescent>()) {
    
    state_->coefficients.resize(MAX_FEATURES, 0.0);
    state_->feature_means.resize(MAX_FEATURES, 0.0);
    state_->feature_stds.resize(MAX_FEATURES, 1.0);
    state_->coefficient_variance.resize(MAX_FEATURES, 0.0);
    state_->intercept = 0.0;
}

ElasticNet::FitResult ElasticNet::fit(
    const std::vector<FeatureEngineering::Features>& X,
    const std::vector<double>& y) noexcept {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    FitResult result{};
    
    if (X.empty() || y.empty() || X.size() != y.size()) {
        return result;
    }
    
    size_t n_samples = X.size();
    size_t n_features = MAX_FEATURES;
    
    // Convert features to dense matrix
    std::vector<double> X_dense(n_samples * n_features);
    for (size_t i = 0; i < n_samples; ++i) {
        std::memcpy(&X_dense[i * n_features], X[i].data(), 
                   n_features * sizeof(double));
    }
    
    // Standardize features if requested
    if (config_.normalize) {
        standardize_features(X_dense.data(), n_samples, n_features);
    }
    
    // Choose solver
    if (n_samples > 10000 && config_.batch_size > 0) {
        result = fit_sgd(X_dense.data(), y.data(), n_samples, n_features);
    } else {
        result = fit_coordinate_descent(X_dense.data(), y.data(), n_samples, n_features);
    }
    
    // Calculate feature importance
    result.feature_importance.resize(n_features);
    double coef_sum = 0.0;
    for (size_t i = 0; i < n_features; ++i) {
        result.feature_importance[i] = std::abs(state_->coefficients[i]);
        coef_sum += result.feature_importance[i];
    }
    
    // Normalize importance
    if (coef_sum > 0) {
        for (auto& imp : result.feature_importance) {
            imp /= coef_sum;
        }
    }
    
    // Get selected features
    result.selected_features = get_active_features();
    result.sparsity = 1.0 - static_cast<double>(result.selected_features.size()) / n_features;
    
    // Calculate R²
    double y_mean = std::accumulate(y.begin(), y.end(), 0.0) / n_samples;
    double ss_tot = 0.0, ss_res = 0.0;
    
    for (size_t i = 0; i < n_samples; ++i) {
        double pred = predict(X[i]);
        ss_res += std::pow(y[i] - pred, 2);
        ss_tot += std::pow(y[i] - y_mean, 2);
    }
    
    result.r_squared = 1.0 - (ss_res / ss_tot);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.fit_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time).count();
    
    // Update model state
    state_->n_samples = n_samples;
    state_->residual_variance = ss_res / (n_samples - result.selected_features.size() - 1);
    state_->version.fetch_add(1);
    
    return result;
}

ElasticNet::FitResult ElasticNet::fit_coordinate_descent(
    const double* X, const double* y,
    size_t n_samples, size_t n_features) noexcept {
    
    FitResult result{};
    
    // Initialize solver
    solver_->initialize(n_samples, n_features);
    
    // Compute initial values
    std::vector<double> Xty(n_features);
    std::vector<double> XtX_diag(n_features);
    
    // Compute X^T * y and diagonal of X^T * X
    #pragma omp parallel for num_threads(config_.n_jobs)
    for (size_t j = 0; j < n_features; ++j) {
        double sum_xy = 0.0;
        double sum_xx = 0.0;
        
        if (config_.use_simd) {
            __m256d sum_xy_vec = _mm256_setzero_pd();
            __m256d sum_xx_vec = _mm256_setzero_pd();
            
            for (size_t i = 0; i < n_samples - 3; i += 4) {
                __m256d x_vec = _mm256_loadu_pd(&X[i * n_features + j]);
                __m256d y_vec = _mm256_loadu_pd(&y[i]);
                
                sum_xy_vec = _mm256_fmadd_pd(x_vec, y_vec, sum_xy_vec);
                sum_xx_vec = _mm256_fmadd_pd(x_vec, x_vec, sum_xx_vec);
            }
            
            double xy_arr[4], xx_arr[4];
            _mm256_storeu_pd(xy_arr, sum_xy_vec);
            _mm256_storeu_pd(xx_arr, sum_xx_vec);
            
            for (int k = 0; k < 4; ++k) {
                sum_xy += xy_arr[k];
                sum_xx += xx_arr[k];
            }
            
            // Handle remaining elements
            for (size_t i = (n_samples / 4) * 4; i < n_samples; ++i) {
                double x_val = X[i * n_features + j];
                sum_xy += x_val * y[i];
                sum_xx += x_val * x_val;
            }
        } else {
            for (size_t i = 0; i < n_samples; ++i) {
                double x_val = X[i * n_features + j];
                sum_xy += x_val * y[i];
                sum_xx += x_val * x_val;
            }
        }
        
        Xty[j] = sum_xy;
        XtX_diag[j] = sum_xx;
    }
    
    // Initialize residuals
    std::vector<double> residuals(y, y + n_samples);
    
    // Coordinate descent iterations
    double prev_loss = std::numeric_limits<double>::max();
    
    for (size_t iter = 0; iter < config_.max_iterations; ++iter) {
        double max_change = 0.0;
        
        // Cycle through features
        for (size_t j = 0; j < n_features; ++j) {
            if (XtX_diag[j] < 1e-10) continue; // Skip constant features
            
            double old_coef = state_->coefficients[j];
            
            // Compute gradient
            double gradient = 0.0;
            for (size_t i = 0; i < n_samples; ++i) {
                gradient += X[i * n_features + j] * residuals[i];
            }
            
            // Update coefficient with soft thresholding
            double l1_penalty = config_.alpha * config_.l1_ratio * n_samples;
            double l2_penalty = config_.alpha * (1.0 - config_.l1_ratio) * n_samples;
            
            double numerator = gradient + old_coef * XtX_diag[j];
            double new_coef = solver_->soft_threshold(numerator, l1_penalty) / 
                            (XtX_diag[j] + l2_penalty);
            
            // Update residuals
            if (std::abs(new_coef - old_coef) > 1e-10) {
                double delta = new_coef - old_coef;
                
                #pragma omp simd
                for (size_t i = 0; i < n_samples; ++i) {
                    residuals[i] -= delta * X[i * n_features + j];
                }
                
                state_->coefficients[j] = new_coef;
                max_change = std::max(max_change, std::abs(delta));
            }
        }
        
        // Check convergence
        if (max_change < config_.tolerance) {
            result.converged = true;
            result.iterations = iter + 1;
            break;
        }
        
        // Compute loss every 10 iterations
        if (iter % 10 == 0) {
            double loss = compute_elastic_net_loss(X, residuals.data(), 
                                                 n_samples, n_features);
            if (std::abs(loss - prev_loss) < config_.tolerance) {
                result.converged = true;
                result.iterations = iter + 1;
                break;
            }
            prev_loss = loss;
        }
        
        result.iterations = iter + 1;
    }
    
    // Compute intercept if requested
    if (config_.fit_intercept) {
        double y_mean = std::accumulate(y, y + n_samples, 0.0) / n_samples;
        double x_dot_coef = 0.0;
        
        for (size_t j = 0; j < n_features; ++j) {
            x_dot_coef += state_->feature_means[j] * state_->coefficients[j];
        }
        
        state_->intercept = y_mean - x_dot_coef;
    }
    
    result.final_loss = compute_elastic_net_loss(X, residuals.data(), 
                                               n_samples, n_features);
    
    return result;
}

double ElasticNet::predict(const FeatureEngineering::Features& features) const noexcept {
    if (!features.valid) return 0.0;
    
    alignas(32) double features_copy[MAX_FEATURES];
    std::memcpy(features_copy, features.data(), MAX_FEATURES * sizeof(double));
    
    // Apply standardization
    if (config_.normalize) {
        apply_standardization(features_copy, MAX_FEATURES);
    }
    
    double result = state_->intercept;
    
    if (config_.use_simd) {
        __m256d sum_vec = _mm256_setzero_pd();
        
        for (size_t i = 0; i < MAX_FEATURES - 3; i += 4) {
            __m256d feat_vec = _mm256_loadu_pd(&features_copy[i]);
            __m256d coef_vec = _mm256_loadu_pd(&state_->coefficients[i]);
            sum_vec = _mm256_fmadd_pd(feat_vec, coef_vec, sum_vec);
        }
        
        double sum_arr[4];
        _mm256_storeu_pd(sum_arr, sum_vec);
        result += sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];
        
        // Handle remaining
        for (size_t i = (MAX_FEATURES / 4) * 4; i < MAX_FEATURES; ++i) {
            result += features_copy[i] * state_->coefficients[i];
        }
    } else {
        for (size_t i = 0; i < MAX_FEATURES; ++i) {
            result += features_copy[i] * state_->coefficients[i];
        }
    }
    
    return result;
}

std::vector<double> ElasticNet::predict_batch(
    const std::vector<FeatureEngineering::Features>& features) const noexcept {
    
    std::vector<double> predictions;
    predictions.reserve(features.size());
    
    for (const auto& feat : features) {
        predictions.push_back(predict(feat));
    }
    
    return predictions;
}

void ElasticNet::standardize_features(double* X, size_t n_samples, 
                                     size_t n_features) noexcept {
    // Compute means and stds
    #pragma omp parallel for num_threads(config_.n_jobs)
    for (size_t j = 0; j < n_features; ++j) {
        double sum = 0.0, sum_sq = 0.0;
        
        for (size_t i = 0; i < n_samples; ++i) {
            double val = X[i * n_features + j];
            sum += val;
            sum_sq += val * val;
        }
        
        double mean = sum / n_samples;
        double variance = (sum_sq / n_samples) - (mean * mean);
        double std = std::sqrt(std::max(variance, 1e-10));
        
        state_->feature_means[j] = mean;
        state_->feature_stds[j] = std;
        
        // Standardize in place
        for (size_t i = 0; i < n_samples; ++i) {
            X[i * n_features + j] = (X[i * n_features + j] - mean) / std;
        }
    }
}

void ElasticNet::apply_standardization(double* features, size_t n_features) const noexcept {
    if (config_.use_simd) {
        for (size_t i = 0; i < n_features - 3; i += 4) {
            __m256d feat_vec = _mm256_loadu_pd(&features[i]);
            __m256d mean_vec = _mm256_loadu_pd(&state_->feature_means[i]);
            __m256d std_vec = _mm256_loadu_pd(&state_->feature_stds[i]);
            
            feat_vec = _mm256_sub_pd(feat_vec, mean_vec);
            feat_vec = _mm256_div_pd(feat_vec, std_vec);
            
            _mm256_storeu_pd(&features[i], feat_vec);
        }
        
        // Handle remaining
        for (size_t i = (n_features / 4) * 4; i < n_features; ++i) {
            features[i] = (features[i] - state_->feature_means[i]) / state_->feature_stds[i];
        }
    } else {
        for (size_t i = 0; i < n_features; ++i) {
            features[i] = (features[i] - state_->feature_means[i]) / state_->feature_stds[i];
        }
    }
}

double ElasticNet::compute_elastic_net_loss(const double* X, const double* residuals,
                                          size_t n_samples, size_t n_features) const noexcept {
    // MSE loss
    double mse = 0.0;
    for (size_t i = 0; i < n_samples; ++i) {
        mse += residuals[i] * residuals[i];
    }
    mse /= (2.0 * n_samples);
    
    // L1 penalty
    double l1_penalty = 0.0;
    for (size_t j = 0; j < n_features; ++j) {
        l1_penalty += std::abs(state_->coefficients[j]);
    }
    l1_penalty *= config_.alpha * config_.l1_ratio;
    
    // L2 penalty
    double l2_penalty = 0.0;
    for (size_t j = 0; j < n_features; ++j) {
        l2_penalty += state_->coefficients[j] * state_->coefficients[j];
    }
    l2_penalty *= 0.5 * config_.alpha * (1.0 - config_.l1_ratio);
    
    return mse + l1_penalty + l2_penalty;
}

std::vector<size_t> ElasticNet::get_active_features(double threshold) const noexcept {
    std::vector<size_t> active;
    
    for (size_t i = 0; i < MAX_FEATURES; ++i) {
        if (std::abs(state_->coefficients[i]) > threshold) {
            active.push_back(i);
        }
    }
    
    return active;
}

std::vector<double> ElasticNet::get_coefficients() const noexcept {
    return state_->coefficients;
}

double ElasticNet::get_intercept() const noexcept {
    return state_->intercept;
}

std::vector<size_t> ElasticNet::get_selected_features() const noexcept {
    return get_active_features();
}

std::vector<double> ElasticNet::get_feature_importance() const noexcept {
    std::vector<double> importance(MAX_FEATURES);
    double sum = 0.0;
    
    for (size_t i = 0; i < MAX_FEATURES; ++i) {
        importance[i] = std::abs(state_->coefficients[i]);
        sum += importance[i];
    }
    
    if (sum > 0) {
        for (auto& imp : importance) {
            imp /= sum;
        }
    }
    
    return importance;
}

bool ElasticNet::save_model(const std::string& path) const noexcept {
    try {
        std::ofstream file(path, std::ios::binary);
        if (!file) return false;
        
        // Write config
        file.write(reinterpret_cast<const char*>(&config_), sizeof(Config));
        
        // Write model state
        file.write(reinterpret_cast<const char*>(&state_->intercept), sizeof(double));
        file.write(reinterpret_cast<const char*>(state_->coefficients.data()), 
                  MAX_FEATURES * sizeof(double));
        file.write(reinterpret_cast<const char*>(state_->feature_means.data()), 
                  MAX_FEATURES * sizeof(double));
        file.write(reinterpret_cast<const char*>(state_->feature_stds.data()), 
                  MAX_FEATURES * sizeof(double));
        
        return file.good();
    } catch (...) {
        return false;
    }
}

bool ElasticNet::load_model(const std::string& path) noexcept {
    try {
        std::ifstream file(path, std::ios::binary);
        if (!file) return false;
        
        // Read config
        file.read(reinterpret_cast<char*>(&config_), sizeof(Config));
        
        // Read model state
        file.read(reinterpret_cast<char*>(&state_->intercept), sizeof(double));
        file.read(reinterpret_cast<char*>(state_->coefficients.data()), 
                 MAX_FEATURES * sizeof(double));
        file.read(reinterpret_cast<char*>(state_->feature_means.data()), 
                 MAX_FEATURES * sizeof(double));
        file.read(reinterpret_cast<char*>(state_->feature_stds.data()), 
                 MAX_FEATURES * sizeof(double));
        
        state_->version.fetch_add(1);
        
        return file.good();
    } catch (...) {
        return false;
    }
}

// CoordinateDescent implementation

void ElasticNet::CoordinateDescent::initialize(size_t n_samples, 
                                              size_t n_features) noexcept {
    residuals.resize(n_samples);
    XtX_diag.resize(n_features);
    Xty.resize(n_features);
    active_set.resize(n_features, 1.0);
}

double ElasticNet::CoordinateDescent::soft_threshold(double value, 
                                                    double threshold) const noexcept {
    if (value > threshold) {
        return value - threshold;
    } else if (value < -threshold) {
        return value + threshold;
    } else {
        return 0.0;
    }
}

// FastLinearScorer implementation

FastLinearScorer::FastLinearScorer(const ElasticNet& model) {
    auto coefs = model.get_coefficients();
    coefficients_.resize(coefs.size());
    
    // Convert to float for speed
    for (size_t i = 0; i < coefs.size(); ++i) {
        coefficients_[i] = static_cast<float>(coefs[i]);
    }
    
    intercept_ = static_cast<float>(model.get_intercept());
    
    // Create feature mask
    feature_mask_.resize(coefs.size());
    auto active = model.get_selected_features();
    for (size_t idx : active) {
        feature_mask_[idx] = 1;
    }
    
    // Define decision rules
    rules_.push_back({0x00000001, 0.8f, AnomalyType::DARK_POOL_ACTIVITY, 0.2f});
    rules_.push_back({0x00000002, 0.7f, AnomalyType::HIDDEN_LIQUIDITY, 0.15f});
    rules_.push_back({0x00000004, 0.6f, AnomalyType::UNUSUAL_VOLUME, 0.1f});
}

FastLinearScorer::ScoringResult FastLinearScorer::score(
    const FeatureEngineering::Features& features) const noexcept {
    
    ScoringResult result{};
    
    // Fast linear scoring with SIMD
    __m256 sum_vec = _mm256_setzero_ps();
    
    for (size_t i = 0; i < coefficients_.size() - 7; i += 8) {
        if (feature_mask_[i] == 0) continue;
        
        // Load features (convert double to float)
        __m256d feat_d1 = _mm256_loadu_pd(&features.values[i]);
        __m256d feat_d2 = _mm256_loadu_pd(&features.values[i + 4]);
        __m128 feat_f1 = _mm256_cvtpd_ps(feat_d1);
        __m128 feat_f2 = _mm256_cvtpd_ps(feat_d2);
        __m256 feat_vec = _mm256_set_m128(feat_f2, feat_f1);
        
        __m256 coef_vec = _mm256_loadu_ps(&coefficients_[i]);
        sum_vec = _mm256_fmadd_ps(feat_vec, coef_vec, sum_vec);
    }
    
    // Horizontal sum
    __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum_low = _mm256_castps256_ps128(sum_vec);
    __m128 sum_128 = _mm_add_ps(sum_low, sum_high);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    
    result.score = _mm_cvtss_f32(sum_128) + intercept_;
    
    // Apply decision rules
    result.anomaly_type = AnomalyType::NONE;
    result.confidence = 1.0 / (1.0 + std::exp(-result.score)); // Sigmoid
    
    for (const auto& rule : rules_) {
        if (result.score > rule.threshold) {
            result.anomaly_type = rule.type;
            result.confidence += rule.confidence_boost;
            result.triggered_rules |= rule.feature_mask;
        }
    }
    
    result.confidence = std::min(result.confidence, 1.0);
    
    return result;
}

} 
