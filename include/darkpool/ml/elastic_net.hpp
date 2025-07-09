#pragma once

#include "darkpool/types.hpp"
#include "darkpool/ml/feature_engineering.hpp"
#include <vector>
#include <atomic>
#include <memory>

namespace darkpool::ml {

class ElasticNet {
public:
    static constexpr size_t MAX_FEATURES = FeatureEngineering::NUM_FEATURES;
    static constexpr size_t MAX_ITERATIONS = 10000;
    
    struct Config {
        double alpha = 1.0;        // Overall regularization strength
        double l1_ratio = 0.5;     // 0 = Ridge, 1 = Lasso, 0.5 = balanced
        double tolerance = 1e-4;   // Convergence tolerance
        size_t max_iterations = 1000;
        bool fit_intercept = true;
        bool normalize = true;     // Standardize features
        bool warm_start = false;   // Use previous solution as init
        size_t n_jobs = 4;        // Parallel threads for coordinate descent
        double learning_rate = 0.01;
        bool use_simd = true;     // SIMD optimizations
        size_t batch_size = 32;   // Mini-batch size for SGD variant
    };
    
    struct FitResult {
        bool converged;
        size_t iterations;
        double final_loss;
        double r_squared;
        std::vector<double> feature_importance;
        std::vector<size_t> selected_features;  // Non-zero coefficients
        double sparsity;  // Percentage of zero coefficients
        int64_t fit_time_us;
    };
    
    struct PredictionResult {
        double value;
        double confidence_interval_lower;
        double confidence_interval_upper;
        double prediction_std;
        std::vector<double> feature_contributions;  // SHAP-like values
    };
    
    explicit ElasticNet(const Config& config = {});
    ~ElasticNet() = default;
    
    // Training
    FitResult fit(const std::vector<FeatureEngineering::Features>& X,
                  const std::vector<double>& y) noexcept;
    
    // Incremental learning
    FitResult partial_fit(const std::vector<FeatureEngineering::Features>& X,
                         const std::vector<double>& y) noexcept;
    
    // Prediction
    double predict(const FeatureEngineering::Features& features) const noexcept;
    std::vector<double> predict_batch(
        const std::vector<FeatureEngineering::Features>& features) const noexcept;
    
    // Prediction with uncertainty
    PredictionResult predict_with_confidence(
        const FeatureEngineering::Features& features) const noexcept;
    
    // Model inspection
    std::vector<double> get_coefficients() const noexcept;
    double get_intercept() const noexcept;
    std::vector<size_t> get_selected_features() const noexcept;
    std::vector<double> get_feature_importance() const noexcept;
    
    // Model persistence
    bool save_model(const std::string& path) const noexcept;
    bool load_model(const std::string& path) noexcept;
    
    // Cross-validation
    struct CVResult {
        std::vector<double> train_scores;
        std::vector<double> val_scores;
        double mean_train_score;
        double mean_val_score;
        double std_train_score;
        double std_val_score;
        Config best_params;
    };
    
    CVResult cross_validate(const std::vector<FeatureEngineering::Features>& X,
                           const std::vector<double>& y,
                           size_t n_folds = 5) const noexcept;

private:
    struct ModelState {
        alignas(64) std::vector<double> coefficients;
        alignas(64) std::vector<double> feature_means;
        alignas(64) std::vector<double> feature_stds;
        alignas(64) double intercept;
        alignas(64) std::atomic<uint64_t> version{0};
        
        // For confidence intervals
        std::vector<double> coefficient_variance;
        double residual_variance;
        size_t n_samples;
        
        // Lock-free update support
        std::atomic<bool> updating{false};
    };
    
    // Coordinate descent solver
    struct CoordinateDescent {
        alignas(64) std::vector<double> residuals;
        alignas(64) std::vector<double> XtX_diag;  // Feature variances
        alignas(64) std::vector<double> Xty;       // Feature-target correlations
        alignas(64) std::vector<double> active_set;
        
        void initialize(size_t n_samples, size_t n_features) noexcept;
        void update_residuals(const double* X, size_t feature_idx,
                            double delta_coef, size_t n_samples) noexcept;
        double soft_threshold(double value, double threshold) const noexcept;
    };
    
    Config config_;
    std::unique_ptr<ModelState> state_;
    mutable std::unique_ptr<CoordinateDescent> solver_;
    
    // Training methods
    FitResult fit_coordinate_descent(const double* X, const double* y,
                                    size_t n_samples, size_t n_features) noexcept;
    
    FitResult fit_sgd(const double* X, const double* y,
                     size_t n_samples, size_t n_features) noexcept;
    
    // Feature preprocessing
    void standardize_features(double* X, size_t n_samples, 
                            size_t n_features) noexcept;
    void apply_standardization(double* features, size_t n_features) const noexcept;
    
    // Loss computation
    double compute_elastic_net_loss(const double* X, const double* y,
                                   size_t n_samples, size_t n_features) const noexcept;
    
    // SIMD optimized operations
    void simd_dot_product(const double* a, const double* b, 
                         double& result, size_t n) const noexcept;
    void simd_axpy(double alpha, const double* x, double* y, size_t n) const noexcept;
    void simd_scale(double* x, double alpha, size_t n) const noexcept;
    
    // Feature selection
    std::vector<size_t> get_active_features(double threshold = 1e-10) const noexcept;
    
    // Parallel coordinate descent
    void parallel_coordinate_update(const double* X, const double* y,
                                   size_t start_idx, size_t end_idx,
                                   size_t n_samples) noexcept;
};

// Fast linear model for real-time scoring
class FastLinearScorer {
public:
    struct ScoringResult {
        double score;
        AnomalyType anomaly_type;
        double confidence;
        uint32_t triggered_rules;
    };
    
    explicit FastLinearScorer(const ElasticNet& model);
    
    // Ultra-fast scoring with pre-computed lookups
    ScoringResult score(const FeatureEngineering::Features& features) const noexcept;
    
    // Batch scoring with SIMD
    std::vector<ScoringResult> score_batch(
        const std::vector<FeatureEngineering::Features>& features) const noexcept;
    
private:
    alignas(64) std::vector<float> coefficients_;  // Reduced precision for speed
    alignas(64) std::vector<float> thresholds_;    // Decision thresholds
    alignas(64) std::vector<uint8_t> feature_mask_; // Active features
    float intercept_;
    
    // Pre-computed decision rules
    struct DecisionRule {
        uint32_t feature_mask;
        float threshold;
        AnomalyType type;
        float confidence_boost;
    };
    std::vector<DecisionRule> rules_;
};

} 
