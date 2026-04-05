#pragma once

#include "mlp_network.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

class NetworkBlueprint {
public:
    NetworkBlueprint &insert_linear(std::size_t output_units);
    NetworkBlueprint &insert_activation(ActivationType activation);
    NetworkBlueprint &insert_relu() { return insert_activation(ActivationType::Relu); }
    NetworkBlueprint &insert_sigmoid() { return insert_activation(ActivationType::Sigmoid); }

    std::vector<OperationConfig> build(std::size_t input_features) const;

private:
    struct Step {
        OperationType type;
        std::size_t output_units = 0;
        ActivationType activation = ActivationType::Linear;
    };

    std::vector<Step> steps_;
};

struct TrainingConfig {
    std::string model_type = "mlp";
    int schema_version = 1;
    std::filesystem::path dataset_path;
    std::filesystem::path load_model_path;
    std::filesystem::path save_model_path;
    std::filesystem::path import_config_path;
    std::filesystem::path import_network_config_path;
    std::filesystem::path export_config_path;
    std::filesystem::path export_network_config_path;
    ExecutionBackend backend = ExecutionBackend::GPU;
    LossType loss = LossType::BCE;
    float learning_rate = 0.01f;
    float lr_decay = 1.0f;
    std::size_t lr_decay_every = 1;
    float min_learning_rate = 0.0f;
    std::filesystem::path results_csv_path;
    std::filesystem::path pr_csv_path;
    float positive_class_weight = 1.0f;
    float negative_class_weight = 1.0f;
    float threshold = 0.5f;
    float pr_scan_min = 0.0f;
    float pr_scan_max = 1.0f;
    float pr_scan_step = 0.02f;
    std::size_t batch_size = 1;
    std::size_t epochs = 20;
    std::size_t print_every = 1;
    bool export_only = false;
    bool eval_only = false;
    bool auto_class_weights = false;
};

struct ClassificationMetrics {
    std::uint64_t tp = 0;
    std::uint64_t fp = 0;
    std::uint64_t tn = 0;
    std::uint64_t fn = 0;
    float accuracy = 0.0f;
    float precision = 0.0f;
    float recall = 0.0f;
    float specificity = 0.0f;
    float f1 = 0.0f;
};

struct TrainingRunResult {
    float initial_cost = 0.0f;
    float final_eval_cost = 0.0f;
    float average_epoch_train_loss = 0.0f;
    std::uint64_t total_training_ms = 0;
    ClassificationMetrics final_metrics;
};

TrainingRunResult run_training_pipeline(const TrainingConfig &config,
                                        const NetworkBlueprint &blueprint);
NetworkBlueprint make_default_mlp_blueprint();
