#include "training_pipeline.h"

#include "defect_dataset.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

namespace {

const char *backend_name(ExecutionBackend backend) {
    return backend == ExecutionBackend::GPU ? "GPU" : "CPU";
}

const char *optimizer_name(OptimizerType optimizer) {
    switch (optimizer) {
        case OptimizerType::SGD:
            return "SGD";
        case OptimizerType::Momentum:
            return "Momentum";
        case OptimizerType::Adam:
            return "Adam";
    }
    return "Unknown";
}

const char *activation_name(ActivationType activation) {
    switch (activation) {
        case ActivationType::Linear:
            return "linear";
        case ActivationType::Relu:
            return "relu";
        case ActivationType::Sigmoid:
            return "sigmoid";
        case ActivationType::Tanh:
            return "tanh";
        case ActivationType::LeakyRelu:
            return "leaky_relu";
        case ActivationType::Gelu:
            return "gelu";
    }
    return "unknown";
}

}  // namespace

NetworkBlueprint &NetworkBlueprint::insert_linear(std::size_t output_units) {
    if (output_units == 0) {
        throw std::invalid_argument("NetworkBlueprint::insert_linear requires a positive output size");
    }
    steps_.push_back(Step{OperationType::Linear, output_units, ActivationType::Linear});
    return *this;
}

NetworkBlueprint &NetworkBlueprint::insert_activation(ActivationType activation) {
    steps_.push_back(Step{OperationType::Relu, 0, activation});
    return *this;
}

std::vector<OperationConfig> NetworkBlueprint::build(std::size_t input_features) const {
    if (input_features == 0) {
        throw std::invalid_argument("NetworkBlueprint::build requires a positive input feature count");
    }
    if (steps_.empty()) {
        throw std::invalid_argument("NetworkBlueprint is empty");
    }

    std::vector<OperationConfig> operations;
    std::size_t current_input = input_features;
    bool seen_linear = false;

    for (const Step &step : steps_) {
        if (step.type == OperationType::Linear) {
            operations.push_back(OperationConfig::linear(current_input, step.output_units));
            current_input = step.output_units;
            seen_linear = true;
            continue;
        }

        if (!seen_linear) {
            throw std::invalid_argument("NetworkBlueprint activations must follow at least one linear layer");
        }

        if (step.activation == ActivationType::Relu) {
            operations.push_back(OperationConfig::relu());
        } else if (step.activation == ActivationType::Sigmoid) {
            operations.push_back(OperationConfig::sigmoid());
        } else if (step.activation == ActivationType::Tanh) {
            operations.push_back(OperationConfig::tanh());
        } else if (step.activation == ActivationType::LeakyRelu) {
            operations.push_back(OperationConfig::leaky_relu());
        } else if (step.activation == ActivationType::Gelu) {
            operations.push_back(OperationConfig::gelu());
        } else {
            throw std::invalid_argument("Unsupported activation in NetworkBlueprint");
        }
    }

    return operations;
}

NetworkBlueprint make_default_mlp_blueprint(ActivationType hidden_activation,
                                            ActivationType output_activation) {
    NetworkBlueprint blueprint;
    blueprint.insert_linear(14)
        .insert_activation(hidden_activation)
        .insert_linear(10)
        .insert_activation(hidden_activation)
        .insert_linear(8)
        .insert_activation(hidden_activation)
        .insert_linear(1)
        .insert_activation(output_activation);
    return blueprint;
}

namespace {

const char *loss_name(LossType loss) {
    return loss == LossType::BCE ? "BCE" : "MSE";
}

float safe_div(std::uint64_t numerator, std::uint64_t denominator) {
    if (denominator == 0) {
        return 0.0f;
    }
    return static_cast<float>(numerator) / static_cast<float>(denominator);
}

void compute_auto_class_weights(const ManufacturingDefectDataset &dataset,
                                float &positive_weight,
                                float &negative_weight)
{
    std::uint64_t positives = 0;
    std::uint64_t negatives = 0;

    for (const auto &sample : dataset.samples()) {
        if (sample.label == 1) {
            ++positives;
        } else {
            ++negatives;
        }
    }

    if (positives == 0 || negatives == 0) {
        throw std::runtime_error("Auto class weights require both classes in the dataset");
    }
    std::cout << "Dataset class distribution: positives=" << positives << ", negatives=" << negatives << "\n";
    const std::uint64_t total = positives + negatives;
    positive_weight = static_cast<float>(total) / (2.0f * static_cast<float>(positives));
    negative_weight = static_cast<float>(total) / (2.0f * static_cast<float>(negatives));
}

ClassificationMetrics evaluate_classification_metrics(
    MlpNetwork &network,
    const ManufacturingDefectDataset &dataset,
    float threshold)
{
    ClassificationMetrics metrics;

    for (const auto &sample : dataset.samples()) {
        const std::vector<float> output = network.forward(sample.features);
        const float probability = output.empty() ? 0.0f : output[0];
        const int prediction = (probability >= threshold) ? 1 : 0;
        const int label = static_cast<int>(sample.label);

        if (prediction == 1 && label == 1) {
            ++metrics.tp;
        } else if (prediction == 1 && label == 0) {
            ++metrics.fp;
        } else if (prediction == 0 && label == 0) {
            ++metrics.tn;
        } else {
            ++metrics.fn;
        }
    }

    const std::uint64_t total = metrics.tp + metrics.fp + metrics.tn + metrics.fn;
    metrics.accuracy = safe_div(metrics.tp + metrics.tn, total);
    metrics.precision = safe_div(metrics.tp, metrics.tp + metrics.fp);
    metrics.recall = safe_div(metrics.tp, metrics.tp + metrics.fn);
    metrics.specificity = safe_div(metrics.tn, metrics.tn + metrics.fp);
    const float pr_sum = metrics.precision + metrics.recall;
    metrics.f1 = pr_sum > 0.0f
        ? 2.0f * (metrics.precision * metrics.recall) / pr_sum
        : 0.0f;

    return metrics;
}

struct ThresholdMetricRow {
    float threshold = 0.0f;
    ClassificationMetrics metrics;
};

struct PrCurveCsvWriter {
    explicit PrCurveCsvWriter(const std::filesystem::path &path)
        : out(path, std::ios::out | std::ios::trunc) {
        if (!out.is_open()) {
            throw std::runtime_error("Failed to open PR CSV for write: " + path.string());
        }
        out << "threshold,precision,recall,f1,specificity,accuracy,tp,fp,tn,fn\n";
    }

    void write_row(float threshold, const ClassificationMetrics &metrics) {
        out << threshold << ','
            << metrics.precision << ','
            << metrics.recall << ','
            << metrics.f1 << ','
            << metrics.specificity << ','
            << metrics.accuracy << ','
            << metrics.tp << ','
            << metrics.fp << ','
            << metrics.tn << ','
            << metrics.fn << '\n';
        if (!out.good()) {
            throw std::runtime_error("Failed to write PR CSV row");
        }
    }

    std::ofstream out;
};

std::vector<ThresholdMetricRow> scan_pr_curve(MlpNetwork &network,
                                              const ManufacturingDefectDataset &dataset,
                                              float min_threshold,
                                              float max_threshold,
                                              float step)
{
    std::vector<ThresholdMetricRow> rows;
    for (float threshold = min_threshold; threshold <= max_threshold + 1e-6f; threshold += step) {
        rows.push_back({threshold, evaluate_classification_metrics(network, dataset, threshold)});
    }
    return rows;
}

void print_classification_metrics(const ClassificationMetrics &metrics) {
    std::cout << "Confusion matrix: TP=" << metrics.tp
              << ", FP=" << metrics.fp
              << ", TN=" << metrics.tn
              << ", FN=" << metrics.fn << "\n";

    std::cout << std::fixed << std::setprecision(6)
              << "Accuracy=" << metrics.accuracy
              << ", Precision=" << metrics.precision
              << ", Recall=" << metrics.recall
              << ", Specificity=" << metrics.specificity
              << ", F1=" << metrics.f1 << "\n";
    std::cout.unsetf(std::ios::floatfield);
}

struct ResultsCsvWriter {
    explicit ResultsCsvWriter(const std::filesystem::path &path)
        : out(path, std::ios::out | std::ios::trunc) {
        if (!out.is_open()) {
            throw std::runtime_error("Failed to open results CSV for write: " + path.string());
        }
        out << "phase,epoch,train_loss,eval_cost,tp,fp,tn,fn,accuracy,precision,recall,specificity,f1,elapsed_ms\n";
    }

    void write_row(const std::string &phase,
                   std::size_t epoch,
                   float train_loss,
                   float eval_cost,
                   const ClassificationMetrics &metrics,
                   std::uint64_t elapsed_ms)
    {
        out << phase << ','
            << epoch << ','
            << train_loss << ','
            << eval_cost << ','
            << metrics.tp << ','
            << metrics.fp << ','
            << metrics.tn << ','
            << metrics.fn << ','
            << metrics.accuracy << ','
            << metrics.precision << ','
            << metrics.recall << ','
            << metrics.specificity << ','
            << metrics.f1 << ','
            << elapsed_ms << '\n';
        if (!out.good()) {
            throw std::runtime_error("Failed to write results CSV row");
        }
    }

    std::ofstream out;
};

}  // namespace

TrainingRunResult run_training_pipeline(const TrainingConfig &config,
                                        const NetworkBlueprint &blueprint) {
    const auto dataset = ManufacturingDefectDataset::load_csv(config.dataset_path);
    std::cout << "Loaded samples: " << dataset.size() << "\n";
    std::cout << "Feature count: " << dataset.feature_count() << "\n\n";

    std::vector<OperationConfig> operations;
    if (!config.load_model_path.empty()) {
        operations = MlpNetwork::load_operations_from_file(config.load_model_path);
        if (operations.empty() || operations.front().type != OperationType::Linear) {
            throw std::runtime_error("Loaded model file does not contain a valid network architecture");
        }
        if (operations.front().input_size != dataset.feature_count()) {
            throw std::runtime_error("Loaded model input size does not match dataset feature count");
        }
    } else {
        operations = blueprint.build(dataset.feature_count());
    }
    MlpNetwork network(std::move(operations));
    network.set_execution_backend(config.backend);
    network.set_optimizer_type(config.optimizer);
    network.set_optimizer_hyperparameters(
        config.momentum,
        config.adam_beta1,
        config.adam_beta2,
        config.adam_epsilon);

    if (!config.load_model_path.empty()) {
        network.load_from_file(config.load_model_path);
        std::cout << "Loaded model: " << config.load_model_path.string() << "\n";
    }

    float positive_weight = config.positive_class_weight;
    float negative_weight = config.negative_class_weight;
    if (config.auto_class_weights) {
        compute_auto_class_weights(dataset, positive_weight, negative_weight);
    }
    network.set_class_weights(positive_weight, negative_weight);
    std::cout << "Using BCE class weights: pos=" << positive_weight
              << ", neg=" << negative_weight << "\n";
    std::cout << "Loss function: " << loss_name(config.loss) << "\n";
    std::cout << "Optimizer: " << optimizer_name(config.optimizer) << "\n";
    if (config.optimizer == OptimizerType::Momentum) {
        std::cout << "Momentum: " << config.momentum << "\n";
    } else if (config.optimizer == OptimizerType::Adam) {
        std::cout << "Adam betas: beta1=" << config.adam_beta1
                  << ", beta2=" << config.adam_beta2
                  << ", epsilon=" << config.adam_epsilon << "\n";
    }
    std::cout << "Execution backend: " << backend_name(config.backend) << "\n";
    std::cout << "Hidden activation: " << activation_name(config.hidden_activation) << "\n";
    std::cout << "Output activation: " << activation_name(config.output_activation) << "\n";
    std::cout << "Learning rate: " << config.learning_rate << "\n";
    std::cout << "LR decay: " << config.lr_decay
              << " every " << config.lr_decay_every
              << " epoch(s), min_lr=" << config.min_learning_rate << "\n";
    std::cout << "Auto class weights: " << (config.auto_class_weights ? "true" : "false") << "\n";
    std::cout << "Threshold: " << config.threshold << "\n";
    std::cout << "Batch size: " << config.batch_size << "\n";
    std::cout << "Epochs: " << config.epochs << "\n";
    std::cout << "Print every: " << config.print_every << "\n";
    std::cout << "Eval only: " << (config.eval_only ? "true" : "false") << "\n";
    std::cout << "Dataset path: " << config.dataset_path.string() << "\n\n";

    TrainingRunResult result;
    result.initial_cost = network.evaluate_cost(dataset, config.loss);
    std::cout << "Initial cost: " << result.initial_cost << "\n";
    const ClassificationMetrics initial_metrics =
        evaluate_classification_metrics(network, dataset, config.threshold);
    print_classification_metrics(initial_metrics);

    std::unique_ptr<ResultsCsvWriter> csv_writer;
    if (!config.results_csv_path.empty()) {
        csv_writer = std::make_unique<ResultsCsvWriter>(config.results_csv_path);
        csv_writer->write_row("initial", 0, 0.0f, result.initial_cost, initial_metrics, 0);
        std::cout << "Results CSV: " << config.results_csv_path.string() << "\n";
    }

    result.final_eval_cost = result.initial_cost;
    result.final_metrics = initial_metrics;
    float accumulated_train_loss = 0.0f;
    const auto training_start = std::chrono::high_resolution_clock::now();

    if (!config.eval_only) {
        for (std::size_t epoch = 1; epoch <= config.epochs; ++epoch) {
            const auto epoch_start = std::chrono::high_resolution_clock::now();
            const std::size_t decay_steps = (epoch - 1) / config.lr_decay_every;
            float current_lr = config.learning_rate;
            if (config.lr_decay < 1.0f && decay_steps > 0) {
                current_lr *= std::pow(config.lr_decay, static_cast<float>(decay_steps));
            }
            if (current_lr < config.min_learning_rate) {
                current_lr = config.min_learning_rate;
            }

            const float epoch_train_loss = network.train_one_epoch(
                dataset,
                current_lr,
                config.loss,
                config.batch_size);
            const auto epoch_end = std::chrono::high_resolution_clock::now();
            const auto epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start);
            accumulated_train_loss += epoch_train_loss;

            const bool should_log = (epoch % config.print_every == 0) || (epoch == config.epochs);
            if (should_log) {
                const auto eval_start = std::chrono::high_resolution_clock::now();
                result.final_eval_cost = network.evaluate_cost(dataset, config.loss);
                result.final_metrics = evaluate_classification_metrics(network, dataset, config.threshold);
                const auto eval_end = std::chrono::high_resolution_clock::now();
                const auto eval_ms = std::chrono::duration_cast<std::chrono::milliseconds>(eval_end - eval_start);

                std::cout << "[Epoch " << epoch << "/" << config.epochs << "] "
                          << "train_loss=" << epoch_train_loss
                          << ", lr=" << current_lr
                          << ", eval_cost=" << result.final_eval_cost
                          << ", train_time=" << epoch_ms.count() << " ms"
                          << ", eval_time=" << eval_ms.count() << " ms\n";
                print_classification_metrics(result.final_metrics);
                if (csv_writer != nullptr) {
                    csv_writer->write_row("epoch", epoch, epoch_train_loss, result.final_eval_cost, result.final_metrics, eval_ms.count());
                }
            }
        }
    } else {
        const auto eval_start = std::chrono::high_resolution_clock::now();
        result.final_eval_cost = network.evaluate_cost(dataset, config.loss);
        result.final_metrics = evaluate_classification_metrics(network, dataset, config.threshold);
        const auto eval_end = std::chrono::high_resolution_clock::now();
        const auto eval_ms = std::chrono::duration_cast<std::chrono::milliseconds>(eval_end - eval_start);
        std::cout << "Eval-only cost: " << result.final_eval_cost
                  << ", eval_time=" << eval_ms.count() << " ms\n";
        print_classification_metrics(result.final_metrics);
        if (csv_writer != nullptr) {
            csv_writer->write_row("eval_only", 0, 0.0f, result.final_eval_cost, result.final_metrics, eval_ms.count());
        }
    }

    const auto training_end = std::chrono::high_resolution_clock::now();
    result.total_training_ms = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(training_end - training_start).count());

    result.average_epoch_train_loss = config.eval_only
        ? 0.0f
        : accumulated_train_loss / static_cast<float>(config.epochs);

    if (!config.save_model_path.empty()) {
        network.save_to_file(config.save_model_path);
        std::cout << "Saved model: " << config.save_model_path.string() << "\n";
    }

    std::cout << "\nTraining summary:\n";
    if (!config.eval_only) {
        std::cout << "Average epoch train loss: " << result.average_epoch_train_loss << "\n";
    }
    std::cout << "Final eval cost: " << result.final_eval_cost << "\n";
    std::cout << "Total training time: " << result.total_training_ms << " ms\n";

    if (!config.pr_csv_path.empty()) {
        const auto pr_rows = scan_pr_curve(
            network,
            dataset,
            config.pr_scan_min,
            config.pr_scan_max,
            config.pr_scan_step);

        PrCurveCsvWriter pr_writer(config.pr_csv_path);
        for (const auto &row : pr_rows) {
            pr_writer.write_row(row.threshold, row.metrics);
        }
        std::cout << "PR CSV: " << config.pr_csv_path.string() << " ("
                  << pr_rows.size() << " thresholds)\n";
    }

    return result;
}
