#include "defect_dataset.h"
#include "gpu_adapter.h"
#include "mlp_network.h"

#include <cmath>
#include <cstdint>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <fstream>

namespace {

struct TrainingConfig {
    std::filesystem::path dataset_path;
    std::filesystem::path load_model_path;
    std::filesystem::path save_model_path;
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

void print_usage() {
    std::cout << "Usage: first_gpu_program [options]\n"
              << "Options:\n"
              << "  --dataset <path>     Dataset CSV path (default: auto-resolve train.csv)\n"
              << "  --load-model <path>  Load model weights from file before running\n"
              << "  --save-model <path>  Save model weights to file after running\n"
              << "  --backend <cpu|gpu>  Execution backend (default: gpu)\n"
              << "  --results-csv <path> Write per-epoch metrics to CSV\n"
              << "  --pr-csv <path>      Write threshold scan for PR curve to CSV\n"
              << "  --loss <bce|mse>     Loss function (default: bce)\n"
              << "  --lr <float>         Learning rate (default: 0.01)\n"
              << "  --lr-decay <float>   Multiplicative LR decay factor (default: 1.0)\n"
              << "  --lr-decay-every <int> Apply LR decay every N epochs (default: 1)\n"
              << "  --min-lr <float>     Lower bound for decayed LR (default: 0.0)\n"
              << "  --pos-weight <float> Positive class weight for BCE (default: 1.0)\n"
              << "  --neg-weight <float> Negative class weight for BCE (default: 1.0)\n"
              << "  --auto-class-weights Auto-compute BCE class weights from dataset\n"
              << "  --threshold <float>  Classification threshold [0,1] (default: 0.5)\n"
              << "  --batch-size <int>   Batch size for training updates (default: 1)\n"
              << "  --epochs <int>       Number of training epochs (default: 20)\n"
              << "  --print-every <int>  Epoch log interval (default: 1)\n"
              << "  --eval-only          Skip training and only run evaluation\n"
              << "  --help               Show this help\n";
}

ExecutionBackend parse_backend(const std::string &value) {
    if (value == "cpu") {
        return ExecutionBackend::CPU;
    }
    if (value == "gpu") {
        return ExecutionBackend::GPU;
    }
    throw std::invalid_argument("Invalid backend: " + value + ". Use cpu or gpu.");
}

LossType parse_loss(const std::string &value) {
    if (value == "bce") {
        return LossType::BCE;
    }
    if (value == "mse") {
        return LossType::MSE;
    }
    throw std::invalid_argument("Invalid loss: " + value + ". Use bce or mse.");
}

TrainingConfig parse_args(int argc, char **argv) {
    TrainingConfig cfg;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const std::string &name) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing value for " + name);
            }
            return argv[++i];
        };

        if (arg == "--help") {
            print_usage();
            std::exit(0);
        } else if (arg == "--dataset") {
            cfg.dataset_path = need_value(arg);
        } else if (arg == "--load-model") {
            cfg.load_model_path = need_value(arg);
        } else if (arg == "--save-model") {
            cfg.save_model_path = need_value(arg);
        } else if (arg == "--results-csv") {
            cfg.results_csv_path = need_value(arg);
        } else if (arg == "--backend") {
            cfg.backend = parse_backend(need_value(arg));
        } else if (arg == "--loss") {
            cfg.loss = parse_loss(need_value(arg));
        } else if (arg == "--lr") {
            cfg.learning_rate = std::stof(need_value(arg));
        } else if (arg == "--lr-decay") {
            cfg.lr_decay = std::stof(need_value(arg));
        } else if (arg == "--lr-decay-every") {
            const int decay_every = std::stoi(need_value(arg));
            if (decay_every <= 0) {
                throw std::invalid_argument("--lr-decay-every must be positive");
            }
            cfg.lr_decay_every = static_cast<std::size_t>(decay_every);
        } else if (arg == "--min-lr") {
            cfg.min_learning_rate = std::stof(need_value(arg));
        } else if (arg == "--pr-csv") {
            cfg.pr_csv_path = need_value(arg);
        } else if (arg == "--pos-weight") {
            cfg.positive_class_weight = std::stof(need_value(arg));
        } else if (arg == "--neg-weight") {
            cfg.negative_class_weight = std::stof(need_value(arg));
        } else if (arg == "--auto-class-weights") {
            cfg.auto_class_weights = true;
        } else if (arg == "--threshold") {
            cfg.threshold = std::stof(need_value(arg));
        } else if (arg == "--pr-min") {
            cfg.pr_scan_min = std::stof(need_value(arg));
        } else if (arg == "--pr-max") {
            cfg.pr_scan_max = std::stof(need_value(arg));
        } else if (arg == "--pr-step") {
            cfg.pr_scan_step = std::stof(need_value(arg));
        } else if (arg == "--batch-size") {
            const int batch_size = std::stoi(need_value(arg));
            if (batch_size <= 0) {
                throw std::invalid_argument("--batch-size must be positive");
            }
            cfg.batch_size = static_cast<std::size_t>(batch_size);
        } else if (arg == "--epochs") {
            const int epochs = std::stoi(need_value(arg));
            if (epochs <= 0) {
                throw std::invalid_argument("--epochs must be positive");
            }
            cfg.epochs = static_cast<std::size_t>(epochs);
        } else if (arg == "--print-every") {
            const int print_every = std::stoi(need_value(arg));
            if (print_every <= 0) {
                throw std::invalid_argument("--print-every must be positive");
            }
            cfg.print_every = static_cast<std::size_t>(print_every);
        } else if (arg == "--eval-only") {
            cfg.eval_only = true;
        } else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }

    if (cfg.dataset_path.empty()) {
        throw std::invalid_argument("Dataset path is not specified");
    }

    if (cfg.learning_rate <= 0.0f || !std::isfinite(cfg.learning_rate)) {
        throw std::invalid_argument("--lr must be a positive finite number");
    }
    if (cfg.lr_decay <= 0.0f || cfg.lr_decay > 1.0f || !std::isfinite(cfg.lr_decay)) {
        throw std::invalid_argument("--lr-decay must be in (0, 1]");
    }
    if (cfg.min_learning_rate < 0.0f || !std::isfinite(cfg.min_learning_rate)) {
        throw std::invalid_argument("--min-lr must be a non-negative finite number");
    }
    if (cfg.min_learning_rate > cfg.learning_rate) {
        throw std::invalid_argument("--min-lr must be <= --lr");
    }
    if (cfg.positive_class_weight <= 0.0f || !std::isfinite(cfg.positive_class_weight)) {
        throw std::invalid_argument("--pos-weight must be a positive finite number");
    }
    if (cfg.negative_class_weight <= 0.0f || !std::isfinite(cfg.negative_class_weight)) {
        throw std::invalid_argument("--neg-weight must be a positive finite number");
    }
    if (cfg.threshold < 0.0f || cfg.threshold > 1.0f || !std::isfinite(cfg.threshold)) {
        throw std::invalid_argument("--threshold must be within [0, 1]");
    }
    if (cfg.pr_scan_min < 0.0f || cfg.pr_scan_min > 1.0f || !std::isfinite(cfg.pr_scan_min)) {
        throw std::invalid_argument("--pr-min must be within [0, 1]");
    }
    if (cfg.pr_scan_max < 0.0f || cfg.pr_scan_max > 1.0f || !std::isfinite(cfg.pr_scan_max)) {
        throw std::invalid_argument("--pr-max must be within [0, 1]");
    }
    if (cfg.pr_scan_step <= 0.0f || !std::isfinite(cfg.pr_scan_step)) {
        throw std::invalid_argument("--pr-step must be a positive finite number");
    }
    if (cfg.pr_scan_min > cfg.pr_scan_max) {
        throw std::invalid_argument("--pr-min must be <= --pr-max");
    }

    return cfg;
}

const char *backend_name(ExecutionBackend backend) {
    return backend == ExecutionBackend::GPU ? "GPU" : "CPU";
}

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

ClassificationMetrics evaluate_threshold_metrics(
    MlpNetwork &network,
    const ManufacturingDefectDataset &dataset,
    float threshold)
{
    return evaluate_classification_metrics(network, dataset, threshold);
}

std::vector<ThresholdMetricRow> scan_pr_curve(MlpNetwork &network,
                                          const ManufacturingDefectDataset &dataset,
                                          float min_threshold,
                                          float max_threshold,
                                          float step)
{
    std::vector<ThresholdMetricRow> rows;
    for (float threshold = min_threshold; threshold <= max_threshold + 1e-6f; threshold += step) {
        rows.push_back({threshold, evaluate_threshold_metrics(network, dataset, threshold)});
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

int main(int argc, char **argv) {
    try {
        const TrainingConfig config = parse_args(argc, argv);

        GpuContext ctx = GpuContext::create_default();
        std::cout << ctx.get_device_info();
        std::cout << "\n";

        const auto dataset = ManufacturingDefectDataset::load_csv(config.dataset_path);
        std::cout << "Loaded samples: " << dataset.size() << "\n";
        std::cout << "Feature count: " << dataset.feature_count() << "\n\n";

        std::vector<OperationConfig> operations = {
            OperationConfig::linear(dataset.feature_count(), 14),
            OperationConfig::relu(),
            OperationConfig::linear(14, 10),
            OperationConfig::relu(),
            OperationConfig::linear(10, 8),
            OperationConfig::relu(),
            OperationConfig::linear(8, 1),
            OperationConfig::sigmoid(),
        };
        MlpNetwork network(ctx, std::move(operations));
        network.set_execution_backend(config.backend);

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
        std::cout << "Execution backend: " << backend_name(config.backend) << "\n";
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

        const float before_cost = network.evaluate_cost(dataset, config.loss);
        std::cout << "Initial cost: " << before_cost << "\n";
        const ClassificationMetrics initial_metrics =
            evaluate_classification_metrics(network, dataset, config.threshold);
        print_classification_metrics(initial_metrics);

        std::unique_ptr<ResultsCsvWriter> csv_writer;
        if (!config.results_csv_path.empty()) {
            csv_writer = std::make_unique<ResultsCsvWriter>(config.results_csv_path);
            csv_writer->write_row("initial", 0, 0.0f, before_cost, initial_metrics, 0);
            std::cout << "Results CSV: " << config.results_csv_path.string() << "\n";
        }

        float last_eval_cost = before_cost;
        ClassificationMetrics last_metrics = initial_metrics;
        float accumulated_train_loss = 0.0f;
        auto training_start = std::chrono::high_resolution_clock::now();

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
                    last_eval_cost = network.evaluate_cost(dataset, config.loss);
                    last_metrics = evaluate_classification_metrics(network, dataset, config.threshold);
                    const auto eval_end = std::chrono::high_resolution_clock::now();
                    const auto eval_ms = std::chrono::duration_cast<std::chrono::milliseconds>(eval_end - eval_start);

                    std::cout << "[Epoch " << epoch << "/" << config.epochs << "] "
                              << "train_loss=" << epoch_train_loss
                              << ", lr=" << current_lr
                              << ", eval_cost=" << last_eval_cost
                              << ", train_time=" << epoch_ms.count() << " ms"
                              << ", eval_time=" << eval_ms.count() << " ms\n";
                    print_classification_metrics(last_metrics);
                    if (csv_writer != nullptr) {
                        csv_writer->write_row("epoch", epoch, epoch_train_loss, last_eval_cost, last_metrics, eval_ms.count());
                    }
                }
            }
        } else {
            const auto eval_start = std::chrono::high_resolution_clock::now();
            last_eval_cost = network.evaluate_cost(dataset, config.loss);
            last_metrics = evaluate_classification_metrics(network, dataset, config.threshold);
            const auto eval_end = std::chrono::high_resolution_clock::now();
            const auto eval_ms = std::chrono::duration_cast<std::chrono::milliseconds>(eval_end - eval_start);
            std::cout << "Eval-only cost: " << last_eval_cost
                      << ", eval_time=" << eval_ms.count() << " ms\n";
            print_classification_metrics(last_metrics);
            if (csv_writer != nullptr) {
                csv_writer->write_row("eval_only", 0, 0.0f, last_eval_cost, last_metrics, eval_ms.count());
            }
        }

        const auto training_end = std::chrono::high_resolution_clock::now();
        const auto total_training_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(training_end - training_start);

        const float average_train_loss = config.eval_only
            ? 0.0f
            : accumulated_train_loss / static_cast<float>(config.epochs);

        if (!config.save_model_path.empty()) {
            network.save_to_file(config.save_model_path);
            std::cout << "Saved model: " << config.save_model_path.string() << "\n";
        }

        std::cout << "\nTraining summary:\n";
        if (!config.eval_only) {
            std::cout << "Average epoch train loss: " << average_train_loss << "\n";
        }
        std::cout << "Final eval cost: " << last_eval_cost << "\n";
        std::cout << "Total training time: " << total_training_ms.count() << " ms\n";

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

        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}