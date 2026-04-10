#include "training_pipeline.h"
#include "config_io.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void print_usage() {
    std::cout << "Usage: first_gpu_program [options]\n"
              << "Options:\n"
              << "  --config <path>      Import training config INI file\n"
              << "  --export-config <path> Export merged training config to INI\n"
              << "  --export-only        Export config(s) and exit without training\n"
              << "  --dataset <path>     Dataset CSV path (default: auto-resolve train.csv)\n"
              << "  --model <mlp|lstm>   Model type (default: mlp)\n"
              << "  --load-model <path>  Load model weights from file before running\n"
              << "  --save-model <path>  Save model weights to file after running\n"
              << "  --backend <cpu|gpu>  Execution backend (default: cpu)\n"
              << "  --optimizer <sgd|momentum|adam|adamw> Optimizer type (default: sgd)\n"
              << "  --momentum <float>   Momentum factor for momentum SGD (default: 0.9)\n"
              << "  --adam-beta1 <float> Adam beta1 (default: 0.9)\n"
              << "  --adam-beta2 <float> Adam beta2 (default: 0.999)\n"
              << "  --adam-eps <float>   Adam epsilon (default: 1e-8)\n"
              << "  --weight-decay <float> AdamW decoupled weight decay (default: 0.0)\n"
              << "  --hidden-act <relu|sigmoid|tanh|leaky_relu|gelu> Hidden layer activation (default: relu)\n"
              << "  --output-act <linear|sigmoid|tanh|relu|leaky_relu|gelu> Output activation (default: sigmoid)\n"
              << "  --results-csv <path> Write per-epoch metrics to CSV\n"
              << "  --pr-csv <path>      Write threshold scan for PR curve to CSV\n"
              << "  --loss <bce|mse|focal> Loss function (default: bce)\n"
              << "  --focal-gamma <float> Focal loss gamma (default: 2.0)\n"
              << "  --focal-alpha <float> Focal loss alpha in [0,1] (default: 0.25)\n"
              << "  --lr <float>         Learning rate (default: 0.01)\n"
              << "  --lr-decay <float>   Multiplicative LR decay factor (default: 1.0)\n"
              << "  --lr-decay-every <int> Apply LR decay every N epochs (default: 1)\n"
              << "  --min-lr <float>     Lower bound for decayed LR (default: 0.0)\n"
              << "  --timeout-sec <float> Stop training after this many seconds (0=disabled)\n"
              << "  --pos-weight <float> Positive class weight for BCE (default: 1.0)\n"
              << "  --neg-weight <float> Negative class weight for BCE (default: 1.0)\n"
              << "  --auto-class-weights Auto-compute BCE class weights from dataset\n"
              << "  --threshold <float>  Classification threshold [0,1] (default: 0.5)\n"
              << "  --batch-size <int>   Batch size for training updates (default: 1)\n"
              << "  --seq-len <int>      LSTM sequence length (default: 8)\n"
              << "  --lstm-hidden <int>  LSTM hidden size (default: 16)\n"
              << "  --epochs <int>       Number of training epochs (default: 20)\n"
              << "  --print-every <int>  Epoch log interval (default: 1)\n"
              << "  --eval-only          Skip training and only run evaluation\n"
              << "  --help               Show this help\n";
}

std::string parse_model_type(const std::string &value) {
    if (value == "mlp" || value == "lstm") {
        return value;
    }
    throw std::invalid_argument("Invalid model: " + value + ". Use mlp or lstm.");
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
    if (value == "focal") {
        return LossType::Focal;
    }
    throw std::invalid_argument("Invalid loss: " + value + ". Use bce, mse or focal.");
}

OptimizerType parse_optimizer(const std::string &value) {
    if (value == "sgd") {
        return OptimizerType::SGD;
    }
    if (value == "momentum") {
        return OptimizerType::Momentum;
    }
    if (value == "adam") {
        return OptimizerType::Adam;
    }
    if (value == "adamw") {
        return OptimizerType::AdamW;
    }
    throw std::invalid_argument("Invalid optimizer: " + value + ". Use sgd, momentum, adam or adamw.");
}

ActivationType parse_activation(const std::string &value) {
    if (value == "linear") {
        return ActivationType::Linear;
    }
    if (value == "relu") {
        return ActivationType::Relu;
    }
    if (value == "sigmoid") {
        return ActivationType::Sigmoid;
    }
    if (value == "tanh") {
        return ActivationType::Tanh;
    }
    if (value == "leaky_relu") {
        return ActivationType::LeakyRelu;
    }
    if (value == "gelu") {
        return ActivationType::Gelu;
    }
    throw std::invalid_argument("Invalid activation: " + value + ". Use linear/relu/sigmoid/tanh/leaky_relu/gelu.");
}

TrainingConfig parse_args(int argc, char **argv) {
    TrainingConfig cfg;

    auto first_pass_need_value = [&](int &index, const std::string &name) -> std::string {
        if (index + 1 >= argc) {
            throw std::invalid_argument("Missing value for " + name);
        }
        return argv[++index];
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            print_usage();
            std::exit(0);
        }
        if (arg == "--config") {
            cfg.import_config_path = first_pass_need_value(i, arg);
        }
    }

    if (!cfg.import_config_path.empty()) {
        load_training_config_file(cfg.import_config_path, cfg);
    }

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
        } else if (arg == "--config") {
            (void)need_value(arg);
        } else if (arg == "--export-config") {
            cfg.export_config_path = need_value(arg);
        } else if (arg == "--export-only") {
            cfg.export_only = true;
        } else if (arg == "--dataset") {
            cfg.dataset_path = need_value(arg);
        } else if (arg == "--model") {
            cfg.model_type = parse_model_type(need_value(arg));
        } else if (arg == "--load-model") {
            cfg.load_model_path = need_value(arg);
        } else if (arg == "--save-model") {
            cfg.save_model_path = need_value(arg);
        } else if (arg == "--results-csv") {
            cfg.results_csv_path = need_value(arg);
        } else if (arg == "--backend") {
            cfg.backend = parse_backend(need_value(arg));
        } else if (arg == "--optimizer") {
            cfg.optimizer = parse_optimizer(need_value(arg));
        } else if (arg == "--momentum") {
            cfg.momentum = std::stof(need_value(arg));
        } else if (arg == "--adam-beta1") {
            cfg.adam_beta1 = std::stof(need_value(arg));
        } else if (arg == "--adam-beta2") {
            cfg.adam_beta2 = std::stof(need_value(arg));
        } else if (arg == "--adam-eps") {
            cfg.adam_epsilon = std::stof(need_value(arg));
        } else if (arg == "--weight-decay") {
            cfg.weight_decay = std::stof(need_value(arg));
        } else if (arg == "--hidden-act") {
            cfg.hidden_activation = parse_activation(need_value(arg));
            if (cfg.hidden_activation == ActivationType::Linear) {
                throw std::invalid_argument("--hidden-act cannot be linear");
            }
        } else if (arg == "--output-act") {
            cfg.output_activation = parse_activation(need_value(arg));
        } else if (arg == "--loss") {
            cfg.loss = parse_loss(need_value(arg));
        } else if (arg == "--focal-gamma") {
            cfg.focal_gamma = std::stof(need_value(arg));
        } else if (arg == "--focal-alpha") {
            cfg.focal_alpha = std::stof(need_value(arg));
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
        } else if (arg == "--timeout-sec") {
            cfg.timeout_sec = std::stof(need_value(arg));
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
        } else if (arg == "--seq-len") {
            const int seq_len = std::stoi(need_value(arg));
            if (seq_len <= 0) {
                throw std::invalid_argument("--seq-len must be positive");
            }
            cfg.lstm_seq_len = static_cast<std::size_t>(seq_len);
        } else if (arg == "--lstm-hidden") {
            const int hidden_size = std::stoi(need_value(arg));
            if (hidden_size <= 0) {
                throw std::invalid_argument("--lstm-hidden must be positive");
            }
            cfg.lstm_hidden_size = static_cast<std::size_t>(hidden_size);
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

    if (cfg.export_only && cfg.export_config_path.empty()) {
        throw std::invalid_argument("--export-only requires --export-config");
    }

    if (cfg.dataset_path.empty() && !cfg.export_only) {
        throw std::invalid_argument("Dataset path is not specified");
    }

    if (cfg.hidden_activation == ActivationType::Linear) {
        throw std::invalid_argument("Hidden activation cannot be linear");
    }

    if (cfg.learning_rate <= 0.0f || !std::isfinite(cfg.learning_rate)) {
        throw std::invalid_argument("--lr must be a positive finite number");
    }
    if (cfg.timeout_sec < 0.0f || !std::isfinite(cfg.timeout_sec)) {
        throw std::invalid_argument("--timeout-sec must be a non-negative finite number");
    }
    if (cfg.momentum < 0.0f || cfg.momentum >= 1.0f || !std::isfinite(cfg.momentum)) {
        throw std::invalid_argument("--momentum must be in [0, 1)");
    }
    if (cfg.adam_beta1 <= 0.0f || cfg.adam_beta1 >= 1.0f || !std::isfinite(cfg.adam_beta1)) {
        throw std::invalid_argument("--adam-beta1 must be in (0, 1)");
    }
    if (cfg.adam_beta2 <= 0.0f || cfg.adam_beta2 >= 1.0f || !std::isfinite(cfg.adam_beta2)) {
        throw std::invalid_argument("--adam-beta2 must be in (0, 1)");
    }
    if (cfg.adam_epsilon <= 0.0f || !std::isfinite(cfg.adam_epsilon)) {
        throw std::invalid_argument("--adam-eps must be positive");
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
    if (cfg.focal_gamma < 0.0f || !std::isfinite(cfg.focal_gamma)) {
        throw std::invalid_argument("--focal-gamma must be a non-negative finite number");
    }
    if (cfg.focal_alpha < 0.0f || cfg.focal_alpha > 1.0f || !std::isfinite(cfg.focal_alpha)) {
        throw std::invalid_argument("--focal-alpha must be within [0, 1]");
    }
    if (cfg.threshold < 0.0f || cfg.threshold > 1.0f || !std::isfinite(cfg.threshold)) {
        throw std::invalid_argument("--threshold must be within [0, 1]");
    }
    if (cfg.lstm_seq_len == 0) {
        throw std::invalid_argument("--seq-len must be positive");
    }
    if (cfg.lstm_hidden_size == 0) {
        throw std::invalid_argument("--lstm-hidden must be positive");
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

}  // namespace

int main(int argc, char **argv) {
    try {
        const TrainingConfig config = parse_args(argc, argv);
        const NetworkBlueprint blueprint = make_default_mlp_blueprint(
            config.hidden_activation,
            config.output_activation);

        if (!config.export_config_path.empty()) {
            save_training_config_file(config.export_config_path, config);
            std::cout << "Exported training config: " << config.export_config_path.string() << "\n";
        }
        if (config.export_only) {
            return 0;
        }

        run_training_pipeline(config, blueprint);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}