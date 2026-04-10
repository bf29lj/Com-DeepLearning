#include "config_io.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace {

constexpr int kConfigSchemaVersion = 1;

std::string parse_error_with_context(const std::filesystem::path &path,
                                     const std::string &detail);

using SectionMap = std::unordered_map<std::string, std::unordered_map<std::string, std::string>>;

std::string trim(const std::string &value) {
    std::size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin])) != 0) {
        ++begin;
    }
    std::size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
        --end;
    }
    return value.substr(begin, end - begin);
}

std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string parse_error_with_context(const std::filesystem::path &path,
                                     const std::string &detail)
{
    return "Config parse error [" + path.string() + "]: " + detail;
}

bool parse_bool(const std::string &value) {
    const std::string lowered = to_lower(trim(value));
    if (lowered == "true" || lowered == "1" || lowered == "yes") {
        return true;
    }
    if (lowered == "false" || lowered == "0" || lowered == "no") {
        return false;
    }
    throw std::invalid_argument("Invalid boolean value: " + value);
}

std::string get_or_empty(const SectionMap &sections,
                         const std::string &section,
                         const std::string &key)
{
    const auto sec_it = sections.find(section);
    if (sec_it == sections.end()) {
        return "";
    }
    const auto key_it = sec_it->second.find(key);
    if (key_it == sec_it->second.end()) {
        return "";
    }
    return key_it->second;
}

SectionMap parse_ini_file(const std::filesystem::path &path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error(parse_error_with_context(path, "Failed to open file"));
    }

    SectionMap sections;
    std::string current_section;
    std::string line;
    std::size_t line_no = 0;
    while (std::getline(input, line)) {
        ++line_no;
        std::string content = trim(line);
        if (content.empty() || content[0] == '#' || content[0] == ';') {
            continue;
        }
        if (content.front() == '[' && content.back() == ']') {
            current_section = to_lower(trim(content.substr(1, content.size() - 2)));
            continue;
        }
        const std::size_t eq_pos = content.find('=');
        if (eq_pos == std::string::npos) {
            throw std::runtime_error(parse_error_with_context(
                path,
                "Invalid line " + std::to_string(line_no) + ": expected key=value"));
        }
        const std::string key = to_lower(trim(content.substr(0, eq_pos)));
        const std::string value = trim(content.substr(eq_pos + 1));
        if (current_section.empty()) {
            throw std::runtime_error(parse_error_with_context(
                path,
                "Invalid line " + std::to_string(line_no) + ": key/value outside section"));
        }
        sections[current_section][key] = value;
    }

    return sections;
}

void validate_schema(const std::filesystem::path &path,
                     const SectionMap &sections,
                     const std::string &expected_kind)
{
    const std::string kind = to_lower(get_or_empty(sections, "meta", "kind"));
    if (!kind.empty() && kind != expected_kind) {
        throw std::runtime_error(parse_error_with_context(
            path,
            "Config kind mismatch: expected " + expected_kind + ", got " + kind));
    }

    const std::string schema_text = get_or_empty(sections, "meta", "schema_version");
    const int schema_version = schema_text.empty() ? 1 : std::stoi(schema_text);
    if (schema_version <= 0) {
        throw std::runtime_error(parse_error_with_context(path, "schema_version must be positive"));
    }
    if (schema_version > kConfigSchemaVersion) {
        throw std::runtime_error(parse_error_with_context(
            path,
            "Unsupported schema_version=" + std::to_string(schema_version) +
                ", current supported max is " + std::to_string(kConfigSchemaVersion)));
    }

    const std::string model = to_lower(get_or_empty(sections, "meta", "model"));
    if (!model.empty() && model != "mlp" && model != "lstm") {
        throw std::runtime_error(parse_error_with_context(path, "Unsupported model type: " + model));
    }
}

void save_config_header(std::ofstream &out,
                        const std::string &kind,
                        const std::string &model)
{
    out << "[meta]\n";
    out << "kind=" << kind << "\n";
    out << "schema_version=" << kConfigSchemaVersion << "\n";
    out << "model=" << model << "\n\n";
}

}  // namespace

void load_training_config_file(const std::filesystem::path &path, TrainingConfig &config) {
    const SectionMap sections = parse_ini_file(path);
    validate_schema(path, sections, "training_config");

    const std::string model = to_lower(get_or_empty(sections, "meta", "model"));
    if (!model.empty()) {
        config.model_type = model;
    }

    const std::string dataset = get_or_empty(sections, "paths", "dataset_path");
    if (!dataset.empty()) {
        config.dataset_path = dataset;
    }
    const std::string load_model = get_or_empty(sections, "paths", "load_model_path");
    if (!load_model.empty()) {
        config.load_model_path = load_model;
    }
    const std::string save_model = get_or_empty(sections, "paths", "save_model_path");
    if (!save_model.empty()) {
        config.save_model_path = save_model;
    }
    const std::string results_csv = get_or_empty(sections, "paths", "results_csv_path");
    if (!results_csv.empty()) {
        config.results_csv_path = results_csv;
    }
    const std::string pr_csv = get_or_empty(sections, "paths", "pr_csv_path");
    if (!pr_csv.empty()) {
        config.pr_csv_path = pr_csv;
    }

    const std::string backend = to_lower(get_or_empty(sections, "runtime", "backend"));
    if (!backend.empty()) {
        if (backend == "cpu") {
            config.backend = ExecutionBackend::CPU;
        } else if (backend == "gpu") {
            config.backend = ExecutionBackend::GPU;
        } else {
            throw std::runtime_error(parse_error_with_context(path, "Invalid backend: " + backend));
        }
    }

    const std::string loss = to_lower(get_or_empty(sections, "runtime", "loss"));
    if (!loss.empty()) {
        if (loss == "bce") {
            config.loss = LossType::BCE;
        } else if (loss == "mse") {
            config.loss = LossType::MSE;
        } else if (loss == "focal") {
            config.loss = LossType::Focal;
        } else {
            throw std::runtime_error(parse_error_with_context(path, "Invalid loss: " + loss));
        }
    }

    const std::string optimizer = to_lower(get_or_empty(sections, "runtime", "optimizer"));
    if (!optimizer.empty()) {
        if (optimizer == "sgd") {
            config.optimizer = OptimizerType::SGD;
        } else if (optimizer == "momentum") {
            config.optimizer = OptimizerType::Momentum;
        } else if (optimizer == "adam") {
            config.optimizer = OptimizerType::Adam;
        } else if (optimizer == "adamw") {
            config.optimizer = OptimizerType::AdamW;
        } else {
            throw std::runtime_error(parse_error_with_context(path, "Invalid optimizer: " + optimizer));
        }
    }

    const std::string lr = get_or_empty(sections, "runtime", "learning_rate");
    if (!lr.empty()) {
        config.learning_rate = std::stof(lr);
    }
    const std::string focal_gamma = get_or_empty(sections, "runtime", "focal_gamma");
    if (!focal_gamma.empty()) {
        config.focal_gamma = std::stof(focal_gamma);
    }
    const std::string focal_alpha = get_or_empty(sections, "runtime", "focal_alpha");
    if (!focal_alpha.empty()) {
        config.focal_alpha = std::stof(focal_alpha);
    }
    const std::string lr_decay = get_or_empty(sections, "runtime", "lr_decay");
    if (!lr_decay.empty()) {
        config.lr_decay = std::stof(lr_decay);
    }
    const std::string lr_decay_every = get_or_empty(sections, "runtime", "lr_decay_every");
    if (!lr_decay_every.empty()) {
        config.lr_decay_every = static_cast<std::size_t>(std::stoul(lr_decay_every));
    }
    const std::string min_learning_rate = get_or_empty(sections, "runtime", "min_learning_rate");
    if (!min_learning_rate.empty()) {
        config.min_learning_rate = std::stof(min_learning_rate);
    }
        const std::string timeout_sec = get_or_empty(sections, "runtime", "timeout_sec");
        if (!timeout_sec.empty()) {
            config.timeout_sec = std::stof(timeout_sec);
        }
    const std::string momentum = get_or_empty(sections, "runtime", "momentum");
    if (!momentum.empty()) {
        config.momentum = std::stof(momentum);
    }
    const std::string adam_beta1 = get_or_empty(sections, "runtime", "adam_beta1");
    if (!adam_beta1.empty()) {
        config.adam_beta1 = std::stof(adam_beta1);
    }
    const std::string adam_beta2 = get_or_empty(sections, "runtime", "adam_beta2");
    if (!adam_beta2.empty()) {
        config.adam_beta2 = std::stof(adam_beta2);
    }
    const std::string adam_epsilon = get_or_empty(sections, "runtime", "adam_epsilon");
    if (!adam_epsilon.empty()) {
        config.adam_epsilon = std::stof(adam_epsilon);
    }
    const std::string weight_decay = get_or_empty(sections, "runtime", "weight_decay");
    if (!weight_decay.empty()) {
        config.weight_decay = std::stof(weight_decay);
    }
    const std::string hidden_activation = to_lower(get_or_empty(sections, "runtime", "hidden_activation"));
    if (!hidden_activation.empty()) {
        if (hidden_activation == "relu") {
            config.hidden_activation = ActivationType::Relu;
        } else if (hidden_activation == "sigmoid") {
            config.hidden_activation = ActivationType::Sigmoid;
        } else if (hidden_activation == "tanh") {
            config.hidden_activation = ActivationType::Tanh;
        } else if (hidden_activation == "leaky_relu") {
            config.hidden_activation = ActivationType::LeakyRelu;
        } else if (hidden_activation == "gelu") {
            config.hidden_activation = ActivationType::Gelu;
        } else {
            throw std::runtime_error(parse_error_with_context(path, "Invalid hidden_activation: " + hidden_activation));
        }
    }
    const std::string output_activation = to_lower(get_or_empty(sections, "runtime", "output_activation"));
    if (!output_activation.empty()) {
        if (output_activation == "linear") {
            config.output_activation = ActivationType::Linear;
        } else if (output_activation == "relu") {
            config.output_activation = ActivationType::Relu;
        } else if (output_activation == "sigmoid") {
            config.output_activation = ActivationType::Sigmoid;
        } else if (output_activation == "tanh") {
            config.output_activation = ActivationType::Tanh;
        } else if (output_activation == "leaky_relu") {
            config.output_activation = ActivationType::LeakyRelu;
        } else if (output_activation == "gelu") {
            config.output_activation = ActivationType::Gelu;
        } else {
            throw std::runtime_error(parse_error_with_context(path, "Invalid output_activation: " + output_activation));
        }
    }
    const std::string batch_size = get_or_empty(sections, "runtime", "batch_size");
    if (!batch_size.empty()) {
        config.batch_size = static_cast<std::size_t>(std::stoul(batch_size));
    }
    const std::string epochs = get_or_empty(sections, "runtime", "epochs");
    if (!epochs.empty()) {
        config.epochs = static_cast<std::size_t>(std::stoul(epochs));
    }
    const std::string print_every = get_or_empty(sections, "runtime", "print_every");
    if (!print_every.empty()) {
        config.print_every = static_cast<std::size_t>(std::stoul(print_every));
    }
    const std::string eval_only = get_or_empty(sections, "runtime", "eval_only");
    if (!eval_only.empty()) {
        config.eval_only = parse_bool(eval_only);
    }
    const std::string lstm_seq_len = get_or_empty(sections, "runtime", "lstm_seq_len");
    if (!lstm_seq_len.empty()) {
        config.lstm_seq_len = static_cast<std::size_t>(std::stoul(lstm_seq_len));
    }
    const std::string lstm_hidden_size = get_or_empty(sections, "runtime", "lstm_hidden_size");
    if (!lstm_hidden_size.empty()) {
        config.lstm_hidden_size = static_cast<std::size_t>(std::stoul(lstm_hidden_size));
    }

    const std::string pos_w = get_or_empty(sections, "metrics", "positive_class_weight");
    if (!pos_w.empty()) {
        config.positive_class_weight = std::stof(pos_w);
    }
    const std::string neg_w = get_or_empty(sections, "metrics", "negative_class_weight");
    if (!neg_w.empty()) {
        config.negative_class_weight = std::stof(neg_w);
    }
    const std::string auto_w = get_or_empty(sections, "metrics", "auto_class_weights");
    if (!auto_w.empty()) {
        config.auto_class_weights = parse_bool(auto_w);
    }
    const std::string threshold = get_or_empty(sections, "metrics", "threshold");
    if (!threshold.empty()) {
        config.threshold = std::stof(threshold);
    }
    const std::string pr_min = get_or_empty(sections, "metrics", "pr_scan_min");
    if (!pr_min.empty()) {
        config.pr_scan_min = std::stof(pr_min);
    }
    const std::string pr_max = get_or_empty(sections, "metrics", "pr_scan_max");
    if (!pr_max.empty()) {
        config.pr_scan_max = std::stof(pr_max);
    }
    const std::string pr_step = get_or_empty(sections, "metrics", "pr_scan_step");
    if (!pr_step.empty()) {
        config.pr_scan_step = std::stof(pr_step);
    }

}

void save_training_config_file(const std::filesystem::path &path, const TrainingConfig &config) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open config file for write: " + path.string());
    }

    save_config_header(out, "training_config", config.model_type);

    out << "[paths]\n";
    out << "dataset_path=" << config.dataset_path.string() << "\n";
    out << "load_model_path=" << config.load_model_path.string() << "\n";
    out << "save_model_path=" << config.save_model_path.string() << "\n";
    out << "results_csv_path=" << config.results_csv_path.string() << "\n";
    out << "pr_csv_path=" << config.pr_csv_path.string() << "\n\n";

    out << "[runtime]\n";
    out << "backend=" << (config.backend == ExecutionBackend::GPU ? "gpu" : "cpu") << "\n";
    const char *loss = "bce";
    if (config.loss == LossType::MSE) {
        loss = "mse";
    } else if (config.loss == LossType::Focal) {
        loss = "focal";
    }
    out << "loss=" << loss << "\n";
    const char *optimizer = "sgd";
    if (config.optimizer == OptimizerType::Momentum) {
        optimizer = "momentum";
    } else if (config.optimizer == OptimizerType::Adam) {
        optimizer = "adam";
    } else if (config.optimizer == OptimizerType::AdamW) {
        optimizer = "adamw";
    }
    out << "optimizer=" << optimizer << "\n";
    out << "learning_rate=" << config.learning_rate << "\n";
    out << "focal_gamma=" << config.focal_gamma << "\n";
    out << "focal_alpha=" << config.focal_alpha << "\n";
    out << "momentum=" << config.momentum << "\n";
    out << "adam_beta1=" << config.adam_beta1 << "\n";
    out << "adam_beta2=" << config.adam_beta2 << "\n";
    out << "adam_epsilon=" << config.adam_epsilon << "\n";
    out << "weight_decay=" << config.weight_decay << "\n";
    out << "lr_decay=" << config.lr_decay << "\n";
    out << "lr_decay_every=" << config.lr_decay_every << "\n";
    out << "min_learning_rate=" << config.min_learning_rate << "\n";
        out << "timeout_sec=" << config.timeout_sec << "\n";
    const char *hidden_activation = "relu";
    if (config.hidden_activation == ActivationType::Sigmoid) {
        hidden_activation = "sigmoid";
    } else if (config.hidden_activation == ActivationType::Tanh) {
        hidden_activation = "tanh";
    } else if (config.hidden_activation == ActivationType::LeakyRelu) {
        hidden_activation = "leaky_relu";
    } else if (config.hidden_activation == ActivationType::Gelu) {
        hidden_activation = "gelu";
    }
    const char *output_activation = "sigmoid";
    if (config.output_activation == ActivationType::Linear) {
        output_activation = "linear";
    } else if (config.output_activation == ActivationType::Relu) {
        output_activation = "relu";
    } else if (config.output_activation == ActivationType::Tanh) {
        output_activation = "tanh";
    } else if (config.output_activation == ActivationType::LeakyRelu) {
        output_activation = "leaky_relu";
    } else if (config.output_activation == ActivationType::Gelu) {
        output_activation = "gelu";
    }
    out << "hidden_activation=" << hidden_activation << "\n";
    out << "output_activation=" << output_activation << "\n";
    out << "batch_size=" << config.batch_size << "\n";
    out << "epochs=" << config.epochs << "\n";
    out << "print_every=" << config.print_every << "\n";
    out << "eval_only=" << (config.eval_only ? "true" : "false") << "\n";
    out << "lstm_seq_len=" << config.lstm_seq_len << "\n";
    out << "lstm_hidden_size=" << config.lstm_hidden_size << "\n\n";

    out << "[metrics]\n";
    out << "positive_class_weight=" << config.positive_class_weight << "\n";
    out << "negative_class_weight=" << config.negative_class_weight << "\n";
    out << "auto_class_weights=" << (config.auto_class_weights ? "true" : "false") << "\n";
    out << "threshold=" << config.threshold << "\n";
    out << "pr_scan_min=" << config.pr_scan_min << "\n";
    out << "pr_scan_max=" << config.pr_scan_max << "\n";
    out << "pr_scan_step=" << config.pr_scan_step << "\n\n";
}
