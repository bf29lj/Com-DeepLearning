#include "mlp_network.h"
#include "gpu_kernels.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace {

constexpr uint32_t kModelFileVersion = 1;
constexpr uint32_t kModelMagic = 0x4D4C5031; // "MLP1"

template <typename T>
void write_binary(std::ofstream &out, const T &value) {
    out.write(reinterpret_cast<const char *>(&value), sizeof(T));
    if (!out.good()) {
        throw std::runtime_error("Failed to write model file");
    }
}

template <typename T>
void read_binary(std::ifstream &in, T &value) {
    in.read(reinterpret_cast<char *>(&value), sizeof(T));
    if (!in.good()) {
        throw std::runtime_error("Failed to read model file");
    }
}

} // namespace

struct MlpNetwork::GpuRuntimeImpl {
        struct LayerDeviceState {
            std::unique_ptr<GpuBuffer> weights;
            std::unique_ptr<GpuBuffer> biases;
            std::unique_ptr<GpuBuffer> grad_weights;
            std::unique_ptr<GpuBuffer> grad_biases;
            std::unique_ptr<GpuBuffer> velocity_weights;
            std::unique_ptr<GpuBuffer> velocity_biases;
            std::unique_ptr<GpuBuffer> adam_m_weights;
            std::unique_ptr<GpuBuffer> adam_v_weights;
            std::unique_ptr<GpuBuffer> adam_m_biases;
            std::unique_ptr<GpuBuffer> adam_v_biases;
        };

        GpuContext context;
        GpuProgram program;
        std::vector<LayerDeviceState> layer_states;
        bool parameters_synced = false;

        GpuRuntimeImpl()
                : context(GpuContext::create_default()),
                    program(gpu::create_gpu_program(context)) {}
};

MlpNetwork::DenseLayer::DenseLayer(std::size_t input,
                                                                     std::size_t output,
                                                                     ActivationType activation_type)
        : input_size(input),
            output_size(output),
            activation(activation_type),
            host_weights(input * output),
            host_biases(output) {}

float MlpNetwork::SgdOptimizer::output_gradient(
    float prediction,
    float target,
    LossType loss_type,
    bool use_bce_sigmoid_shortcut,
    float positive_weight,
    float negative_weight,
    float focal_gamma,
    float focal_alpha,
    float (*loss_derivative_wrt_prediction_fn)(float, float, LossType, float, float, float, float))
{
    if (use_bce_sigmoid_shortcut) {
        return positive_weight * target * (prediction - 1.0f) + negative_weight * (1.0f - target) * prediction;
    }
    return loss_derivative_wrt_prediction_fn(
        prediction,
        target,
        loss_type,
        positive_weight,
        negative_weight,
        focal_gamma,
        focal_alpha);
}

void MlpNetwork::set_optimizer_hyperparameters(float momentum,
                                               float adam_beta1,
                                               float adam_beta2,
                                               float adam_epsilon)
{
    if (!std::isfinite(momentum) || momentum < 0.0f || momentum >= 1.0f) {
        throw std::invalid_argument("Momentum must be in [0, 1)");
    }
    if (!std::isfinite(adam_beta1) || adam_beta1 <= 0.0f || adam_beta1 >= 1.0f) {
        throw std::invalid_argument("Adam beta1 must be in (0, 1)");
    }
    if (!std::isfinite(adam_beta2) || adam_beta2 <= 0.0f || adam_beta2 >= 1.0f) {
        throw std::invalid_argument("Adam beta2 must be in (0, 1)");
    }
    if (!std::isfinite(adam_epsilon) || adam_epsilon <= 0.0f) {
        throw std::invalid_argument("Adam epsilon must be positive");
    }

    momentum_ = momentum;
    adam_beta1_ = adam_beta1;
    adam_beta2_ = adam_beta2;
    adam_epsilon_ = adam_epsilon;
}

void MlpNetwork::set_weight_decay(float weight_decay) {
    if (!std::isfinite(weight_decay) || weight_decay < 0.0f) {
        throw std::invalid_argument("Weight decay must be a non-negative finite number");
    }
    weight_decay_ = weight_decay;
}

void MlpNetwork::SgdOptimizer::update_dense_layer(DenseLayer &layer,
                                                  const std::vector<float> &input,
                                                  const std::vector<float> &grad_output,
                                                  std::vector<float> &grad_input,
                                                  float learning_rate)
{
    for (std::size_t out = 0; out < layer.output_size; ++out) {
        const float delta = grad_output[out];
        layer.host_biases[out] -= learning_rate * delta;

        for (std::size_t in = 0; in < layer.input_size; ++in) {
            const std::size_t weight_index = out * layer.input_size + in;
            const float weight_before_update = layer.host_weights[weight_index];
            const float grad_w = delta * input[in];
            layer.host_weights[weight_index] -= learning_rate * grad_w;
            grad_input[in] += weight_before_update * delta;
        }
    }
}

MlpNetwork::MlpNetwork(std::vector<std::size_t> layer_sizes)
    : MlpNetwork(build_operations_from_layer_sizes(layer_sizes)) {}

MlpNetwork::MlpNetwork(std::vector<OperationConfig> operations) {
    if (operations.empty()) {
        throw std::invalid_argument("Operation pipeline cannot be empty");
    }

    operations_ = std::move(operations);
    for (std::size_t i = 0; i < operations_.size(); ++i) {
        const auto &op = operations_[i];
        if (op.type == OperationType::Linear) {
            if (op.input_size == 0 || op.output_size == 0) {
                throw std::invalid_argument("Linear operation requires non-zero input_size and output_size");
            }
            OperationType next_op_type = OperationType::Linear;
            if (i + 1 < operations_.size()) {
                next_op_type = operations_[i + 1].type;
            }

            layers_.emplace_back(
                op.input_size,
                op.output_size,
                operation_to_activation(next_op_type));
            initialize_layer(layers_.back(), next_op_type);

            OptimizerState state;
            state.momentum_w.assign(op.input_size * op.output_size, 0.0f);
            state.momentum_b.assign(op.output_size, 0.0f);
            state.adam_m_w.assign(op.input_size * op.output_size, 0.0f);
            state.adam_m_b.assign(op.output_size, 0.0f);
            state.adam_v_w.assign(op.input_size * op.output_size, 0.0f);
            state.adam_v_b.assign(op.output_size, 0.0f);
            optimizer_states_.push_back(std::move(state));
        }
    }
}

std::vector<OperationConfig> MlpNetwork::build_operations_from_layer_sizes(
    const std::vector<std::size_t> &layer_sizes)
{
    if (layer_sizes.size() < 2) {
        throw std::invalid_argument("Network needs at least input and output layer");
    }

    std::vector<OperationConfig> operations;
    for (std::size_t i = 0; i + 1 < layer_sizes.size(); ++i) {
        operations.push_back(OperationConfig::linear(layer_sizes[i], layer_sizes[i + 1]));
        const bool is_last_linear = (i + 2 == layer_sizes.size());
        operations.push_back(is_last_linear ? OperationConfig::sigmoid() : OperationConfig::relu());
    }
    return operations;
}

void MlpNetwork::initialize_layer(DenseLayer &layer, OperationType next_op_type) {
    std::mt19937 gen(42u + static_cast<unsigned int>(layer.input_size * 31 + layer.output_size));

    const float fan_in = static_cast<float>(layer.input_size);
    const float fan_out = static_cast<float>(layer.output_size);

    float stddev = 0.0f;
    if (next_op_type == OperationType::Relu) {
        // He initialization keeps variance stable through ReLU units.
        stddev = std::sqrt(2.0f / fan_in);
    } else {
        // Xavier/Glorot initialization is a good default for sigmoid/linear outputs.
        stddev = std::sqrt(2.0f / (fan_in + fan_out));
    }

    if (!std::isfinite(stddev) || stddev <= std::numeric_limits<float>::min()) {
        throw std::runtime_error("Invalid stddev in weight initialization");
    }

    std::normal_distribution<float> dist(0.0f, stddev);

    for (float &w : layer.host_weights) {
        w = dist(gen);
    }

    std::fill(layer.host_biases.begin(), layer.host_biases.end(), 0.0f);
}

void MlpNetwork::set_class_weights(float positive_weight, float negative_weight) {
    if (positive_weight <= 0.0f || negative_weight <= 0.0f || !std::isfinite(positive_weight) ||
        !std::isfinite(negative_weight)) {
        throw std::invalid_argument("Class weights must be positive finite numbers");
    }
    positive_class_weight_ = positive_weight;
    negative_class_weight_ = negative_weight;
}

void MlpNetwork::set_focal_parameters(float gamma, float alpha) {
    if (gamma < 0.0f || !std::isfinite(gamma)) {
        throw std::invalid_argument("Focal gamma must be non-negative and finite");
    }
    if (alpha < 0.0f || alpha > 1.0f || !std::isfinite(alpha)) {
        throw std::invalid_argument("Focal alpha must be within [0, 1]");
    }
    focal_gamma_ = gamma;
    focal_alpha_ = alpha;
}

MlpNetwork::ForwardCache MlpNetwork::forward_with_cache_cpu(const std::vector<float> &input) const {
    if (operations_.empty()) {
        throw std::runtime_error("Operation pipeline is empty");
    }

    ForwardCache cache;
    cache.op_inputs.reserve(operations_.size());
    cache.op_outputs.reserve(operations_.size());

    std::vector<float> current = input;
    std::size_t linear_index = 0;

    for (const auto &op : operations_) {
        cache.op_inputs.push_back(current);

        if (op.type == OperationType::Linear) {
            if (linear_index >= layers_.size()) {
                throw std::runtime_error("Internal linear layer index out of range");
            }

            const DenseLayer &layer = layers_[linear_index++];
            if (current.size() != layer.input_size) {
                throw std::invalid_argument("Input size mismatch for linear operation");
            }

            std::vector<float> output(layer.output_size, 0.0f);
            for (std::size_t out = 0; out < layer.output_size; ++out) {
                float value = layer.host_biases[out];
                for (std::size_t in = 0; in < layer.input_size; ++in) {
                    value += layer.host_weights[out * layer.input_size + in] * current[in];
                }
                output[out] = value;
            }
            current = std::move(output);
        } else if (op.type == OperationType::Relu) {
            for (float &v : current) {
                v = std::max(0.0f, v);
            }
        } else if (op.type == OperationType::Sigmoid) {
            for (float &v : current) {
                v = 1.0f / (1.0f + std::exp(-v));
            }
        } else if (op.type == OperationType::Tanh) {
            for (float &v : current) {
                v = std::tanh(v);
            }
        } else if (op.type == OperationType::Gelu) {
            for (float &v : current) {
                const float x = v;
                const float c = 0.044715f;
                const float s = 0.7978845608f;
                const float u = s * (x + c * x * x * x);
                v = 0.5f * x * (1.0f + std::tanh(u));
            }
        } else if (op.type == OperationType::LeakyRelu) {
            for (float &v : current) {
                v = (v > 0.0f) ? v : 0.01f * v;
            }
        }

        cache.op_outputs.push_back(current);
    }

    return cache;
}

ActivationType MlpNetwork::operation_to_activation(OperationType op_type) {
    switch (op_type) {
        case OperationType::Relu:
            return ActivationType::Relu;
        case OperationType::Sigmoid:
            return ActivationType::Sigmoid;
        case OperationType::Tanh:
            return ActivationType::Tanh;
        case OperationType::LeakyRelu:
            return ActivationType::LeakyRelu;
        case OperationType::Gelu:
            return ActivationType::Gelu;
        case OperationType::Linear:
            return ActivationType::Linear;
    }

    throw std::invalid_argument("Unsupported operation type for activation conversion");
}

std::vector<float> MlpNetwork::forward_cpu(const std::vector<float> &input) const {
    const ForwardCache cache = forward_with_cache_cpu(input);
    if (cache.op_outputs.empty()) {
        throw std::runtime_error("Forward cache is empty");
    }
    return cache.op_outputs.back();
}

void MlpNetwork::ensure_gpu_runtime() const {
    if (gpu_runtime_ == nullptr) {
        gpu_runtime_ = std::make_unique<GpuRuntimeImpl>();
    }
}

void MlpNetwork::invalidate_gpu_parameter_cache() {
    if (gpu_runtime_ != nullptr) {
        gpu_runtime_->parameters_synced = false;
    }
}

void MlpNetwork::sync_gpu_layers_to_host() {
    if (gpu_runtime_ == nullptr || !gpu_runtime_->parameters_synced) {
        return;
    }

    for (std::size_t i = 0; i < layers_.size(); ++i) {
        auto &state = gpu_runtime_->layer_states[i];
        if (!state.weights || !state.biases) {
            continue;
        }
        layers_[i].host_weights = state.weights->to_host<float>();
        layers_[i].host_biases = state.biases->to_host<float>();
    }
}

namespace {

GpuBuffer make_zero_buffer(GpuContext &context, std::size_t float_count) {
    std::vector<float> zeros(float_count, 0.0f);
    return GpuBuffer::from_host(context, zeros);
}

void enqueue_fill_float(GpuContext &context,
                        const GpuProgram &program,
                        GpuBuffer &buffer,
                        float value,
                        std::size_t float_count)
{
    GpuKernel fill_kernel(program, "fill_float", context);
    fill_kernel.set_args({
        KernelArg::buffer(buffer.get_buffer()),
        KernelArg::scalar_float(value),
        KernelArg::scalar_uint(static_cast<uint32_t>(float_count)),
    });
    fill_kernel.enqueue_1d(float_count);
}

} // namespace

MlpNetwork::ForwardCache MlpNetwork::forward_with_cache_gpu(const std::vector<float> &input) const {
    if (operations_.empty()) {
        throw std::runtime_error("Operation pipeline is empty");
    }
    ensure_gpu_runtime();

    if (!gpu_runtime_->parameters_synced) {
        gpu_runtime_->layer_states.clear();
        gpu_runtime_->layer_states.resize(layers_.size());
        for (std::size_t i = 0; i < layers_.size(); ++i) {
            auto &state = gpu_runtime_->layer_states[i];
            const DenseLayer &layer = layers_[i];
            state.weights = std::make_unique<GpuBuffer>(GpuBuffer::from_host(gpu_runtime_->context, layer.host_weights));
            state.biases = std::make_unique<GpuBuffer>(GpuBuffer::from_host(gpu_runtime_->context, layer.host_biases));
            state.grad_weights = std::make_unique<GpuBuffer>(make_zero_buffer(gpu_runtime_->context, layer.host_weights.size()));
            state.grad_biases = std::make_unique<GpuBuffer>(make_zero_buffer(gpu_runtime_->context, layer.host_biases.size()));

            state.velocity_weights = std::make_unique<GpuBuffer>(make_zero_buffer(gpu_runtime_->context, layer.host_weights.size()));
            state.velocity_biases = std::make_unique<GpuBuffer>(make_zero_buffer(gpu_runtime_->context, layer.host_biases.size()));
            state.adam_m_weights = std::make_unique<GpuBuffer>(make_zero_buffer(gpu_runtime_->context, layer.host_weights.size()));
            state.adam_v_weights = std::make_unique<GpuBuffer>(make_zero_buffer(gpu_runtime_->context, layer.host_weights.size()));
            state.adam_m_biases = std::make_unique<GpuBuffer>(make_zero_buffer(gpu_runtime_->context, layer.host_biases.size()));
            state.adam_v_biases = std::make_unique<GpuBuffer>(make_zero_buffer(gpu_runtime_->context, layer.host_biases.size()));
        }
        gpu_runtime_->parameters_synced = true;
    }

    ForwardCache cache;
    cache.op_inputs.reserve(operations_.size());
    cache.op_outputs.reserve(operations_.size());

    std::vector<float> current = input;
    std::size_t linear_index = 0;

    for (const auto &op : operations_) {
        cache.op_inputs.push_back(current);

        if (op.type == OperationType::Linear) {
            if (linear_index >= layers_.size()) {
                throw std::runtime_error("Internal linear layer index out of range");
            }

            const DenseLayer &layer = layers_[linear_index++];
            if (current.size() != layer.input_size) {
                throw std::invalid_argument("Input size mismatch for linear operation");
            }

            auto &device_state = gpu_runtime_->layer_states[linear_index - 1];

            GpuBuffer input_buf = GpuBuffer::from_host(gpu_runtime_->context, current);
            GpuBuffer output = GpuBuffer::allocate(gpu_runtime_->context, layer.output_size * sizeof(float));

            GpuKernel dense_kernel(gpu_runtime_->program, "dense_forward", gpu_runtime_->context);
            std::vector<KernelArg> args = {
                KernelArg::buffer(device_state.weights->get_buffer()),
                KernelArg::buffer(device_state.biases->get_buffer()),
                KernelArg::buffer(input_buf.get_buffer()),
                KernelArg::buffer(output.get_buffer()),
                KernelArg::scalar_uint(static_cast<uint32_t>(layer.input_size)),
            };
            dense_kernel.set_args(args);
            dense_kernel.enqueue_1d(layer.output_size);

            current = output.to_host<float>();
        } else {
            GpuBuffer values = GpuBuffer::from_host(gpu_runtime_->context, current);
            const char *kernel_name = nullptr;
            if (op.type == OperationType::Relu) {
                kernel_name = "relu_activation";
            } else if (op.type == OperationType::Sigmoid) {
                kernel_name = "sigmoid_activation";
            } else if (op.type == OperationType::Tanh) {
                kernel_name = "tanh_activation";
            } else if (op.type == OperationType::LeakyRelu) {
                kernel_name = "leaky_relu_activation";
            } else if (op.type == OperationType::Gelu) {
                kernel_name = "gelu_activation";
            }

            if (kernel_name != nullptr) {
                GpuKernel act_kernel(gpu_runtime_->program, kernel_name, gpu_runtime_->context);
                act_kernel.set_args({KernelArg::buffer(values.get_buffer())});
                act_kernel.enqueue_1d(current.size());
                current = values.to_host<float>();
            }
        }

        cache.op_outputs.push_back(current);
    }

    return cache;
}

std::vector<float> MlpNetwork::forward_gpu(const std::vector<float> &input) const {
    const ForwardCache cache = forward_with_cache_gpu(input);
    if (cache.op_outputs.empty()) {
        throw std::runtime_error("Forward cache is empty");
    }
    return cache.op_outputs.back();
}

std::vector<float> MlpNetwork::forward(const std::vector<float> &input) {
    if (execution_backend_ == ExecutionBackend::GPU) {
        return forward_gpu(input);
    }
    return forward_cpu(input);
}

float MlpNetwork::compute_loss(float prediction,
                               float target,
                               LossType loss_type,
                               float positive_weight,
                               float negative_weight,
                               float focal_gamma,
                               float focal_alpha) {
    if (loss_type == LossType::MSE) {
        const float diff = prediction - target;
        return diff * diff;
    }

    const float epsilon = 1e-7f;
    const float p = std::clamp(prediction, epsilon, 1.0f - epsilon);
    if (loss_type == LossType::Focal) {
        if (target >= 0.5f) {
            const float modulating = std::pow(1.0f - p, focal_gamma);
            return -(positive_weight * focal_alpha) * modulating * std::log(p);
        }
        const float modulating = std::pow(p, focal_gamma);
        return -(negative_weight * (1.0f - focal_alpha)) * modulating * std::log(1.0f - p);
    }
    return -(positive_weight * target * std::log(p) +
             negative_weight * (1.0f - target) * std::log(1.0f - p));
}

float MlpNetwork::loss_derivative_wrt_prediction(float prediction,
                                                 float target,
                                                 LossType loss_type,
                                                 float positive_weight,
                                                 float negative_weight,
                                                 float focal_gamma,
                                                 float focal_alpha) {
    if (loss_type == LossType::MSE) {
        return 2.0f * (prediction - target);
    }

    const float epsilon = 1e-7f;
    const float p = std::clamp(prediction, epsilon, 1.0f - epsilon);
    if (loss_type == LossType::Focal) {
        if (target >= 0.5f) {
            const float one_minus_p = 1.0f - p;
            const float modulating = std::pow(one_minus_p, focal_gamma);
            const float modulating_grad =
                focal_gamma * std::pow(one_minus_p, focal_gamma - 1.0f);
            const float positive_scale = positive_weight * focal_alpha;
            return positive_scale * (modulating_grad * std::log(p) - modulating / p);
        }
        const float modulating = std::pow(p, focal_gamma);
        const float modulating_grad = focal_gamma * std::pow(p, focal_gamma - 1.0f);
        const float negative_scale = negative_weight * (1.0f - focal_alpha);
        return negative_scale * (modulating / (1.0f - p) - modulating_grad * std::log(1.0f - p));
    }
    return -(positive_weight * target / p) + (negative_weight * (1.0f - target) / (1.0f - p));
}

float MlpNetwork::activation_derivative(OperationType op_type,
                                        float pre_activation,
                                        float post_activation)
{
    if (op_type == OperationType::Linear) {
        return 1.0f;
    }

    if (op_type == OperationType::Relu) {
        return pre_activation > 0.0f ? 1.0f : 0.0f;
    }

    if (op_type == OperationType::Tanh) {
        return 1.0f - post_activation * post_activation;
    }

    if (op_type == OperationType::Gelu) {
        const float x = pre_activation;
        const float c = 0.044715f;
        const float s = 0.7978845608f;
        const float x2 = x * x;
        const float x3 = x2 * x;
        const float u = s * (x + c * x3);
        const float t = std::tanh(u);
        const float sech2 = 1.0f - t * t;
        const float du_dx = s * (1.0f + 3.0f * c * x2);
        return 0.5f * (1.0f + t) + 0.5f * x * sech2 * du_dx;
    }

    if (op_type == OperationType::LeakyRelu) {
        return pre_activation > 0.0f ? 1.0f : 0.01f;
    }

    return post_activation * (1.0f - post_activation);
}

bool MlpNetwork::should_use_bce_sigmoid_shortcut(LossType loss_type) const {
    return enable_bce_sigmoid_shortcut_ &&
           optimizer_type_ == OptimizerType::SGD &&
           loss_type == LossType::BCE &&
           !operations_.empty() &&
           operations_.back().type == OperationType::Sigmoid;
}

namespace {

float pow_uint(float base, std::uint64_t exp) {
    float result = 1.0f;
    for (std::uint64_t i = 0; i < exp; ++i) {
        result *= base;
    }
    return result;
}

} // namespace

void MlpNetwork::backward_update_cpu(const ForwardCache &cache,
                                     float target,
                                     float learning_rate,
                                     LossType loss_type)
{
    if (cache.op_outputs.empty()) {
        throw std::runtime_error("Forward cache is empty");
    }

    const std::vector<float> &network_output = cache.op_outputs.back();
    if (network_output.size() != 1) {
        throw std::runtime_error("Current training path expects a single output unit");
    }

    const bool use_bce_sigmoid_shortcut = should_use_bce_sigmoid_shortcut(loss_type);
    std::vector<float> gradient(network_output.size(), 0.0f);
    gradient[0] = SgdOptimizer::output_gradient(
        network_output[0],
        target,
        loss_type,
        use_bce_sigmoid_shortcut,
        positive_class_weight_,
        negative_class_weight_,
        focal_gamma_,
        focal_alpha_,
        loss_derivative_wrt_prediction);

    std::size_t reverse_linear_index = layers_.size();

    for (std::size_t op_index = operations_.size(); op_index-- > 0;) {
        const OperationType op_type = operations_[op_index].type;
        const std::vector<float> &op_input = cache.op_inputs[op_index];
        const std::vector<float> &op_output = cache.op_outputs[op_index];

        if (op_type == OperationType::Relu ||
            op_type == OperationType::Sigmoid ||
            op_type == OperationType::Tanh ||
            op_type == OperationType::LeakyRelu ||
            op_type == OperationType::Gelu) {
            const bool skip_output_sigmoid_derivative =
                use_bce_sigmoid_shortcut &&
                op_type == OperationType::Sigmoid &&
                op_index + 1 == operations_.size();
            if (skip_output_sigmoid_derivative) {
                continue;
            }

            if (gradient.size() != op_output.size()) {
                throw std::runtime_error("Gradient size mismatch in activation backward pass");
            }

            for (std::size_t i = 0; i < gradient.size(); ++i) {
                const float local_derivative = activation_derivative(op_type, op_input[i], op_output[i]);
                gradient[i] *= local_derivative;
            }
            continue;
        }

        if (reverse_linear_index == 0) {
            throw std::runtime_error("Internal linear layer reverse index out of range");
        }

        DenseLayer &layer = layers_[--reverse_linear_index];
        if (gradient.size() != layer.output_size || op_input.size() != layer.input_size) {
            throw std::runtime_error("Gradient/input size mismatch in linear backward pass");
        }

        std::vector<float> input_gradient(layer.input_size, 0.0f);

        SgdOptimizer::update_dense_layer(layer, op_input, gradient, input_gradient, learning_rate);

        gradient = std::move(input_gradient);
    }
}

float MlpNetwork::evaluate_cost_cpu(const ManufacturingDefectDataset &dataset, LossType loss_type) const {
    if (dataset.size() == 0) {
        throw std::invalid_argument("Dataset is empty");
    }

    float total_loss = 0.0f;
    for (const auto &sample : dataset.samples()) {
        const std::vector<float> prediction = forward_cpu(sample.features);
        const float y_hat = prediction.empty() ? 0.0f : prediction[0];
        const float y = static_cast<float>(sample.label);
        total_loss += compute_loss(y_hat, y, loss_type, positive_class_weight_, negative_class_weight_, focal_gamma_, focal_alpha_);
    }

    return total_loss / static_cast<float>(dataset.size());
}

float MlpNetwork::evaluate_cost(const ManufacturingDefectDataset &dataset, LossType loss_type) const {
    if (execution_backend_ == ExecutionBackend::GPU) {
        return evaluate_cost_gpu(dataset, loss_type);
    }
    return evaluate_cost_cpu(dataset, loss_type);
}

float MlpNetwork::evaluate_cost_gpu(const ManufacturingDefectDataset &dataset, LossType loss_type) const {
    if (dataset.size() == 0) {
        throw std::invalid_argument("Dataset is empty");
    }

    float total_loss = 0.0f;
    for (const auto &sample : dataset.samples()) {
        const std::vector<float> prediction = forward_gpu(sample.features);
        const float y_hat = prediction.empty() ? 0.0f : prediction[0];
        const float y = static_cast<float>(sample.label);
        total_loss += compute_loss(y_hat, y, loss_type, positive_class_weight_, negative_class_weight_, focal_gamma_, focal_alpha_);
    }

    return total_loss / static_cast<float>(dataset.size());
}

float MlpNetwork::train_one_epoch_internal(const ManufacturingDefectDataset &dataset,
                                           float learning_rate,
                                           LossType loss_type,
                                           std::size_t batch_size)
{
    if (dataset.size() == 0) {
        throw std::invalid_argument("Dataset is empty");
    }
    if (learning_rate <= 0.0f) {
        throw std::invalid_argument("Learning rate must be positive");
    }
    if (batch_size == 0) {
        throw std::invalid_argument("Batch size must be positive");
    }

    float total_loss = 0.0f;
    const bool use_bce_sigmoid_shortcut = should_use_bce_sigmoid_shortcut(loss_type);

    struct LayerGradients {
        std::vector<float> d_weights;
        std::vector<float> d_biases;
    };

    std::vector<LayerGradients> batch_grads;
    batch_grads.reserve(layers_.size());
    for (const DenseLayer &layer : layers_) {
        batch_grads.push_back({
            std::vector<float>(layer.host_weights.size(), 0.0f),
            std::vector<float>(layer.host_biases.size(), 0.0f),
        });
    }

    auto clear_batch_grads = [&batch_grads]() {
        for (LayerGradients &g : batch_grads) {
            std::fill(g.d_weights.begin(), g.d_weights.end(), 0.0f);
            std::fill(g.d_biases.begin(), g.d_biases.end(), 0.0f);
        }
    };

    auto accumulate_sample_gradients = [&](const ForwardCache &cache, float target) {
        if (cache.op_outputs.empty()) {
            throw std::runtime_error("Forward cache is empty");
        }

        const std::vector<float> &network_output = cache.op_outputs.back();
        if (network_output.size() != 1) {
            throw std::runtime_error("Current training path expects a single output unit");
        }

        std::vector<float> gradient(network_output.size(), 0.0f);
        gradient[0] = SgdOptimizer::output_gradient(
            network_output[0],
            target,
            loss_type,
            use_bce_sigmoid_shortcut,
            positive_class_weight_,
            negative_class_weight_,
            focal_gamma_,
            focal_alpha_,
            loss_derivative_wrt_prediction);

        std::size_t reverse_linear_index = layers_.size();

        for (std::size_t op_index = operations_.size(); op_index-- > 0;) {
            const OperationType op_type = operations_[op_index].type;
            const std::vector<float> &op_input = cache.op_inputs[op_index];
            const std::vector<float> &op_output = cache.op_outputs[op_index];

            if (op_type == OperationType::Relu ||
                op_type == OperationType::Sigmoid ||
                op_type == OperationType::Tanh ||
                op_type == OperationType::LeakyRelu ||
                op_type == OperationType::Gelu) {
                const bool skip_output_sigmoid_derivative =
                    use_bce_sigmoid_shortcut &&
                    op_type == OperationType::Sigmoid &&
                    op_index + 1 == operations_.size();
                if (skip_output_sigmoid_derivative) {
                    continue;
                }

                if (gradient.size() != op_output.size()) {
                    throw std::runtime_error("Gradient size mismatch in activation backward pass");
                }

                for (std::size_t i = 0; i < gradient.size(); ++i) {
                    const float local_derivative = activation_derivative(op_type, op_input[i], op_output[i]);
                    gradient[i] *= local_derivative;
                }
                continue;
            }

            if (reverse_linear_index == 0) {
                throw std::runtime_error("Internal linear layer reverse index out of range");
            }

            DenseLayer &layer = layers_[--reverse_linear_index];
            LayerGradients &layer_grad = batch_grads[reverse_linear_index];
            if (gradient.size() != layer.output_size || op_input.size() != layer.input_size) {
                throw std::runtime_error("Gradient/input size mismatch in linear backward pass");
            }

            std::vector<float> input_gradient(layer.input_size, 0.0f);
            for (std::size_t out = 0; out < layer.output_size; ++out) {
                const float delta = gradient[out];
                layer_grad.d_biases[out] += delta;

                for (std::size_t in = 0; in < layer.input_size; ++in) {
                    const std::size_t weight_index = out * layer.input_size + in;
                    layer_grad.d_weights[weight_index] += delta * op_input[in];
                    input_gradient[in] += layer.host_weights[weight_index] * delta;
                }
            }

            gradient = std::move(input_gradient);
        }
    };

    std::vector<std::size_t> sample_indices(dataset.size());
    std::iota(sample_indices.begin(), sample_indices.end(), 0);
    static std::mt19937 rng(42u);
    std::shuffle(sample_indices.begin(), sample_indices.end(), rng);

    for (std::size_t begin = 0; begin < sample_indices.size(); begin += batch_size) {
        const std::size_t end = std::min(begin + batch_size, sample_indices.size());
        const std::size_t current_batch_size = end - begin;
        clear_batch_grads();

        for (std::size_t pos = begin; pos < end; ++pos) {
            const DefectSample &sample = dataset.sample(sample_indices[pos]);
            const ForwardCache cache = forward_with_cache_cpu(sample.features);
            const std::vector<float> &current = cache.op_outputs.back();
            const float target = static_cast<float>(sample.label);
            if (current.empty()) {
                throw std::runtime_error("Network output is empty");
            }

            total_loss += compute_loss(current[0], target, loss_type, positive_class_weight_, negative_class_weight_, focal_gamma_, focal_alpha_);
            accumulate_sample_gradients(cache, target);
        }

        const float inv_batch = 1.0f / static_cast<float>(current_batch_size);
        const std::uint64_t next_step = optimizer_step_ + 1;
        const float beta1_pow = pow_uint(adam_beta1_, next_step);
        const float beta2_pow = pow_uint(adam_beta2_, next_step);
        const float bias_corr1 = 1.0f - beta1_pow;
        const float bias_corr2 = 1.0f - beta2_pow;
        for (std::size_t layer_idx = 0; layer_idx < layers_.size(); ++layer_idx) {
            DenseLayer &layer = layers_[layer_idx];
            LayerGradients &layer_grad = batch_grads[layer_idx];
            OptimizerState &state = optimizer_states_[layer_idx];

            for (std::size_t i = 0; i < layer.host_weights.size(); ++i) {
                const float grad = layer_grad.d_weights[i] * inv_batch;
                if (optimizer_type_ == OptimizerType::SGD) {
                    layer.host_weights[i] -= learning_rate * grad;
                } else if (optimizer_type_ == OptimizerType::Momentum) {
                    state.momentum_w[i] = momentum_ * state.momentum_w[i] - learning_rate * grad;
                    layer.host_weights[i] += state.momentum_w[i];
                } else if (optimizer_type_ == OptimizerType::AdamW) {
                    state.adam_m_w[i] = adam_beta1_ * state.adam_m_w[i] + (1.0f - adam_beta1_) * grad;
                    state.adam_v_w[i] = adam_beta2_ * state.adam_v_w[i] + (1.0f - adam_beta2_) * grad * grad;
                    const float m_hat = state.adam_m_w[i] / bias_corr1;
                    const float v_hat = state.adam_v_w[i] / bias_corr2;
                    const float weight_before = layer.host_weights[i];
                    const float adam_step = m_hat / (std::sqrt(v_hat) + adam_epsilon_);
                    layer.host_weights[i] = weight_before - learning_rate * (adam_step + weight_decay_ * weight_before);
                } else {
                    state.adam_m_w[i] = adam_beta1_ * state.adam_m_w[i] + (1.0f - adam_beta1_) * grad;
                    state.adam_v_w[i] = adam_beta2_ * state.adam_v_w[i] + (1.0f - adam_beta2_) * grad * grad;
                    const float m_hat = state.adam_m_w[i] / bias_corr1;
                    const float v_hat = state.adam_v_w[i] / bias_corr2;
                    layer.host_weights[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + adam_epsilon_);
                }
            }
            for (std::size_t i = 0; i < layer.host_biases.size(); ++i) {
                const float grad = layer_grad.d_biases[i] * inv_batch;
                if (optimizer_type_ == OptimizerType::SGD) {
                    layer.host_biases[i] -= learning_rate * grad;
                } else if (optimizer_type_ == OptimizerType::Momentum) {
                    state.momentum_b[i] = momentum_ * state.momentum_b[i] - learning_rate * grad;
                    layer.host_biases[i] += state.momentum_b[i];
                } else if (optimizer_type_ == OptimizerType::AdamW) {
                    state.adam_m_b[i] = adam_beta1_ * state.adam_m_b[i] + (1.0f - adam_beta1_) * grad;
                    state.adam_v_b[i] = adam_beta2_ * state.adam_v_b[i] + (1.0f - adam_beta2_) * grad * grad;
                    const float m_hat = state.adam_m_b[i] / bias_corr1;
                    const float v_hat = state.adam_v_b[i] / bias_corr2;
                    layer.host_biases[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + adam_epsilon_);
                } else {
                    state.adam_m_b[i] = adam_beta1_ * state.adam_m_b[i] + (1.0f - adam_beta1_) * grad;
                    state.adam_v_b[i] = adam_beta2_ * state.adam_v_b[i] + (1.0f - adam_beta2_) * grad * grad;
                    const float m_hat = state.adam_m_b[i] / bias_corr1;
                    const float v_hat = state.adam_v_b[i] / bias_corr2;
                    layer.host_biases[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + adam_epsilon_);
                }
            }
        }
        optimizer_step_ = next_step;

    }

    invalidate_gpu_parameter_cache();
    return total_loss / static_cast<float>(dataset.size());
}

float MlpNetwork::train_one_epoch_cpu(const ManufacturingDefectDataset &dataset,
                                      float learning_rate,
                                      LossType loss_type,
                                      std::size_t batch_size)
{
    return train_one_epoch_internal(dataset, learning_rate, loss_type, batch_size);
}

void MlpNetwork::backward_update_gpu(const ForwardCache &cache,
                                     float target,
                                     float learning_rate,
                                     LossType loss_type)
{
    (void)learning_rate;
    ensure_gpu_runtime();
    if (cache.op_outputs.empty()) {
        throw std::runtime_error("Forward cache is empty");
    }

    const std::vector<float> &network_output = cache.op_outputs.back();
    if (network_output.size() != 1) {
        throw std::runtime_error("Current GPU training path expects a single output unit");
    }

    const bool use_bce_sigmoid_shortcut = should_use_bce_sigmoid_shortcut(loss_type);
    std::vector<float> gradient(network_output.size(), 0.0f);
    gradient[0] = SgdOptimizer::output_gradient(
        network_output[0],
        target,
        loss_type,
        use_bce_sigmoid_shortcut,
        positive_class_weight_,
        negative_class_weight_,
        focal_gamma_,
        focal_alpha_,
        loss_derivative_wrt_prediction);

    std::size_t reverse_linear_index = layers_.size();

    for (std::size_t op_index = operations_.size(); op_index-- > 0;) {
        const OperationType op_type = operations_[op_index].type;
        const std::vector<float> &op_input = cache.op_inputs[op_index];
        const std::vector<float> &op_output = cache.op_outputs[op_index];

        if (op_type == OperationType::Relu ||
            op_type == OperationType::Sigmoid ||
            op_type == OperationType::Tanh ||
            op_type == OperationType::LeakyRelu ||
            op_type == OperationType::Gelu) {
            const bool skip_output_sigmoid_derivative =
                use_bce_sigmoid_shortcut &&
                op_type == OperationType::Sigmoid &&
                op_index + 1 == operations_.size();
            if (skip_output_sigmoid_derivative) {
                continue;
            }

            if (gradient.size() != op_output.size()) {
                throw std::runtime_error("Gradient size mismatch in activation backward pass");
            }

            GpuBuffer grad_buf = GpuBuffer::from_host(gpu_runtime_->context, gradient);
            GpuBuffer deriv_buf = GpuBuffer::allocate(gpu_runtime_->context, gradient.size() * sizeof(float));
            const std::vector<float> *source = &op_output;
            const char *deriv_kernel_name = nullptr;
            if (op_type == OperationType::Relu) {
                deriv_kernel_name = "relu_derivative";
            } else if (op_type == OperationType::Sigmoid) {
                deriv_kernel_name = "sigmoid_derivative";
            } else if (op_type == OperationType::Tanh) {
                deriv_kernel_name = "tanh_derivative";
            } else if (op_type == OperationType::LeakyRelu) {
                deriv_kernel_name = "leaky_relu_derivative";
            } else if (op_type == OperationType::Gelu) {
                deriv_kernel_name = "gelu_derivative";
                source = &op_input;
            }

            GpuBuffer source_buf = GpuBuffer::from_host(gpu_runtime_->context, *source);
            GpuKernel deriv_kernel(gpu_runtime_->program, deriv_kernel_name, gpu_runtime_->context);
            deriv_kernel.set_args({
                KernelArg::buffer(source_buf.get_buffer()),
                KernelArg::buffer(deriv_buf.get_buffer()),
            });
            deriv_kernel.enqueue_1d(gradient.size());

            GpuKernel mul_kernel(gpu_runtime_->program, "elementwise_multiply_inplace", gpu_runtime_->context);
            mul_kernel.set_args({
                KernelArg::buffer(grad_buf.get_buffer()),
                KernelArg::buffer(deriv_buf.get_buffer()),
            });
            mul_kernel.enqueue_1d(gradient.size());

            gradient = grad_buf.to_host<float>();
            continue;
        }

        if (reverse_linear_index == 0) {
            throw std::runtime_error("Internal linear layer reverse index out of range");
        }

        DenseLayer &layer = layers_[--reverse_linear_index];
        if (gradient.size() != layer.output_size || op_input.size() != layer.input_size) {
            throw std::runtime_error("Gradient/input size mismatch in linear backward pass");
        }

        auto &device_state = gpu_runtime_->layer_states[reverse_linear_index];
        if (!device_state.weights || !device_state.biases || !device_state.grad_weights || !device_state.grad_biases) {
            throw std::runtime_error("GPU layer state is not initialized");
        }

        GpuBuffer input_buf = GpuBuffer::from_host(gpu_runtime_->context, op_input);
        GpuBuffer grad_output_buf = GpuBuffer::from_host(gpu_runtime_->context, gradient);
        GpuBuffer grad_input_buf = GpuBuffer::allocate(gpu_runtime_->context, layer.input_size * sizeof(float));

        GpuKernel grad_input_kernel(gpu_runtime_->program, "dense_backward_input", gpu_runtime_->context);
        grad_input_kernel.set_args({
            KernelArg::buffer(device_state.weights->get_buffer()),
            KernelArg::buffer(grad_output_buf.get_buffer()),
            KernelArg::buffer(grad_input_buf.get_buffer()),
            KernelArg::scalar_uint(static_cast<uint32_t>(layer.input_size)),
            KernelArg::scalar_uint(static_cast<uint32_t>(layer.output_size)),
        });
        grad_input_kernel.enqueue_1d(layer.input_size);

        GpuKernel accumulate_kernel(gpu_runtime_->program, "dense_accumulate_grads", gpu_runtime_->context);
        accumulate_kernel.set_args({
            KernelArg::buffer(device_state.grad_weights->get_buffer()),
            KernelArg::buffer(device_state.grad_biases->get_buffer()),
            KernelArg::buffer(input_buf.get_buffer()),
            KernelArg::buffer(grad_output_buf.get_buffer()),
            KernelArg::scalar_uint(static_cast<uint32_t>(layer.input_size)),
        });
        accumulate_kernel.enqueue_1d(layer.output_size);

        gradient = grad_input_buf.to_host<float>();
    }
}

float MlpNetwork::train_one_epoch_internal_gpu(const ManufacturingDefectDataset &dataset,
                                               float learning_rate,
                                               LossType loss_type,
                                               std::size_t batch_size)
{
    if (dataset.size() == 0) {
        throw std::invalid_argument("Dataset is empty");
    }
    if (learning_rate <= 0.0f) {
        throw std::invalid_argument("Learning rate must be positive");
    }
    if (batch_size == 0) {
        throw std::invalid_argument("Batch size must be positive");
    }

    ensure_gpu_runtime();
    if (gpu_runtime_ == nullptr) {
        throw std::runtime_error("Failed to initialize GPU runtime");
    }

    float total_loss = 0.0f;
    std::vector<std::size_t> sample_indices(dataset.size());
    std::iota(sample_indices.begin(), sample_indices.end(), 0);
    static std::mt19937 rng(42u);
    std::shuffle(sample_indices.begin(), sample_indices.end(), rng);

    if (gpu_runtime_->layer_states.size() != layers_.size() || !gpu_runtime_->parameters_synced) {
        const DefectSample &warmup_sample = dataset.sample(sample_indices.front());
        (void)forward_with_cache_gpu(warmup_sample.features);
    }

    auto clear_batch_gradients = [&]() {
        for (std::size_t layer_idx = 0; layer_idx < layers_.size(); ++layer_idx) {
            const DenseLayer &layer = layers_[layer_idx];
            auto &state = gpu_runtime_->layer_states[layer_idx];
            enqueue_fill_float(gpu_runtime_->context,
                               gpu_runtime_->program,
                               *state.grad_weights,
                               0.0f,
                               layer.host_weights.size());
            enqueue_fill_float(gpu_runtime_->context,
                               gpu_runtime_->program,
                               *state.grad_biases,
                               0.0f,
                               layer.host_biases.size());
        }
    };

    auto apply_optimizer_update = [&](std::size_t actual_batch_size) {
        const float inv_batch = 1.0f / static_cast<float>(actual_batch_size);
        const std::uint64_t next_step = optimizer_step_ + 1;
        const float beta1_pow = pow_uint(adam_beta1_, next_step);
        const float beta2_pow = pow_uint(adam_beta2_, next_step);
        const float bias_corr1 = 1.0f - beta1_pow;
        const float bias_corr2 = 1.0f - beta2_pow;

        for (std::size_t layer_idx = 0; layer_idx < layers_.size(); ++layer_idx) {
            const DenseLayer &layer = layers_[layer_idx];
            auto &state = gpu_runtime_->layer_states[layer_idx];

            if (optimizer_type_ == OptimizerType::SGD) {
                GpuKernel update_kernel(gpu_runtime_->program, "dense_sgd_update", gpu_runtime_->context);
                update_kernel.set_args({
                    KernelArg::buffer(state.weights->get_buffer()),
                    KernelArg::buffer(state.biases->get_buffer()),
                    KernelArg::buffer(state.grad_weights->get_buffer()),
                    KernelArg::buffer(state.grad_biases->get_buffer()),
                    KernelArg::scalar_float(learning_rate),
                    KernelArg::scalar_float(inv_batch),
                    KernelArg::scalar_uint(static_cast<uint32_t>(layer.input_size)),
                });
                update_kernel.enqueue_1d(layer.output_size);
            } else if (optimizer_type_ == OptimizerType::Momentum) {
                GpuKernel update_kernel(gpu_runtime_->program, "dense_momentum_update", gpu_runtime_->context);
                update_kernel.set_args({
                    KernelArg::buffer(state.weights->get_buffer()),
                    KernelArg::buffer(state.biases->get_buffer()),
                    KernelArg::buffer(state.velocity_weights->get_buffer()),
                    KernelArg::buffer(state.velocity_biases->get_buffer()),
                    KernelArg::buffer(state.grad_weights->get_buffer()),
                    KernelArg::buffer(state.grad_biases->get_buffer()),
                    KernelArg::scalar_float(learning_rate),
                    KernelArg::scalar_float(momentum_),
                    KernelArg::scalar_float(inv_batch),
                    KernelArg::scalar_uint(static_cast<uint32_t>(layer.input_size)),
                });
                update_kernel.enqueue_1d(layer.output_size);
            } else {
                GpuKernel update_kernel(gpu_runtime_->program, "dense_adam_update", gpu_runtime_->context);
                update_kernel.set_args({
                    KernelArg::buffer(state.weights->get_buffer()),
                    KernelArg::buffer(state.biases->get_buffer()),
                    KernelArg::buffer(state.adam_m_weights->get_buffer()),
                    KernelArg::buffer(state.adam_v_weights->get_buffer()),
                    KernelArg::buffer(state.adam_m_biases->get_buffer()),
                    KernelArg::buffer(state.adam_v_biases->get_buffer()),
                    KernelArg::buffer(state.grad_weights->get_buffer()),
                    KernelArg::buffer(state.grad_biases->get_buffer()),
                    KernelArg::scalar_float(learning_rate),
                    KernelArg::scalar_float(adam_beta1_),
                    KernelArg::scalar_float(adam_beta2_),
                    KernelArg::scalar_float(adam_epsilon_),
                    KernelArg::scalar_float(weight_decay_),
                    KernelArg::scalar_float(bias_corr1),
                    KernelArg::scalar_float(bias_corr2),
                    KernelArg::scalar_float(inv_batch),
                    KernelArg::scalar_uint(static_cast<uint32_t>(layer.input_size)),
                });
                update_kernel.enqueue_1d(layer.output_size);
            }
        }
        optimizer_step_ = next_step;
    };

    for (std::size_t begin = 0; begin < sample_indices.size(); begin += batch_size) {
        const std::size_t end = std::min(begin + batch_size, sample_indices.size());
        const std::size_t current_batch_size = end - begin;

        clear_batch_gradients();

        for (std::size_t pos = begin; pos < end; ++pos) {
            const DefectSample &sample = dataset.sample(sample_indices[pos]);
            const ForwardCache cache = forward_with_cache_gpu(sample.features);
            const std::vector<float> &current = cache.op_outputs.back();
            if (current.empty()) {
                throw std::runtime_error("Network output is empty");
            }

            const float target = static_cast<float>(sample.label);
            total_loss += compute_loss(current[0], target, loss_type, positive_class_weight_, negative_class_weight_, focal_gamma_, focal_alpha_);
            backward_update_gpu(cache, target, learning_rate, loss_type);
        }

        apply_optimizer_update(current_batch_size);
    }

    gpu_runtime_->context.get_queue().finish();
    sync_gpu_layers_to_host();

    return total_loss / static_cast<float>(dataset.size());
}

float MlpNetwork::train_one_epoch_gpu(const ManufacturingDefectDataset &dataset,
                                      float learning_rate,
                                      LossType loss_type,
                                      std::size_t batch_size)
{
    return train_one_epoch_internal_gpu(dataset, learning_rate, loss_type, batch_size);
}

float MlpNetwork::train_one_epoch(const ManufacturingDefectDataset &dataset,
                                  float learning_rate,
                                  LossType loss_type,
                                  std::size_t batch_size)
{
    if (execution_backend_ == ExecutionBackend::GPU) {
        return train_one_epoch_gpu(dataset, learning_rate, loss_type, batch_size);
    }
    return train_one_epoch_cpu(dataset, learning_rate, loss_type, batch_size);
}

void MlpNetwork::save_to_file(const std::filesystem::path &model_path) const {
    std::ofstream out(model_path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open model file for write: " + model_path.string());
    }

    write_binary(out, kModelMagic);
    write_binary(out, kModelFileVersion);

    const uint32_t op_count = static_cast<uint32_t>(operations_.size());
    const uint32_t layer_count = static_cast<uint32_t>(layers_.size());
    write_binary(out, op_count);
    write_binary(out, layer_count);

    for (const auto &op : operations_) {
        write_binary(out, static_cast<uint32_t>(op.type));
        write_binary(out, static_cast<uint64_t>(op.input_size));
        write_binary(out, static_cast<uint64_t>(op.output_size));
    }

    for (const auto &layer : layers_) {
        write_binary(out, static_cast<uint64_t>(layer.input_size));
        write_binary(out, static_cast<uint64_t>(layer.output_size));

        const uint64_t weight_count = static_cast<uint64_t>(layer.host_weights.size());
        const uint64_t bias_count = static_cast<uint64_t>(layer.host_biases.size());
        write_binary(out, weight_count);
        write_binary(out, bias_count);

        out.write(reinterpret_cast<const char *>(layer.host_weights.data()),
                  static_cast<std::streamsize>(weight_count * sizeof(float)));
        out.write(reinterpret_cast<const char *>(layer.host_biases.data()),
                  static_cast<std::streamsize>(bias_count * sizeof(float)));
        if (!out.good()) {
            throw std::runtime_error("Failed while writing model parameters");
        }
    }
}

std::vector<OperationConfig> MlpNetwork::load_operations_from_file(const std::filesystem::path &model_path) {
    std::ifstream in(model_path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open model file for read: " + model_path.string());
    }

    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t op_count = 0;
    uint32_t layer_count = 0;
    read_binary(in, magic);
    read_binary(in, version);
    read_binary(in, op_count);
    read_binary(in, layer_count);

    if (magic != kModelMagic) {
        throw std::runtime_error("Invalid model file magic");
    }
    if (version != kModelFileVersion) {
        throw std::runtime_error("Unsupported model file version: " + std::to_string(version));
    }
    if (op_count == 0 || layer_count == 0) {
        throw std::runtime_error("Model file contains an empty operation pipeline");
    }

    std::vector<OperationConfig> operations;
    operations.reserve(op_count);
    for (std::size_t i = 0; i < op_count; ++i) {
        uint32_t op_type_raw = 0;
        uint64_t input_size = 0;
        uint64_t output_size = 0;
        read_binary(in, op_type_raw);
        read_binary(in, input_size);
        read_binary(in, output_size);

        operations.push_back(OperationConfig{
            static_cast<OperationType>(op_type_raw),
            static_cast<std::size_t>(input_size),
            static_cast<std::size_t>(output_size)
        });
    }

    return operations;
}

void MlpNetwork::load_from_file(const std::filesystem::path &model_path) {
    std::ifstream in(model_path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open model file for read: " + model_path.string());
    }

    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t op_count = 0;
    uint32_t layer_count = 0;
    read_binary(in, magic);
    read_binary(in, version);
    read_binary(in, op_count);
    read_binary(in, layer_count);

    if (magic != kModelMagic) {
        throw std::runtime_error("Invalid model file magic");
    }
    if (version != kModelFileVersion) {
        throw std::runtime_error("Unsupported model file version: " + std::to_string(version));
    }
    if (op_count != operations_.size() || layer_count != layers_.size()) {
        throw std::runtime_error("Model architecture mismatch (operation/layer count)");
    }

    for (std::size_t i = 0; i < operations_.size(); ++i) {
        uint32_t op_type_raw = 0;
        uint64_t input_size = 0;
        uint64_t output_size = 0;
        read_binary(in, op_type_raw);
        read_binary(in, input_size);
        read_binary(in, output_size);

        if (op_type_raw != static_cast<uint32_t>(operations_[i].type) ||
            input_size != operations_[i].input_size ||
            output_size != operations_[i].output_size) {
            throw std::runtime_error("Model operation config mismatch");
        }
    }

    for (std::size_t i = 0; i < layers_.size(); ++i) {
        DenseLayer &layer = layers_[i];
        uint64_t input_size = 0;
        uint64_t output_size = 0;
        uint64_t weight_count = 0;
        uint64_t bias_count = 0;
        read_binary(in, input_size);
        read_binary(in, output_size);
        read_binary(in, weight_count);
        read_binary(in, bias_count);

        if (input_size != layer.input_size || output_size != layer.output_size) {
            throw std::runtime_error("Model layer shape mismatch");
        }
        if (weight_count != layer.host_weights.size() || bias_count != layer.host_biases.size()) {
            throw std::runtime_error("Model parameter size mismatch");
        }

        in.read(reinterpret_cast<char *>(layer.host_weights.data()),
                static_cast<std::streamsize>(weight_count * sizeof(float)));
        in.read(reinterpret_cast<char *>(layer.host_biases.data()),
                static_cast<std::streamsize>(bias_count * sizeof(float)));
        if (!in.good()) {
            throw std::runtime_error("Failed while reading model parameters");
        }
    }

    invalidate_gpu_parameter_cache();

}

MlpNetwork::~MlpNetwork() = default;