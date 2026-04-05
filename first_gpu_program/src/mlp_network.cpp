#include "mlp_network.h"

#include <algorithm>
#include <cmath>
#include <fstream>
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

MlpNetwork::DenseLayer::DenseLayer(std::size_t input,
                                   std::size_t output,
                                   ActivationType activation_type,
                                   GpuContext &ctx)
    : input_size(input),
      output_size(output),
      activation(activation_type),
    host_weights(input * output),
    host_biases(output),
      weights(GpuBuffer::allocate(ctx, input * output * sizeof(float))),
      biases(GpuBuffer::allocate(ctx, output * sizeof(float))) {}

float MlpNetwork::SgdOptimizer::output_gradient(
    float prediction,
    float target,
    LossType loss_type,
    bool use_bce_sigmoid_shortcut,
    float positive_weight,
    float negative_weight,
    float (*loss_derivative_wrt_prediction_fn)(float, float, LossType, float, float))
{
    if (use_bce_sigmoid_shortcut) {
        return positive_weight * target * (prediction - 1.0f) + negative_weight * (1.0f - target) * prediction;
    }
    return loss_derivative_wrt_prediction_fn(prediction, target, loss_type, positive_weight, negative_weight);
}

void MlpNetwork::SgdOptimizer::update_dense_layer_cpu(DenseLayer &layer,
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

MlpNetwork::MlpNetwork(GpuContext &ctx, std::vector<std::size_t> layer_sizes)
    : MlpNetwork(ctx, build_operations_from_layer_sizes(layer_sizes)) {}

MlpNetwork::MlpNetwork(GpuContext &ctx, std::vector<OperationConfig> operations)
    : context_(ctx), gpu_prog_(create_kernels(ctx)) {
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
                operation_to_activation(next_op_type),
                context_);
            initialize_layer(layers_.back(), next_op_type);
        }
    }
}

GpuProgram MlpNetwork::create_kernels(GpuContext &ctx) {
    return gpu::create_gpu_program(ctx);
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
    sync_layer_to_device_gpu(layer);
}

void MlpNetwork::sync_layer_to_device_gpu(DenseLayer &layer) {
    layer.weights.copy_from_host(layer.host_weights.data(),
                                 layer.host_weights.size() * sizeof(float));
    layer.biases.copy_from_host(layer.host_biases.data(),
                                layer.host_biases.size() * sizeof(float));
}

void MlpNetwork::sync_all_layers_to_device_gpu() {
    for (DenseLayer &layer : layers_) {
        sync_layer_to_device_gpu(layer);
    }
}

void MlpNetwork::sync_layer_to_host_gpu(DenseLayer &layer) {
    layer.weights.copy_to_host(layer.host_weights.data(),
                               layer.host_weights.size() * sizeof(float));
    layer.biases.copy_to_host(layer.host_biases.data(),
                              layer.host_biases.size() * sizeof(float));
}

void MlpNetwork::sync_all_layers_to_host_gpu() {
    for (DenseLayer &layer : layers_) {
        sync_layer_to_host_gpu(layer);
    }
}

void MlpNetwork::set_class_weights(float positive_weight, float negative_weight) {
    if (positive_weight <= 0.0f || negative_weight <= 0.0f || !std::isfinite(positive_weight) ||
        !std::isfinite(negative_weight)) {
        throw std::invalid_argument("Class weights must be positive finite numbers");
    }
    positive_class_weight_ = positive_weight;
    negative_class_weight_ = negative_weight;
}

void MlpNetwork::run_dense_layer_gpu(DenseLayer &layer, GpuBuffer &input, GpuBuffer &output) {
    gpu::DenseKernelOp<gpu::DenseForwardKernel> dense_op(gpu_prog_, context_);
    dense_op.forward(layer.weights, layer.biases, input, output,
                     static_cast<uint32_t>(layer.input_size));
}

void MlpNetwork::apply_activation_gpu(ActivationType type, GpuBuffer &values) {
    if (type == ActivationType::Linear) {
        return;
    }

    if (type == ActivationType::Relu) {
        gpu::ElementwiseKernelOp<gpu::ReluActivationKernel> relu_op(gpu_prog_, context_);
        relu_op.apply(values);
    } else {
        gpu::ElementwiseKernelOp<gpu::SigmoidActivationKernel> sigmoid_op(gpu_prog_, context_);
        sigmoid_op.apply(values);
    }
}

std::vector<float> MlpNetwork::forward_internal_gpu(const std::vector<float> &input,
                                                    ForwardCache *cache)
{
    if (operations_.empty()) {
        throw std::runtime_error("Operation pipeline is empty");
    }

    if (cache != nullptr) {
        cache->op_inputs.clear();
        cache->op_outputs.clear();
        cache->op_inputs.reserve(operations_.size());
        cache->op_outputs.reserve(operations_.size());
    }

    GpuBuffer current = GpuBuffer::from_host(context_, input);
    std::size_t current_dim = input.size();
    std::size_t linear_index = 0;

    for (const auto &op : operations_) {
        if (cache != nullptr) {
            cache->op_inputs.push_back(current.to_host<float>());
        }

        if (op.type == OperationType::Linear) {
            if (linear_index >= layers_.size()) {
                throw std::runtime_error("Internal linear layer index out of range");
            }

            DenseLayer &layer = layers_[linear_index++];
            if (current_dim != layer.input_size) {
                throw std::invalid_argument("Input size mismatch for linear operation");
            }

            GpuBuffer output = GpuBuffer::allocate(context_, layer.output_size * sizeof(float));
            run_dense_layer_gpu(layer, current, output);
            current = std::move(output);
            current_dim = layer.output_size;
        } else {
            apply_activation_gpu(operation_to_activation(op.type), current);
        }

        if (cache != nullptr) {
            cache->op_outputs.push_back(current.to_host<float>());
        }
    }

    return current.to_host<float>();
}

MlpNetwork::ForwardCache MlpNetwork::forward_with_cache_gpu(const std::vector<float> &input) {
    ForwardCache cache;
    (void)forward_internal_gpu(input, &cache);
    return cache;
}

MlpNetwork::ForwardCache MlpNetwork::forward_with_cache_for_backend(const std::vector<float> &input,
                                                                    ExecutionBackend backend) const
{
    if (backend == ExecutionBackend::CPU) {
        return forward_with_cache_cpu(input);
    }
    return const_cast<MlpNetwork *>(this)->forward_with_cache_gpu(input);
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

std::vector<float> MlpNetwork::forward_gpu(const std::vector<float> &input) {
    return forward_internal_gpu(input, nullptr);
}

std::vector<float> MlpNetwork::forward(const std::vector<float> &input) {
    if (execution_backend_ == ExecutionBackend::CPU) {
        return forward_cpu(input);
    }
    return forward_gpu(input);
}

float MlpNetwork::compute_loss(float prediction,
                               float target,
                               LossType loss_type,
                               float positive_weight,
                               float negative_weight) {
    if (loss_type == LossType::MSE) {
        const float diff = prediction - target;
        return diff * diff;
    }

    const float epsilon = 1e-7f;
    const float p = std::clamp(prediction, epsilon, 1.0f - epsilon);
    return -(positive_weight * target * std::log(p) +
             negative_weight * (1.0f - target) * std::log(1.0f - p));
}

float MlpNetwork::loss_derivative_wrt_prediction(float prediction,
                                                 float target,
                                                 LossType loss_type,
                                                 float positive_weight,
                                                 float negative_weight) {
    if (loss_type == LossType::MSE) {
        return 2.0f * (prediction - target);
    }

    const float epsilon = 1e-7f;
    const float p = std::clamp(prediction, epsilon, 1.0f - epsilon);
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

    return post_activation * (1.0f - post_activation);
}

bool MlpNetwork::should_use_bce_sigmoid_shortcut(LossType loss_type) const {
    return enable_bce_sigmoid_shortcut_ &&
           optimizer_type_ == OptimizerType::SGD &&
           loss_type == LossType::BCE &&
           !operations_.empty() &&
           operations_.back().type == OperationType::Sigmoid;
}

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
        loss_derivative_wrt_prediction);

    std::size_t reverse_linear_index = layers_.size();

    for (std::size_t op_index = operations_.size(); op_index-- > 0;) {
        const OperationType op_type = operations_[op_index].type;
        const std::vector<float> &op_input = cache.op_inputs[op_index];
        const std::vector<float> &op_output = cache.op_outputs[op_index];

        if (op_type == OperationType::Relu || op_type == OperationType::Sigmoid) {
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

        SgdOptimizer::update_dense_layer_cpu(layer, op_input, gradient, input_gradient, learning_rate);

        gradient = std::move(input_gradient);
    }
}

void MlpNetwork::backward_update_gpu(const ForwardCache &cache,
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

    const std::vector<float> target_vec(network_output.size(), target);
    GpuBuffer prediction_buf = GpuBuffer::from_host(context_, network_output);
    GpuBuffer target_buf = GpuBuffer::from_host(context_, target_vec);
    GpuBuffer gradient = GpuBuffer::allocate(context_, network_output.size() * sizeof(float));

    if (use_bce_sigmoid_shortcut) {
        std::vector<float> grad_host(network_output.size(), 0.0f);
        grad_host[0] = SgdOptimizer::output_gradient(
                network_output[0],
                target,
                loss_type,
                true,
                positive_class_weight_,
                negative_class_weight_,
                loss_derivative_wrt_prediction);
        gradient.copy_from_host(grad_host.data(), grad_host.size() * sizeof(float));
    } else if (loss_type == LossType::MSE) {
        GpuKernel loss_grad_kernel(gpu_prog_, "mse_loss_derivative", context_);
        loss_grad_kernel.set_args({
            KernelArg::buffer(prediction_buf.get_buffer()),
            KernelArg::buffer(target_buf.get_buffer()),
            KernelArg::buffer(gradient.get_buffer()),
            KernelArg::scalar_float(1.0f),
        });
        loss_grad_kernel.enqueue_1d(network_output.size());
    } else {
        GpuKernel loss_grad_kernel(gpu_prog_, "bce_loss_derivative", context_);
        loss_grad_kernel.set_args({
            KernelArg::buffer(prediction_buf.get_buffer()),
            KernelArg::buffer(target_buf.get_buffer()),
            KernelArg::buffer(gradient.get_buffer()),
            KernelArg::scalar_float(1e-7f),
            KernelArg::scalar_float(1.0f),
            KernelArg::scalar_float(positive_class_weight_),
            KernelArg::scalar_float(negative_class_weight_),
        });
        loss_grad_kernel.enqueue_1d(network_output.size());
    }

    std::size_t reverse_linear_index = layers_.size();

    for (std::size_t op_index = operations_.size(); op_index-- > 0;) {
        const OperationType op_type = operations_[op_index].type;
        const std::vector<float> &op_input = cache.op_inputs[op_index];
        const std::vector<float> &op_output = cache.op_outputs[op_index];

        if (op_type == OperationType::Relu || op_type == OperationType::Sigmoid) {
            const bool skip_output_sigmoid_derivative =
                use_bce_sigmoid_shortcut &&
                op_type == OperationType::Sigmoid &&
                op_index + 1 == operations_.size();
            if (skip_output_sigmoid_derivative) {
                continue;
            }

            GpuBuffer derivative = GpuBuffer::allocate(context_, op_output.size() * sizeof(float));
            GpuBuffer activated = GpuBuffer::from_host(context_, op_output);

            if (op_type == OperationType::Relu) {
                GpuKernel derivative_kernel(gpu_prog_, "relu_derivative", context_);
                derivative_kernel.set_args({
                    KernelArg::buffer(activated.get_buffer()),
                    KernelArg::buffer(derivative.get_buffer()),
                });
                derivative_kernel.enqueue_1d(op_output.size());
            } else {
                GpuKernel derivative_kernel(gpu_prog_, "sigmoid_derivative", context_);
                derivative_kernel.set_args({
                    KernelArg::buffer(activated.get_buffer()),
                    KernelArg::buffer(derivative.get_buffer()),
                });
                derivative_kernel.enqueue_1d(op_output.size());
            }

            GpuKernel multiply_kernel(gpu_prog_, "elementwise_multiply_inplace", context_);
            multiply_kernel.set_args({
                KernelArg::buffer(gradient.get_buffer()),
                KernelArg::buffer(derivative.get_buffer()),
            });
            multiply_kernel.enqueue_1d(op_output.size());
            continue;
        }

        if (reverse_linear_index == 0) {
            throw std::runtime_error("Internal linear layer reverse index out of range");
        }

        DenseLayer &layer = layers_[--reverse_linear_index];
        if (op_input.size() != layer.input_size) {
            throw std::runtime_error("Input size mismatch in linear backward pass");
        }

        GpuBuffer input_buf = GpuBuffer::from_host(context_, op_input);
        GpuBuffer input_gradient = GpuBuffer::allocate(context_, layer.input_size * sizeof(float));

        GpuKernel input_grad_kernel(gpu_prog_, "dense_backward_input", context_);
        input_grad_kernel.set_args({
            KernelArg::buffer(layer.weights.get_buffer()),
            KernelArg::buffer(gradient.get_buffer()),
            KernelArg::buffer(input_gradient.get_buffer()),
            KernelArg::scalar_uint(static_cast<uint32_t>(layer.input_size)),
            KernelArg::scalar_uint(static_cast<uint32_t>(layer.output_size)),
        });
        input_grad_kernel.enqueue_1d(layer.input_size);

        GpuKernel update_kernel(gpu_prog_, "dense_sgd_update", context_);
        update_kernel.set_args({
            KernelArg::buffer(layer.weights.get_buffer()),
            KernelArg::buffer(layer.biases.get_buffer()),
            KernelArg::buffer(input_buf.get_buffer()),
            KernelArg::buffer(gradient.get_buffer()),
            KernelArg::scalar_float(learning_rate),
            KernelArg::scalar_uint(static_cast<uint32_t>(layer.input_size)),
        });
        update_kernel.enqueue_1d(layer.output_size);

        gradient = std::move(input_gradient);
    }

    context_.get_queue().finish();
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
        total_loss += compute_loss(y_hat, y, loss_type, positive_class_weight_, negative_class_weight_);
    }

    return total_loss / static_cast<float>(dataset.size());
}

float MlpNetwork::evaluate_cost_gpu(const ManufacturingDefectDataset &dataset, LossType loss_type) {
    if (dataset.size() == 0) {
        throw std::invalid_argument("Dataset is empty");
    }

    float total_loss = 0.0f;
    for (const auto &sample : dataset.samples()) {
        const std::vector<float> prediction = forward_gpu(sample.features);
        const float y_hat = prediction.empty() ? 0.0f : prediction[0];
        const float y = static_cast<float>(sample.label);
        total_loss += compute_loss(y_hat, y, loss_type, positive_class_weight_, negative_class_weight_);
    }

    return total_loss / static_cast<float>(dataset.size());
}

float MlpNetwork::evaluate_cost(const ManufacturingDefectDataset &dataset, LossType loss_type) const {
    if (execution_backend_ == ExecutionBackend::CPU) {
        return evaluate_cost_cpu(dataset, loss_type);
    }
    return const_cast<MlpNetwork *>(this)->evaluate_cost_gpu(dataset, loss_type);
}

float MlpNetwork::train_one_epoch_internal(const ManufacturingDefectDataset &dataset,
                                           float learning_rate,
                                           LossType loss_type,
                                           ExecutionBackend backend,
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
            loss_derivative_wrt_prediction);

        std::size_t reverse_linear_index = layers_.size();

        for (std::size_t op_index = operations_.size(); op_index-- > 0;) {
            const OperationType op_type = operations_[op_index].type;
            const std::vector<float> &op_input = cache.op_inputs[op_index];
            const std::vector<float> &op_output = cache.op_outputs[op_index];

            if (op_type == OperationType::Relu || op_type == OperationType::Sigmoid) {
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
            const ForwardCache cache = forward_with_cache_for_backend(sample.features, backend);
            const std::vector<float> &current = cache.op_outputs.back();
            const float target = static_cast<float>(sample.label);
            if (current.empty()) {
                throw std::runtime_error("Network output is empty");
            }

            total_loss += compute_loss(current[0], target, loss_type, positive_class_weight_, negative_class_weight_);
            accumulate_sample_gradients(cache, target);
        }

        const float inv_batch = 1.0f / static_cast<float>(current_batch_size);
        for (std::size_t layer_idx = 0; layer_idx < layers_.size(); ++layer_idx) {
            DenseLayer &layer = layers_[layer_idx];
            LayerGradients &layer_grad = batch_grads[layer_idx];

            for (std::size_t i = 0; i < layer.host_weights.size(); ++i) {
                layer.host_weights[i] -= learning_rate * layer_grad.d_weights[i] * inv_batch;
            }
            for (std::size_t i = 0; i < layer.host_biases.size(); ++i) {
                layer.host_biases[i] -= learning_rate * layer_grad.d_biases[i] * inv_batch;
            }
        }

        if (backend == ExecutionBackend::GPU) {
            // GPU forward path reads device-side parameters, so batch updates must be flushed each step.
            sync_all_layers_to_device_gpu();
        }
    }

    if (backend == ExecutionBackend::CPU) {
        // Keep device parameters synchronized in case backend switches to GPU later.
        sync_all_layers_to_device_gpu();
    } else {
        // Keep host parameters synchronized in case backend switches to CPU later.
        sync_all_layers_to_host_gpu();
    }

    return total_loss / static_cast<float>(dataset.size());
}

float MlpNetwork::train_one_epoch_cpu(const ManufacturingDefectDataset &dataset,
                                      float learning_rate,
                                      LossType loss_type,
                                      std::size_t batch_size)
{
    return train_one_epoch_internal(dataset, learning_rate, loss_type, ExecutionBackend::CPU, batch_size);
}

float MlpNetwork::train_one_epoch_gpu(const ManufacturingDefectDataset &dataset,
                                      float learning_rate,
                                      LossType loss_type,
                                      std::size_t batch_size)
{
    return train_one_epoch_internal(dataset, learning_rate, loss_type, ExecutionBackend::GPU, batch_size);
}

float MlpNetwork::train_one_epoch(const ManufacturingDefectDataset &dataset,
                                  float learning_rate,
                                  LossType loss_type,
                                  std::size_t batch_size)
{
    if (execution_backend_ == ExecutionBackend::CPU) {
        return train_one_epoch_cpu(dataset, learning_rate, loss_type, batch_size);
    }
    return train_one_epoch_gpu(dataset, learning_rate, loss_type, batch_size);
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

    sync_all_layers_to_device_gpu();
}