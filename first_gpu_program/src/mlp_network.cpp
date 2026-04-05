#include "mlp_network.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

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

MlpNetwork::MlpNetwork(GpuContext &ctx, std::vector<std::size_t> layer_sizes)
    : MlpNetwork(ctx, build_operations_from_layer_sizes(layer_sizes)) {}

MlpNetwork::MlpNetwork(GpuContext &ctx, std::vector<OperationConfig> operations)
    : context_(ctx), gpu_prog_(create_kernels(ctx)) {
    if (operations.empty()) {
        throw std::invalid_argument("Operation pipeline cannot be empty");
    }

    operations_ = std::move(operations);
    for (const auto &op : operations_) {
        if (op.type == OperationType::Linear) {
            if (op.input_size == 0 || op.output_size == 0) {
                throw std::invalid_argument("Linear operation requires non-zero input_size and output_size");
            }
            layers_.emplace_back(op.input_size, op.output_size, ActivationType::Linear, context_);
            initialize_layer(layers_.back());
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

void MlpNetwork::initialize_layer(DenseLayer &layer) {
    std::mt19937 gen(42u + static_cast<unsigned int>(layer.input_size * 31 + layer.output_size));
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    for (float &w : layer.host_weights) {
        w = dist(gen) * 0.2f;
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

float MlpNetwork::compute_loss(float prediction, float target, LossType loss_type) {
    if (loss_type == LossType::MSE) {
        const float diff = prediction - target;
        return diff * diff;
    }

    const float epsilon = 1e-7f;
    const float p = std::clamp(prediction, epsilon, 1.0f - epsilon);
    return -(target * std::log(p) + (1.0f - target) * std::log(1.0f - p));
}

float MlpNetwork::loss_derivative_wrt_prediction(float prediction, float target, LossType loss_type) {
    if (loss_type == LossType::MSE) {
        return 2.0f * (prediction - target);
    }

    const float epsilon = 1e-7f;
    const float p = std::clamp(prediction, epsilon, 1.0f - epsilon);
    return -(target / p) + ((1.0f - target) / (1.0f - p));
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

    std::vector<float> gradient(network_output.size(), 0.0f);
    const bool use_sigmoid_bce_shortcut =
        (loss_type == LossType::BCE) &&
        (!operations_.empty()) &&
        (operations_.back().type == OperationType::Sigmoid);

    if (use_sigmoid_bce_shortcut) {
        gradient[0] = network_output[0] - target;
    } else {
        gradient[0] = loss_derivative_wrt_prediction(network_output[0], target, loss_type);
    }

    std::size_t reverse_linear_index = layers_.size();

    for (std::size_t op_index = operations_.size(); op_index-- > 0;) {
        const OperationType op_type = operations_[op_index].type;
        const std::vector<float> &op_input = cache.op_inputs[op_index];
        const std::vector<float> &op_output = cache.op_outputs[op_index];

        if (op_type == OperationType::Relu || op_type == OperationType::Sigmoid) {
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

        for (std::size_t out = 0; out < layer.output_size; ++out) {
            const float delta = gradient[out];
            layer.host_biases[out] -= learning_rate * delta;

            for (std::size_t in = 0; in < layer.input_size; ++in) {
                const std::size_t weight_index = out * layer.input_size + in;
                const float weight_before_update = layer.host_weights[weight_index];
                const float grad_w = delta * op_input[in];
                layer.host_weights[weight_index] -= learning_rate * grad_w;
                input_gradient[in] += weight_before_update * delta;
            }
        }

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
        total_loss += compute_loss(y_hat, y, loss_type);
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
        total_loss += compute_loss(y_hat, y, loss_type);
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
                                           ExecutionBackend backend)
{
    if (dataset.size() == 0) {
        throw std::invalid_argument("Dataset is empty");
    }
    if (learning_rate <= 0.0f) {
        throw std::invalid_argument("Learning rate must be positive");
    }

    float total_loss = 0.0f;

    for (const auto &sample : dataset.samples()) {
        const ForwardCache cache = forward_with_cache_for_backend(sample.features, backend);
        const std::vector<float> &current = cache.op_outputs.back();
        const float target = static_cast<float>(sample.label);
        if (current.empty()) {
            throw std::runtime_error("Network output is empty");
        }

        total_loss += compute_loss(current[0], target, loss_type);
        backward_update_cpu(cache, target, learning_rate, loss_type);

        if (backend == ExecutionBackend::GPU) {
            // Keep device parameters in sync so the next sample still uses GPU forward.
            sync_all_layers_to_device_gpu();
        }
    }

    if (backend == ExecutionBackend::CPU) {
        // Keep device parameters synchronized in case backend switches to GPU later.
        sync_all_layers_to_device_gpu();
    }

    return total_loss / static_cast<float>(dataset.size());
}

float MlpNetwork::train_one_epoch_cpu(const ManufacturingDefectDataset &dataset,
                                      float learning_rate,
                                      LossType loss_type)
{
    return train_one_epoch_internal(dataset, learning_rate, loss_type, ExecutionBackend::CPU);
}

float MlpNetwork::train_one_epoch_gpu(const ManufacturingDefectDataset &dataset,
                                      float learning_rate,
                                      LossType loss_type)
{
    return train_one_epoch_internal(dataset, learning_rate, loss_type, ExecutionBackend::GPU);
}

float MlpNetwork::train_one_epoch(const ManufacturingDefectDataset &dataset,
                                  float learning_rate,
                                  LossType loss_type)
{
    if (execution_backend_ == ExecutionBackend::CPU) {
        return train_one_epoch_cpu(dataset, learning_rate, loss_type);
    }
    return train_one_epoch_gpu(dataset, learning_rate, loss_type);
}