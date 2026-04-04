#pragma once

#include "gpu_adapter.h"
#include "gpu_kernels.h"
#include "defect_dataset.h"

#include <cstddef>
#include <string>
#include <vector>

enum class ActivationType {
    Linear,
    Relu,
    Sigmoid,
};

enum class LossType {
    BCE,
    MSE,
};

// 1D operation kinds in a configurable forward pipeline.
enum class OperationType {
    Linear,
    Relu,
    Sigmoid,
};

// Declarative operation config used to build a sequential 1D network.
struct OperationConfig {
    OperationType type;
    // Used only when type == Linear.
    std::size_t input_size = 0;
    // Used only when type == Linear.
    std::size_t output_size = 0;

    // Adds a Linear operation with explicit 1D dimensions.
    static OperationConfig linear(std::size_t input, std::size_t output) {
        return {OperationType::Linear, input, output};
    }

    // Adds a ReLU activation operation.
    static OperationConfig relu() {
        return {OperationType::Relu, 0, 0};
    }

    // Adds a Sigmoid activation operation.
    static OperationConfig sigmoid() {
        return {OperationType::Sigmoid, 0, 0};
    }
};

// Configurable 1D MLP forward network.
class MlpNetwork {
public:
    // Compatibility constructor: converts layer sizes into
    // Linear + activation operation sequence.
    MlpNetwork(GpuContext &ctx, std::vector<std::size_t> layer_sizes);

    // Preferred constructor: explicit operation pipeline definition.
    MlpNetwork(GpuContext &ctx, std::vector<OperationConfig> operations);

    // Runs a single 1D forward pass.
    std::vector<float> forward(const std::vector<float> &input);

    // Trains one epoch with sample-wise SGD and returns average epoch loss.
    float train_one_epoch(const ManufacturingDefectDataset &dataset,
                          float learning_rate,
                          LossType loss_type);

    // Evaluates average loss over a dataset using the selected loss function.
    float evaluate_cost(const ManufacturingDefectDataset &dataset,
                        LossType loss_type) const;

    // Returns the configured operation pipeline.
    const std::vector<OperationConfig> &operations() const { return operations_; }

private:
    struct DenseLayer {
        std::size_t input_size;
        std::size_t output_size;
        ActivationType activation;
        std::vector<float> host_weights;
        std::vector<float> host_biases;
        GpuBuffer weights;
        GpuBuffer biases;

        DenseLayer(std::size_t input, std::size_t output, ActivationType activation_type,
                   GpuContext &ctx);
    };

    static GpuProgram create_kernels(GpuContext &ctx);
    static std::vector<OperationConfig> build_operations_from_layer_sizes(
        const std::vector<std::size_t> &layer_sizes);
    void initialize_layer(DenseLayer &layer);
    void sync_layer_to_device(DenseLayer &layer);
    void run_dense_layer(DenseLayer &layer, GpuBuffer &input, GpuBuffer &output);
    void apply_activation(ActivationType type, GpuBuffer &values);
    std::vector<float> forward_host(const std::vector<float> &input) const;
    static float compute_loss(float prediction, float target, LossType loss_type);
    static float loss_derivative_wrt_prediction(float prediction, float target, LossType loss_type);
    static float activation_derivative(OperationType op_type, float pre_activation, float post_activation);
    static ActivationType operation_to_activation(OperationType op_type);

    GpuContext &context_;
    GpuProgram gpu_prog_;
    std::vector<OperationConfig> operations_;
    std::vector<DenseLayer> layers_;
};