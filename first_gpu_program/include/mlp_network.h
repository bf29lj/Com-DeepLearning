#pragma once

#include "gpu_adapter.h"
#include "gpu_kernels.h"
#include "defect_dataset.h"

#include <cstddef>
#include <filesystem>
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

enum class ExecutionBackend {
    CPU,
    GPU,
};

enum class OptimizerType {
    SGD,
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

    // Sets execution backend used by dispatching APIs.
    void set_execution_backend(ExecutionBackend backend) { execution_backend_ = backend; }
    ExecutionBackend execution_backend() const { return execution_backend_; }

    // Sets optimizer behavior used by training backprop/update path.
    void set_optimizer_type(OptimizerType optimizer_type) { optimizer_type_ = optimizer_type; }
    OptimizerType optimizer_type() const { return optimizer_type_; }
    void set_enable_bce_sigmoid_shortcut(bool enable) { enable_bce_sigmoid_shortcut_ = enable; }
    bool enable_bce_sigmoid_shortcut() const { return enable_bce_sigmoid_shortcut_; }
    void set_class_weights(float positive_weight, float negative_weight);
    float positive_class_weight() const { return positive_class_weight_; }
    float negative_class_weight() const { return negative_class_weight_; }

    // Runs a single 1D forward pass using selected backend.
    std::vector<float> forward(const std::vector<float> &input);
    std::vector<float> forward_cpu(const std::vector<float> &input) const;
    std::vector<float> forward_gpu(const std::vector<float> &input);

    // Trains one epoch using selected backend and returns average epoch loss.
    float train_one_epoch(const ManufacturingDefectDataset &dataset,
                          float learning_rate,
                          LossType loss_type);
    float train_one_epoch_cpu(const ManufacturingDefectDataset &dataset,
                              float learning_rate,
                              LossType loss_type);
    float train_one_epoch_gpu(const ManufacturingDefectDataset &dataset,
                              float learning_rate,
                              LossType loss_type);

    // Evaluates average loss over a dataset using selected backend.
    float evaluate_cost(const ManufacturingDefectDataset &dataset,
                        LossType loss_type) const;
    float evaluate_cost_cpu(const ManufacturingDefectDataset &dataset,
                            LossType loss_type) const;
    float evaluate_cost_gpu(const ManufacturingDefectDataset &dataset,
                            LossType loss_type);

    // Returns the configured operation pipeline.
    const std::vector<OperationConfig> &operations() const { return operations_; }

    // Persists model structure and parameters to disk.
    void save_to_file(const std::filesystem::path &model_path) const;

    // Loads model parameters from disk and syncs to the selected backend.
    void load_from_file(const std::filesystem::path &model_path);

private:
    struct ForwardCache {
        std::vector<std::vector<float>> op_inputs;
        std::vector<std::vector<float>> op_outputs;
    };

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

    struct SgdOptimizer {
        static float output_gradient(float prediction,
                                     float target,
                                     LossType loss_type,
                                     bool use_bce_sigmoid_shortcut,
                                     float positive_weight,
                                     float negative_weight,
                                     float (*loss_derivative_wrt_prediction)(float, float, LossType, float, float));
        static void update_dense_layer_cpu(DenseLayer &layer,
                                           const std::vector<float> &input,
                                           const std::vector<float> &grad_output,
                                           std::vector<float> &grad_input,
                                           float learning_rate);
    };

    static GpuProgram create_kernels(GpuContext &ctx);
    static std::vector<OperationConfig> build_operations_from_layer_sizes(
        const std::vector<std::size_t> &layer_sizes);
    void initialize_layer(DenseLayer &layer);
    void sync_layer_to_device_gpu(DenseLayer &layer);
    void sync_all_layers_to_device_gpu();
    void sync_layer_to_host_gpu(DenseLayer &layer);
    void sync_all_layers_to_host_gpu();
    void run_dense_layer_gpu(DenseLayer &layer, GpuBuffer &input, GpuBuffer &output);
    void apply_activation_gpu(ActivationType type, GpuBuffer &values);
    std::vector<float> forward_internal_gpu(const std::vector<float> &input,
                                            ForwardCache *cache);
    ForwardCache forward_with_cache_cpu(const std::vector<float> &input) const;
    ForwardCache forward_with_cache_gpu(const std::vector<float> &input);
    ForwardCache forward_with_cache_for_backend(const std::vector<float> &input,
                                                ExecutionBackend backend) const;
    float train_one_epoch_internal(const ManufacturingDefectDataset &dataset,
                                   float learning_rate,
                                   LossType loss_type,
                                   ExecutionBackend backend);
    void backward_update_cpu(const ForwardCache &cache,
                             float target,
                             float learning_rate,
                             LossType loss_type);
    void backward_update_gpu(const ForwardCache &cache,
                             float target,
                             float learning_rate,
                             LossType loss_type);
    static float compute_loss(float prediction,
                              float target,
                              LossType loss_type,
                              float positive_weight,
                              float negative_weight);
    static float loss_derivative_wrt_prediction(float prediction,
                                                float target,
                                                LossType loss_type,
                                                float positive_weight,
                                                float negative_weight);
    static float activation_derivative(OperationType op_type, float pre_activation, float post_activation);
    bool should_use_bce_sigmoid_shortcut(LossType loss_type) const;
    static ActivationType operation_to_activation(OperationType op_type);

    GpuContext &context_;
    GpuProgram gpu_prog_;
    std::vector<OperationConfig> operations_;
    std::vector<DenseLayer> layers_;
    ExecutionBackend execution_backend_ = ExecutionBackend::GPU;
    OptimizerType optimizer_type_ = OptimizerType::SGD;
    bool enable_bce_sigmoid_shortcut_ = true;
    float positive_class_weight_ = 1.0f;
    float negative_class_weight_ = 1.0f;
};