#pragma once

#include "defect_dataset.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

enum class ActivationType {
    Linear,
    Relu,
    Sigmoid,
    Tanh,
    LeakyRelu,
    Gelu,
};

enum class LossType {
    BCE,
    MSE,
    Focal,
};

enum class ExecutionBackend {
    CPU,
    GPU,
};

enum class OptimizerType {
    SGD,
    Momentum,
    Adam,
};

// 1D operation kinds in a configurable forward pipeline.
enum class OperationType {
    Linear,
    Relu,
    Sigmoid,
    Tanh,
    LeakyRelu,
    Gelu,
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

    // Adds a Tanh activation operation.
    static OperationConfig tanh() {
        return {OperationType::Tanh, 0, 0};
    }

    // Adds a LeakyReLU activation operation.
    static OperationConfig leaky_relu() {
        return {OperationType::LeakyRelu, 0, 0};
    }

    // Adds a GeLU activation operation.
    static OperationConfig gelu() {
        return {OperationType::Gelu, 0, 0};
    }
};

// Configurable 1D MLP forward network.
class MlpNetwork {
public:
    // Compatibility constructor: converts layer sizes into
    // Linear + activation operation sequence.
    MlpNetwork(std::vector<std::size_t> layer_sizes);

    // Preferred constructor: explicit operation pipeline definition.
    MlpNetwork(std::vector<OperationConfig> operations);
    ~MlpNetwork();

    // Stores the requested execution backend for compatibility.
    void set_execution_backend(ExecutionBackend backend) { execution_backend_ = backend; }
    ExecutionBackend execution_backend() const { return execution_backend_; }

    // Sets optimizer behavior used by training backprop/update path.
    void set_optimizer_type(OptimizerType optimizer_type) { optimizer_type_ = optimizer_type; }
    OptimizerType optimizer_type() const { return optimizer_type_; }
    void set_optimizer_hyperparameters(float momentum, float adam_beta1, float adam_beta2, float adam_epsilon);
    void set_enable_bce_sigmoid_shortcut(bool enable) { enable_bce_sigmoid_shortcut_ = enable; }
    bool enable_bce_sigmoid_shortcut() const { return enable_bce_sigmoid_shortcut_; }
    void set_class_weights(float positive_weight, float negative_weight);
    void set_focal_parameters(float gamma, float alpha);
    float positive_class_weight() const { return positive_class_weight_; }
    float negative_class_weight() const { return negative_class_weight_; }

    // Runs a single 1D forward pass on the CPU.
    std::vector<float> forward(const std::vector<float> &input);
    std::vector<float> forward_cpu(const std::vector<float> &input) const;
    std::vector<float> forward_gpu(const std::vector<float> &input) const;

    // Trains one epoch on the CPU and returns average epoch loss.
    float train_one_epoch(const ManufacturingDefectDataset &dataset,
                          float learning_rate,
                          LossType loss_type,
                          std::size_t batch_size = 1);
    float train_one_epoch_cpu(const ManufacturingDefectDataset &dataset,
                              float learning_rate,
                              LossType loss_type,
                              std::size_t batch_size = 1);
    float train_one_epoch_gpu(const ManufacturingDefectDataset &dataset,
                              float learning_rate,
                              LossType loss_type,
                              std::size_t batch_size = 1);

    // Evaluates average loss over a dataset on the CPU.
    float evaluate_cost(const ManufacturingDefectDataset &dataset,
                        LossType loss_type) const;
    float evaluate_cost_cpu(const ManufacturingDefectDataset &dataset,
                            LossType loss_type) const;
    float evaluate_cost_gpu(const ManufacturingDefectDataset &dataset,
                            LossType loss_type) const;

    // Returns the configured operation pipeline.
    const std::vector<OperationConfig> &operations() const { return operations_; }

    // Persists model structure and parameters to disk.
    void save_to_file(const std::filesystem::path &model_path) const;

    // Reads only the operation pipeline from a saved model file.
    static std::vector<OperationConfig> load_operations_from_file(const std::filesystem::path &model_path);

    // Loads model parameters from disk.
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

        DenseLayer(std::size_t input, std::size_t output, ActivationType activation_type);
    };

    struct OptimizerState {
        std::vector<float> momentum_w;
        std::vector<float> momentum_b;
        std::vector<float> adam_m_w;
        std::vector<float> adam_m_b;
        std::vector<float> adam_v_w;
        std::vector<float> adam_v_b;
    };

    struct SgdOptimizer {
        static float output_gradient(float prediction,
                                     float target,
                                     LossType loss_type,
                                     bool use_bce_sigmoid_shortcut,
                                     float positive_weight,
                                     float negative_weight,
                                     float focal_gamma,
                                     float focal_alpha,
                                     float (*loss_derivative_wrt_prediction)(float, float, LossType, float, float, float, float));
        static void update_dense_layer(DenseLayer &layer,
                                       const std::vector<float> &input,
                                       const std::vector<float> &grad_output,
                                       std::vector<float> &grad_input,
                                       float learning_rate);
    };

    static std::vector<OperationConfig> build_operations_from_layer_sizes(
        const std::vector<std::size_t> &layer_sizes);
    void initialize_layer(DenseLayer &layer, OperationType next_op_type);
    ForwardCache forward_with_cache_cpu(const std::vector<float> &input) const;
    ForwardCache forward_with_cache_gpu(const std::vector<float> &input) const;
    float train_one_epoch_internal(const ManufacturingDefectDataset &dataset,
                                   float learning_rate,
                                   LossType loss_type,
                                   std::size_t batch_size);
    float train_one_epoch_internal_gpu(const ManufacturingDefectDataset &dataset,
                                       float learning_rate,
                                       LossType loss_type,
                                       std::size_t batch_size);
    void backward_update_cpu(const ForwardCache &cache,
                             float target,
                             float learning_rate,
                             LossType loss_type);
    void backward_update_gpu(const ForwardCache &cache,
                             float target,
                             float learning_rate,
                             LossType loss_type);
    void ensure_gpu_runtime() const;
    void sync_gpu_layers_to_host();
    void invalidate_gpu_parameter_cache();
    static float compute_loss(float prediction,
                              float target,
                              LossType loss_type,
                              float positive_weight,
                              float negative_weight,
                              float focal_gamma,
                              float focal_alpha);
    static float loss_derivative_wrt_prediction(float prediction,
                                                float target,
                                                LossType loss_type,
                                                float positive_weight,
                                                float negative_weight,
                                                float focal_gamma,
                                                float focal_alpha);
    static float activation_derivative(OperationType op_type, float pre_activation, float post_activation);
    bool should_use_bce_sigmoid_shortcut(LossType loss_type) const;
    static ActivationType operation_to_activation(OperationType op_type);

    std::vector<OperationConfig> operations_;
    std::vector<DenseLayer> layers_;
    std::vector<OptimizerState> optimizer_states_;
    struct GpuRuntimeImpl;
    mutable std::unique_ptr<GpuRuntimeImpl> gpu_runtime_;
    ExecutionBackend execution_backend_ = ExecutionBackend::CPU;
    OptimizerType optimizer_type_ = OptimizerType::SGD;
    float momentum_ = 0.9f;
    float adam_beta1_ = 0.9f;
    float adam_beta2_ = 0.999f;
    float adam_epsilon_ = 1e-8f;
    std::uint64_t optimizer_step_ = 0;
    bool enable_bce_sigmoid_shortcut_ = true;
    float positive_class_weight_ = 1.0f;
    float negative_class_weight_ = 1.0f;
    float focal_gamma_ = 2.0f;
    float focal_alpha_ = 0.25f;
};