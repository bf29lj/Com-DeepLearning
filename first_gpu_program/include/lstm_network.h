#pragma once

#include "defect_dataset.h"
#include "mlp_network.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

struct SequenceSample {
    const std::vector<DefectSample> *source_samples = nullptr;
    std::size_t start_index = 0;
    std::size_t length = 0;
    std::vector<std::vector<float>> timesteps;
    uint8_t label = 0;
};

class LstmNetwork {
public:
    LstmNetwork(std::size_t input_size, std::size_t hidden_size);
    ~LstmNetwork();

    void set_execution_backend(ExecutionBackend backend) { execution_backend_ = backend; }
    ExecutionBackend execution_backend() const { return execution_backend_; }

    void set_optimizer_type(OptimizerType optimizer_type) { optimizer_type_ = optimizer_type; }
    void set_optimizer_hyperparameters(float momentum, float adam_beta1, float adam_beta2, float adam_epsilon);
    void set_class_weights(float positive_weight, float negative_weight);
    void set_focal_parameters(float gamma, float alpha);
    void save_to_file(const std::filesystem::path &path) const;
    void load_from_file(const std::filesystem::path &path);

    float predict_probability(const std::vector<std::vector<float>> &sequence) const;
    float predict_probability(const SequenceSample &sample) const;
    float evaluate_cost(const std::vector<SequenceSample> &dataset, LossType loss_type) const;
    float train_one_epoch(const std::vector<SequenceSample> &dataset,
                          float learning_rate,
                          LossType loss_type,
                          std::size_t batch_size,
                          float timeout_sec = 0.0f,
                          bool *timed_out = nullptr);

private:
    struct StepCache {
        std::vector<float> z;
        std::vector<float> f;
        std::vector<float> i;
        std::vector<float> g;
        std::vector<float> o;
        std::vector<float> c;
        std::vector<float> h;
        std::vector<float> c_prev;
    };

    struct ForwardCache {
        std::vector<StepCache> steps;
        float logit = 0.0f;
        float probability = 0.0f;
    };

    struct ParamState {
        std::vector<float> momentum;
        std::vector<float> adam_m;
        std::vector<float> adam_v;
    };

    struct Gradients {
        std::vector<float> dWf;
        std::vector<float> dWi;
        std::vector<float> dWg;
        std::vector<float> dWo;
        std::vector<float> dbf;
        std::vector<float> dbi;
        std::vector<float> dbg;
        std::vector<float> dbo;
        std::vector<float> dWy;
        float dby = 0.0f;
    };

    ForwardCache forward_with_cache_cpu(const std::vector<std::vector<float>> &sequence) const;
    ForwardCache forward_with_cache_cpu(const SequenceSample &sample) const;
    ForwardCache forward_with_cache_gpu(const SequenceSample &sample) const;
    ForwardCache forward_with_cache(const SequenceSample &sample) const;
    void ensure_gpu_runtime() const;
    void invalidate_gpu_parameter_cache();
    float compute_loss(float prediction, float target, LossType loss_type) const;
    float output_logit_gradient(float prediction, float target, LossType loss_type) const;
    void clear_gradients(Gradients &grads) const;
    void accumulate_gradients_from_cache(const ForwardCache &cache,
                                         float target,
                                         LossType loss_type,
                                         Gradients &grads) const;
    void apply_optimizer(std::vector<float> &params,
                         const std::vector<float> &grads,
                         ParamState &state,
                         float learning_rate,
                         float inv_batch,
                         std::uint64_t step);

    static float sigmoid(float x);

    std::size_t input_size_ = 0;
    std::size_t hidden_size_ = 0;
    std::size_t z_size_ = 0;

    std::vector<float> Wf_;
    std::vector<float> Wi_;
    std::vector<float> Wg_;
    std::vector<float> Wo_;
    std::vector<float> bf_;
    std::vector<float> bi_;
    std::vector<float> bg_;
    std::vector<float> bo_;
    std::vector<float> Wy_;
    float by_ = 0.0f;

    ParamState state_Wf_;
    ParamState state_Wi_;
    ParamState state_Wg_;
    ParamState state_Wo_;
    ParamState state_bf_;
    ParamState state_bi_;
    ParamState state_bg_;
    ParamState state_bo_;
    ParamState state_Wy_;
    float state_by_momentum_ = 0.0f;
    float state_by_adam_m_ = 0.0f;
    float state_by_adam_v_ = 0.0f;

    OptimizerType optimizer_type_ = OptimizerType::SGD;
    float momentum_ = 0.9f;
    float adam_beta1_ = 0.9f;
    float adam_beta2_ = 0.999f;
    float adam_epsilon_ = 1e-8f;
    std::uint64_t optimizer_step_ = 0;

    float positive_class_weight_ = 1.0f;
    float negative_class_weight_ = 1.0f;
    float focal_gamma_ = 2.0f;
    float focal_alpha_ = 0.25f;

    struct GpuRuntimeImpl;
    mutable std::unique_ptr<GpuRuntimeImpl> gpu_runtime_;
    ExecutionBackend execution_backend_ = ExecutionBackend::CPU;
};
