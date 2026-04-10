#include "lstm_network.h"

#include "gpu_kernels.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>

namespace {

float clamp_probability(float p) {
    return std::clamp(p, 1e-7f, 1.0f - 1e-7f);
}

float pow_uint(float base, std::uint64_t exp) {
    float result = 1.0f;
    for (std::uint64_t i = 0; i < exp; ++i) {
        result *= base;
    }
    return result;
}

template <typename T>
void write_binary(std::ofstream &out, const T &value) {
    out.write(reinterpret_cast<const char *>(&value), sizeof(T));
}

template <typename T>
void read_binary(std::ifstream &in, T &value) {
    in.read(reinterpret_cast<char *>(&value), sizeof(T));
}

void write_float_vector(std::ofstream &out, const std::vector<float> &vec) {
    std::uint64_t size = static_cast<std::uint64_t>(vec.size());
    write_binary(out, size);
    if (!vec.empty()) {
        out.write(reinterpret_cast<const char *>(vec.data()), static_cast<std::streamsize>(vec.size() * sizeof(float)));
    }
}

void read_float_vector(std::ifstream &in, std::vector<float> &vec) {
    std::uint64_t size = 0;
    read_binary(in, size);
    vec.resize(static_cast<std::size_t>(size));
    if (!vec.empty()) {
        in.read(reinterpret_cast<char *>(vec.data()), static_cast<std::streamsize>(vec.size() * sizeof(float)));
    }
}

void require_stream_ok(bool ok, const char *message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

void enqueue_fill_float_lstm(GpuContext &context,
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

void enqueue_lstm_gate_update(GpuContext &context,
                              const GpuProgram &program,
                              OptimizerType optimizer,
                              GpuBuffer &weights,
                              GpuBuffer &biases,
                              GpuBuffer &grad_weights,
                              GpuBuffer &grad_biases,
                              GpuBuffer &vel_weights,
                              GpuBuffer &vel_biases,
                              GpuBuffer &adam_m_weights,
                              GpuBuffer &adam_v_weights,
                              GpuBuffer &adam_m_biases,
                              GpuBuffer &adam_v_biases,
                              float learning_rate,
                              float momentum,
                              float adam_beta1,
                              float adam_beta2,
                              float adam_epsilon,
                              float weight_decay,
                              float bias_corr1,
                              float bias_corr2,
                              float inv_batch,
                              std::size_t input_size,
                              std::size_t output_size)
{
    if (optimizer == OptimizerType::SGD) {
        GpuKernel update_kernel(program, "dense_sgd_update", context);
        update_kernel.set_args({
            KernelArg::buffer(weights.get_buffer()),
            KernelArg::buffer(biases.get_buffer()),
            KernelArg::buffer(grad_weights.get_buffer()),
            KernelArg::buffer(grad_biases.get_buffer()),
            KernelArg::scalar_float(learning_rate),
            KernelArg::scalar_float(inv_batch),
            KernelArg::scalar_uint(static_cast<uint32_t>(input_size)),
        });
        update_kernel.enqueue_1d(output_size);
        return;
    }

    if (optimizer == OptimizerType::Momentum) {
        GpuKernel update_kernel(program, "dense_momentum_update", context);
        update_kernel.set_args({
            KernelArg::buffer(weights.get_buffer()),
            KernelArg::buffer(biases.get_buffer()),
            KernelArg::buffer(vel_weights.get_buffer()),
            KernelArg::buffer(vel_biases.get_buffer()),
            KernelArg::buffer(grad_weights.get_buffer()),
            KernelArg::buffer(grad_biases.get_buffer()),
            KernelArg::scalar_float(learning_rate),
            KernelArg::scalar_float(momentum),
            KernelArg::scalar_float(inv_batch),
            KernelArg::scalar_uint(static_cast<uint32_t>(input_size)),
        });
        update_kernel.enqueue_1d(output_size);
        return;
    }

    GpuKernel update_kernel(program, "dense_adam_update", context);
    update_kernel.set_args({
        KernelArg::buffer(weights.get_buffer()),
        KernelArg::buffer(biases.get_buffer()),
        KernelArg::buffer(adam_m_weights.get_buffer()),
        KernelArg::buffer(adam_v_weights.get_buffer()),
        KernelArg::buffer(adam_m_biases.get_buffer()),
        KernelArg::buffer(adam_v_biases.get_buffer()),
        KernelArg::buffer(grad_weights.get_buffer()),
        KernelArg::buffer(grad_biases.get_buffer()),
        KernelArg::scalar_float(learning_rate),
        KernelArg::scalar_float(adam_beta1),
        KernelArg::scalar_float(adam_beta2),
        KernelArg::scalar_float(adam_epsilon),
        KernelArg::scalar_float(weight_decay),
        KernelArg::scalar_float(bias_corr1),
        KernelArg::scalar_float(bias_corr2),
        KernelArg::scalar_float(inv_batch),
        KernelArg::scalar_uint(static_cast<uint32_t>(input_size)),
    });
    update_kernel.enqueue_1d(output_size);
}

} // namespace

struct LstmNetwork::GpuRuntimeImpl {
        GpuContext context;
        GpuProgram program;

        std::unique_ptr<GpuBuffer> Wf;
        std::unique_ptr<GpuBuffer> Wi;
        std::unique_ptr<GpuBuffer> Wg;
        std::unique_ptr<GpuBuffer> Wo;
        std::unique_ptr<GpuBuffer> bf;
        std::unique_ptr<GpuBuffer> bi;
        std::unique_ptr<GpuBuffer> bg;
        std::unique_ptr<GpuBuffer> bo;
        std::unique_ptr<GpuBuffer> Wy;

        std::unique_ptr<GpuBuffer> gradWf;
        std::unique_ptr<GpuBuffer> gradWi;
        std::unique_ptr<GpuBuffer> gradWg;
        std::unique_ptr<GpuBuffer> gradWo;
        std::unique_ptr<GpuBuffer> gradbf;
        std::unique_ptr<GpuBuffer> gradbi;
        std::unique_ptr<GpuBuffer> gradbg;
        std::unique_ptr<GpuBuffer> gradbo;

        std::unique_ptr<GpuBuffer> velWf;
        std::unique_ptr<GpuBuffer> velWi;
        std::unique_ptr<GpuBuffer> velWg;
        std::unique_ptr<GpuBuffer> velWo;
        std::unique_ptr<GpuBuffer> velbf;
        std::unique_ptr<GpuBuffer> velbi;
        std::unique_ptr<GpuBuffer> velbg;
        std::unique_ptr<GpuBuffer> velbo;

        std::unique_ptr<GpuBuffer> adamMWf;
        std::unique_ptr<GpuBuffer> adamMWi;
        std::unique_ptr<GpuBuffer> adamMWg;
        std::unique_ptr<GpuBuffer> adamMWo;
        std::unique_ptr<GpuBuffer> adamMbf;
        std::unique_ptr<GpuBuffer> adamMbi;
        std::unique_ptr<GpuBuffer> adamMbg;
        std::unique_ptr<GpuBuffer> adamMbo;

        std::unique_ptr<GpuBuffer> adamVWf;
        std::unique_ptr<GpuBuffer> adamVWi;
        std::unique_ptr<GpuBuffer> adamVWg;
        std::unique_ptr<GpuBuffer> adamVWo;
        std::unique_ptr<GpuBuffer> adamVbf;
        std::unique_ptr<GpuBuffer> adamVbi;
        std::unique_ptr<GpuBuffer> adamVbg;
        std::unique_ptr<GpuBuffer> adamVbo;

        bool parameters_synced = false;

        GpuRuntimeImpl()
                : context(GpuContext::create_default()),
                    program(gpu::create_gpu_program(context)) {}
};

LstmNetwork::LstmNetwork(std::size_t input_size, std::size_t hidden_size)
    : input_size_(input_size),
      hidden_size_(hidden_size),
      z_size_(input_size + hidden_size),
      Wf_(hidden_size * z_size_),
      Wi_(hidden_size * z_size_),
      Wg_(hidden_size * z_size_),
      Wo_(hidden_size * z_size_),
      bf_(hidden_size, 0.0f),
      bi_(hidden_size, 0.0f),
      bg_(hidden_size, 0.0f),
      bo_(hidden_size, 0.0f),
      Wy_(hidden_size, 0.0f) {
    if (input_size_ == 0 || hidden_size_ == 0) {
        throw std::invalid_argument("LSTM input_size and hidden_size must be positive");
    }

    const float stddev = std::sqrt(1.0f / static_cast<float>(z_size_));
    std::mt19937 gen(1337u);
    std::normal_distribution<float> dist(0.0f, stddev);

    for (float &w : Wf_) w = dist(gen);
    for (float &w : Wi_) w = dist(gen);
    for (float &w : Wg_) w = dist(gen);
    for (float &w : Wo_) w = dist(gen);
    for (float &w : Wy_) w = dist(gen);

    auto init_state = [](auto &state, std::size_t size) {
        state.momentum.assign(size, 0.0f);
        state.adam_m.assign(size, 0.0f);
        state.adam_v.assign(size, 0.0f);
    };

    init_state(state_Wf_, Wf_.size());
    init_state(state_Wi_, Wi_.size());
    init_state(state_Wg_, Wg_.size());
    init_state(state_Wo_, Wo_.size());
    init_state(state_bf_, bf_.size());
    init_state(state_bi_, bi_.size());
    init_state(state_bg_, bg_.size());
    init_state(state_bo_, bo_.size());
    init_state(state_Wy_, Wy_.size());
}

LstmNetwork::~LstmNetwork() = default;

void LstmNetwork::ensure_gpu_runtime() const {
    if (gpu_runtime_ == nullptr) {
        gpu_runtime_ = std::make_unique<GpuRuntimeImpl>();
    }

    if (!gpu_runtime_->parameters_synced) {
        auto zero_like = [&](std::size_t count) {
            std::vector<float> zeros(count, 0.0f);
            return GpuBuffer::from_host(gpu_runtime_->context, zeros);
        };

        gpu_runtime_->Wf = std::make_unique<GpuBuffer>(GpuBuffer::from_host(gpu_runtime_->context, Wf_));
        gpu_runtime_->Wi = std::make_unique<GpuBuffer>(GpuBuffer::from_host(gpu_runtime_->context, Wi_));
        gpu_runtime_->Wg = std::make_unique<GpuBuffer>(GpuBuffer::from_host(gpu_runtime_->context, Wg_));
        gpu_runtime_->Wo = std::make_unique<GpuBuffer>(GpuBuffer::from_host(gpu_runtime_->context, Wo_));
        gpu_runtime_->bf = std::make_unique<GpuBuffer>(GpuBuffer::from_host(gpu_runtime_->context, bf_));
        gpu_runtime_->bi = std::make_unique<GpuBuffer>(GpuBuffer::from_host(gpu_runtime_->context, bi_));
        gpu_runtime_->bg = std::make_unique<GpuBuffer>(GpuBuffer::from_host(gpu_runtime_->context, bg_));
        gpu_runtime_->bo = std::make_unique<GpuBuffer>(GpuBuffer::from_host(gpu_runtime_->context, bo_));
        gpu_runtime_->Wy = std::make_unique<GpuBuffer>(GpuBuffer::from_host(gpu_runtime_->context, Wy_));

        gpu_runtime_->gradWf = std::make_unique<GpuBuffer>(zero_like(Wf_.size()));
        gpu_runtime_->gradWi = std::make_unique<GpuBuffer>(zero_like(Wi_.size()));
        gpu_runtime_->gradWg = std::make_unique<GpuBuffer>(zero_like(Wg_.size()));
        gpu_runtime_->gradWo = std::make_unique<GpuBuffer>(zero_like(Wo_.size()));
        gpu_runtime_->gradbf = std::make_unique<GpuBuffer>(zero_like(bf_.size()));
        gpu_runtime_->gradbi = std::make_unique<GpuBuffer>(zero_like(bi_.size()));
        gpu_runtime_->gradbg = std::make_unique<GpuBuffer>(zero_like(bg_.size()));
        gpu_runtime_->gradbo = std::make_unique<GpuBuffer>(zero_like(bo_.size()));

        gpu_runtime_->velWf = std::make_unique<GpuBuffer>(zero_like(Wf_.size()));
        gpu_runtime_->velWi = std::make_unique<GpuBuffer>(zero_like(Wi_.size()));
        gpu_runtime_->velWg = std::make_unique<GpuBuffer>(zero_like(Wg_.size()));
        gpu_runtime_->velWo = std::make_unique<GpuBuffer>(zero_like(Wo_.size()));
        gpu_runtime_->velbf = std::make_unique<GpuBuffer>(zero_like(bf_.size()));
        gpu_runtime_->velbi = std::make_unique<GpuBuffer>(zero_like(bi_.size()));
        gpu_runtime_->velbg = std::make_unique<GpuBuffer>(zero_like(bg_.size()));
        gpu_runtime_->velbo = std::make_unique<GpuBuffer>(zero_like(bo_.size()));

        gpu_runtime_->adamMWf = std::make_unique<GpuBuffer>(zero_like(Wf_.size()));
        gpu_runtime_->adamMWi = std::make_unique<GpuBuffer>(zero_like(Wi_.size()));
        gpu_runtime_->adamMWg = std::make_unique<GpuBuffer>(zero_like(Wg_.size()));
        gpu_runtime_->adamMWo = std::make_unique<GpuBuffer>(zero_like(Wo_.size()));
        gpu_runtime_->adamMbf = std::make_unique<GpuBuffer>(zero_like(bf_.size()));
        gpu_runtime_->adamMbi = std::make_unique<GpuBuffer>(zero_like(bi_.size()));
        gpu_runtime_->adamMbg = std::make_unique<GpuBuffer>(zero_like(bg_.size()));
        gpu_runtime_->adamMbo = std::make_unique<GpuBuffer>(zero_like(bo_.size()));

        gpu_runtime_->adamVWf = std::make_unique<GpuBuffer>(zero_like(Wf_.size()));
        gpu_runtime_->adamVWi = std::make_unique<GpuBuffer>(zero_like(Wi_.size()));
        gpu_runtime_->adamVWg = std::make_unique<GpuBuffer>(zero_like(Wg_.size()));
        gpu_runtime_->adamVWo = std::make_unique<GpuBuffer>(zero_like(Wo_.size()));
        gpu_runtime_->adamVbf = std::make_unique<GpuBuffer>(zero_like(bf_.size()));
        gpu_runtime_->adamVbi = std::make_unique<GpuBuffer>(zero_like(bi_.size()));
        gpu_runtime_->adamVbg = std::make_unique<GpuBuffer>(zero_like(bg_.size()));
        gpu_runtime_->adamVbo = std::make_unique<GpuBuffer>(zero_like(bo_.size()));

        gpu_runtime_->parameters_synced = true;
    }
}

void LstmNetwork::invalidate_gpu_parameter_cache() {
    if (gpu_runtime_ != nullptr) {
        gpu_runtime_->parameters_synced = false;
    }
}

void LstmNetwork::set_optimizer_hyperparameters(float momentum,
                                                float adam_beta1,
                                                float adam_beta2,
                                                float adam_epsilon) {
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

void LstmNetwork::set_weight_decay(float weight_decay) {
    if (!std::isfinite(weight_decay) || weight_decay < 0.0f) {
        throw std::invalid_argument("Weight decay must be a non-negative finite number");
    }
    weight_decay_ = weight_decay;
}

void LstmNetwork::set_class_weights(float positive_weight, float negative_weight) {
    if (positive_weight <= 0.0f || negative_weight <= 0.0f || !std::isfinite(positive_weight) ||
        !std::isfinite(negative_weight)) {
        throw std::invalid_argument("Class weights must be positive finite numbers");
    }
    positive_class_weight_ = positive_weight;
    negative_class_weight_ = negative_weight;
}

void LstmNetwork::set_focal_parameters(float gamma, float alpha) {
    if (gamma < 0.0f || !std::isfinite(gamma)) {
        throw std::invalid_argument("Focal gamma must be non-negative and finite");
    }
    if (alpha < 0.0f || alpha > 1.0f || !std::isfinite(alpha)) {
        throw std::invalid_argument("Focal alpha must be within [0, 1]");
    }
    focal_gamma_ = gamma;
    focal_alpha_ = alpha;
}

void LstmNetwork::save_to_file(const std::filesystem::path &path) const {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    require_stream_ok(out.is_open(), "Failed to open LSTM model file for write");

    const char magic[8] = {'L', 'S', 'T', 'M', 'B', 'I', 'N', '1'};
    out.write(magic, sizeof(magic));

    write_binary(out, static_cast<std::uint64_t>(input_size_));
    write_binary(out, static_cast<std::uint64_t>(hidden_size_));

    write_binary(out, by_);
    write_binary(out, momentum_);
    write_binary(out, adam_beta1_);
    write_binary(out, adam_beta2_);
    write_binary(out, adam_epsilon_);
    write_binary(out, positive_class_weight_);
    write_binary(out, negative_class_weight_);

    const std::uint32_t optimizer_type = static_cast<std::uint32_t>(optimizer_type_);
    write_binary(out, optimizer_type);

    write_float_vector(out, Wf_);
    write_float_vector(out, Wi_);
    write_float_vector(out, Wg_);
    write_float_vector(out, Wo_);
    write_float_vector(out, bf_);
    write_float_vector(out, bi_);
    write_float_vector(out, bg_);
    write_float_vector(out, bo_);
    write_float_vector(out, Wy_);

    require_stream_ok(out.good(), "Failed while writing LSTM model file");
}

void LstmNetwork::load_from_file(const std::filesystem::path &path) {
    std::ifstream in(path, std::ios::binary);
    require_stream_ok(in.is_open(), "Failed to open LSTM model file for read");

    char magic[8] = {};
    in.read(magic, sizeof(magic));
    const char expected[8] = {'L', 'S', 'T', 'M', 'B', 'I', 'N', '1'};
    if (std::memcmp(magic, expected, sizeof(magic)) != 0) {
        throw std::runtime_error("Invalid LSTM model file magic");
    }

    std::uint64_t input_size = 0;
    std::uint64_t hidden_size = 0;
    read_binary(in, input_size);
    read_binary(in, hidden_size);
    if (static_cast<std::size_t>(input_size) != input_size_ ||
        static_cast<std::size_t>(hidden_size) != hidden_size_) {
        throw std::runtime_error("LSTM model shape mismatch with current network");
    }

    read_binary(in, by_);
    read_binary(in, momentum_);
    read_binary(in, adam_beta1_);
    read_binary(in, adam_beta2_);
    read_binary(in, adam_epsilon_);
    read_binary(in, positive_class_weight_);
    read_binary(in, negative_class_weight_);

    std::uint32_t optimizer_type = 0;
    read_binary(in, optimizer_type);
    optimizer_type_ = static_cast<OptimizerType>(optimizer_type);

    read_float_vector(in, Wf_);
    read_float_vector(in, Wi_);
    read_float_vector(in, Wg_);
    read_float_vector(in, Wo_);
    read_float_vector(in, bf_);
    read_float_vector(in, bi_);
    read_float_vector(in, bg_);
    read_float_vector(in, bo_);
    read_float_vector(in, Wy_);

    require_stream_ok(in.good(), "Failed while reading LSTM model file");
    if (Wf_.size() != hidden_size_ * z_size_ || Wi_.size() != hidden_size_ * z_size_ ||
        Wg_.size() != hidden_size_ * z_size_ || Wo_.size() != hidden_size_ * z_size_ ||
        bf_.size() != hidden_size_ || bi_.size() != hidden_size_ ||
        bg_.size() != hidden_size_ || bo_.size() != hidden_size_ || Wy_.size() != hidden_size_) {
        throw std::runtime_error("Corrupted LSTM model parameter sizes");
    }

    optimizer_step_ = 0;
    std::fill(state_Wf_.momentum.begin(), state_Wf_.momentum.end(), 0.0f);
    std::fill(state_Wi_.momentum.begin(), state_Wi_.momentum.end(), 0.0f);
    std::fill(state_Wg_.momentum.begin(), state_Wg_.momentum.end(), 0.0f);
    std::fill(state_Wo_.momentum.begin(), state_Wo_.momentum.end(), 0.0f);
    std::fill(state_bf_.momentum.begin(), state_bf_.momentum.end(), 0.0f);
    std::fill(state_bi_.momentum.begin(), state_bi_.momentum.end(), 0.0f);
    std::fill(state_bg_.momentum.begin(), state_bg_.momentum.end(), 0.0f);
    std::fill(state_bo_.momentum.begin(), state_bo_.momentum.end(), 0.0f);
    std::fill(state_Wy_.momentum.begin(), state_Wy_.momentum.end(), 0.0f);
    std::fill(state_Wf_.adam_m.begin(), state_Wf_.adam_m.end(), 0.0f);
    std::fill(state_Wi_.adam_m.begin(), state_Wi_.adam_m.end(), 0.0f);
    std::fill(state_Wg_.adam_m.begin(), state_Wg_.adam_m.end(), 0.0f);
    std::fill(state_Wo_.adam_m.begin(), state_Wo_.adam_m.end(), 0.0f);
    std::fill(state_bf_.adam_m.begin(), state_bf_.adam_m.end(), 0.0f);
    std::fill(state_bi_.adam_m.begin(), state_bi_.adam_m.end(), 0.0f);
    std::fill(state_bg_.adam_m.begin(), state_bg_.adam_m.end(), 0.0f);
    std::fill(state_bo_.adam_m.begin(), state_bo_.adam_m.end(), 0.0f);
    std::fill(state_Wy_.adam_m.begin(), state_Wy_.adam_m.end(), 0.0f);
    std::fill(state_Wf_.adam_v.begin(), state_Wf_.adam_v.end(), 0.0f);
    std::fill(state_Wi_.adam_v.begin(), state_Wi_.adam_v.end(), 0.0f);
    std::fill(state_Wg_.adam_v.begin(), state_Wg_.adam_v.end(), 0.0f);
    std::fill(state_Wo_.adam_v.begin(), state_Wo_.adam_v.end(), 0.0f);
    std::fill(state_bf_.adam_v.begin(), state_bf_.adam_v.end(), 0.0f);
    std::fill(state_bi_.adam_v.begin(), state_bi_.adam_v.end(), 0.0f);
    std::fill(state_bg_.adam_v.begin(), state_bg_.adam_v.end(), 0.0f);
    std::fill(state_bo_.adam_v.begin(), state_bo_.adam_v.end(), 0.0f);
    std::fill(state_Wy_.adam_v.begin(), state_Wy_.adam_v.end(), 0.0f);
    state_by_momentum_ = 0.0f;
    state_by_adam_m_ = 0.0f;
    state_by_adam_v_ = 0.0f;
    invalidate_gpu_parameter_cache();
}

float LstmNetwork::sigmoid(float x) {
    const float clamped = std::clamp(x, -40.0f, 40.0f);
    return 1.0f / (1.0f + std::exp(-clamped));
}

LstmNetwork::ForwardCache LstmNetwork::forward_with_cache_cpu(const SequenceSample &sample) const {
    std::size_t seq_len = sample.timesteps.size();
    bool use_source_view = false;
    if (sample.source_samples != nullptr && sample.length > 0) {
        use_source_view = true;
        seq_len = sample.length;
        if (sample.start_index + sample.length > sample.source_samples->size()) {
            throw std::invalid_argument("LSTM sequence sample window is out of range");
        }
    }

    if (seq_len == 0) {
        throw std::invalid_argument("LSTM sequence cannot be empty");
    }

    ForwardCache cache;
    cache.steps.reserve(seq_len);

    std::vector<float> h_prev(hidden_size_, 0.0f);
    std::vector<float> c_prev(hidden_size_, 0.0f);

    for (std::size_t t = 0; t < seq_len; ++t) {
        const std::vector<float> &x_t = use_source_view
            ? (*sample.source_samples)[sample.start_index + t].features
            : sample.timesteps[t];

        if (x_t.size() != input_size_) {
            throw std::invalid_argument("LSTM sequence timestep feature size mismatch");
        }

        StepCache step;
        step.z.assign(z_size_, 0.0f);
        step.f.assign(hidden_size_, 0.0f);
        step.i.assign(hidden_size_, 0.0f);
        step.g.assign(hidden_size_, 0.0f);
        step.o.assign(hidden_size_, 0.0f);
        step.c.assign(hidden_size_, 0.0f);
        step.h.assign(hidden_size_, 0.0f);
        step.c_prev = c_prev;

        for (std::size_t k = 0; k < input_size_; ++k) {
            step.z[k] = x_t[k];
        }
        for (std::size_t k = 0; k < hidden_size_; ++k) {
            step.z[input_size_ + k] = h_prev[k];
        }

        for (std::size_t r = 0; r < hidden_size_; ++r) {
            float af = bf_[r];
            float ai = bi_[r];
            float ag = bg_[r];
            float ao = bo_[r];
            for (std::size_t k = 0; k < z_size_; ++k) {
                const std::size_t idx = r * z_size_ + k;
                const float z = step.z[k];
                af += Wf_[idx] * z;
                ai += Wi_[idx] * z;
                ag += Wg_[idx] * z;
                ao += Wo_[idx] * z;
            }

            step.f[r] = sigmoid(af);
            step.i[r] = sigmoid(ai);
            step.g[r] = std::tanh(ag);
            step.o[r] = sigmoid(ao);
            step.c[r] = step.f[r] * c_prev[r] + step.i[r] * step.g[r];
            step.h[r] = step.o[r] * std::tanh(step.c[r]);
        }

        h_prev = step.h;
        c_prev = step.c;
        cache.steps.push_back(std::move(step));
    }

    const StepCache &last = cache.steps.back();
    cache.logit = by_;
    for (std::size_t r = 0; r < hidden_size_; ++r) {
        cache.logit += Wy_[r] * last.h[r];
    }
    cache.probability = sigmoid(cache.logit);
    return cache;
}

LstmNetwork::ForwardCache LstmNetwork::forward_with_cache_cpu(const std::vector<std::vector<float>> &sequence) const {
    SequenceSample sample;
    sample.timesteps = sequence;
    return forward_with_cache_cpu(sample);
}

LstmNetwork::ForwardCache LstmNetwork::forward_with_cache_gpu(const SequenceSample &sample) const {
    ensure_gpu_runtime();

    std::size_t seq_len = sample.timesteps.size();
    bool use_source_view = false;
    if (sample.source_samples != nullptr && sample.length > 0) {
        use_source_view = true;
        seq_len = sample.length;
        if (sample.start_index + sample.length > sample.source_samples->size()) {
            throw std::invalid_argument("LSTM sequence sample window is out of range");
        }
    }

    if (seq_len == 0) {
        throw std::invalid_argument("LSTM sequence cannot be empty");
    }

    ForwardCache cache;
    cache.steps.reserve(seq_len);

    std::vector<float> h_prev(hidden_size_, 0.0f);
    std::vector<float> c_prev(hidden_size_, 0.0f);

    GpuBuffer f_buf = GpuBuffer::allocate(gpu_runtime_->context, hidden_size_ * sizeof(float));
    GpuBuffer i_buf = GpuBuffer::allocate(gpu_runtime_->context, hidden_size_ * sizeof(float));
    GpuBuffer g_buf = GpuBuffer::allocate(gpu_runtime_->context, hidden_size_ * sizeof(float));
    GpuBuffer o_buf = GpuBuffer::allocate(gpu_runtime_->context, hidden_size_ * sizeof(float));
    GpuBuffer z_buf = GpuBuffer::allocate(gpu_runtime_->context, z_size_ * sizeof(float));

    GpuKernel dense_f(gpu_runtime_->program, "dense_forward", gpu_runtime_->context);
    GpuKernel dense_i(gpu_runtime_->program, "dense_forward", gpu_runtime_->context);
    GpuKernel dense_g(gpu_runtime_->program, "dense_forward", gpu_runtime_->context);
    GpuKernel dense_o(gpu_runtime_->program, "dense_forward", gpu_runtime_->context);
    GpuKernel sigmoid_f(gpu_runtime_->program, "sigmoid_activation", gpu_runtime_->context);
    GpuKernel sigmoid_i(gpu_runtime_->program, "sigmoid_activation", gpu_runtime_->context);
    GpuKernel tanh_g(gpu_runtime_->program, "tanh_activation", gpu_runtime_->context);
    GpuKernel sigmoid_o(gpu_runtime_->program, "sigmoid_activation", gpu_runtime_->context);

    for (std::size_t t = 0; t < seq_len; ++t) {
        const std::vector<float> &x_t = use_source_view
            ? (*sample.source_samples)[sample.start_index + t].features
            : sample.timesteps[t];

        if (x_t.size() != input_size_) {
            throw std::invalid_argument("LSTM sequence timestep feature size mismatch");
        }

        StepCache step;
        step.z.assign(z_size_, 0.0f);
        step.f.assign(hidden_size_, 0.0f);
        step.i.assign(hidden_size_, 0.0f);
        step.g.assign(hidden_size_, 0.0f);
        step.o.assign(hidden_size_, 0.0f);
        step.c.assign(hidden_size_, 0.0f);
        step.h.assign(hidden_size_, 0.0f);
        step.c_prev = c_prev;

        for (std::size_t k = 0; k < input_size_; ++k) {
            step.z[k] = x_t[k];
        }
        for (std::size_t k = 0; k < hidden_size_; ++k) {
            step.z[input_size_ + k] = h_prev[k];
        }

        z_buf.copy_from_host_async(step.z.data(), z_size_ * sizeof(float));

        dense_f.set_args({
            KernelArg::buffer(gpu_runtime_->Wf->get_buffer()),
            KernelArg::buffer(gpu_runtime_->bf->get_buffer()),
            KernelArg::buffer(z_buf.get_buffer()),
            KernelArg::buffer(f_buf.get_buffer()),
            KernelArg::scalar_uint(static_cast<uint32_t>(z_size_)),
        });
        dense_f.enqueue_1d(hidden_size_);

        dense_i.set_args({
            KernelArg::buffer(gpu_runtime_->Wi->get_buffer()),
            KernelArg::buffer(gpu_runtime_->bi->get_buffer()),
            KernelArg::buffer(z_buf.get_buffer()),
            KernelArg::buffer(i_buf.get_buffer()),
            KernelArg::scalar_uint(static_cast<uint32_t>(z_size_)),
        });
        dense_i.enqueue_1d(hidden_size_);

        dense_g.set_args({
            KernelArg::buffer(gpu_runtime_->Wg->get_buffer()),
            KernelArg::buffer(gpu_runtime_->bg->get_buffer()),
            KernelArg::buffer(z_buf.get_buffer()),
            KernelArg::buffer(g_buf.get_buffer()),
            KernelArg::scalar_uint(static_cast<uint32_t>(z_size_)),
        });
        dense_g.enqueue_1d(hidden_size_);

        dense_o.set_args({
            KernelArg::buffer(gpu_runtime_->Wo->get_buffer()),
            KernelArg::buffer(gpu_runtime_->bo->get_buffer()),
            KernelArg::buffer(z_buf.get_buffer()),
            KernelArg::buffer(o_buf.get_buffer()),
            KernelArg::scalar_uint(static_cast<uint32_t>(z_size_)),
        });
        dense_o.enqueue_1d(hidden_size_);

        sigmoid_f.set_args({KernelArg::buffer(f_buf.get_buffer())});
        sigmoid_f.enqueue_1d(hidden_size_);

        sigmoid_i.set_args({KernelArg::buffer(i_buf.get_buffer())});
        sigmoid_i.enqueue_1d(hidden_size_);

        tanh_g.set_args({KernelArg::buffer(g_buf.get_buffer())});
        tanh_g.enqueue_1d(hidden_size_);

        sigmoid_o.set_args({KernelArg::buffer(o_buf.get_buffer())});
        sigmoid_o.enqueue_1d(hidden_size_);

        f_buf.copy_to_host_async(step.f.data(), hidden_size_ * sizeof(float));
        i_buf.copy_to_host_async(step.i.data(), hidden_size_ * sizeof(float));
        g_buf.copy_to_host_async(step.g.data(), hidden_size_ * sizeof(float));
        o_buf.copy_to_host_async(step.o.data(), hidden_size_ * sizeof(float));
        gpu_runtime_->context.get_queue().finish();

        for (std::size_t r = 0; r < hidden_size_; ++r) {
            step.c[r] = step.f[r] * c_prev[r] + step.i[r] * step.g[r];
            step.h[r] = step.o[r] * std::tanh(step.c[r]);
        }

        h_prev = step.h;
        c_prev = step.c;
        cache.steps.push_back(std::move(step));
    }

    const StepCache &last = cache.steps.back();
    cache.logit = by_;
    for (std::size_t r = 0; r < hidden_size_; ++r) {
        cache.logit += Wy_[r] * last.h[r];
    }
    cache.probability = sigmoid(cache.logit);
    return cache;
}

LstmNetwork::ForwardCache LstmNetwork::forward_with_cache(const SequenceSample &sample) const {
    if (execution_backend_ == ExecutionBackend::GPU) {
        return forward_with_cache_gpu(sample);
    }
    return forward_with_cache_cpu(sample);
}

float LstmNetwork::compute_loss(float prediction, float target, LossType loss_type) const {
    if (loss_type == LossType::MSE) {
        const float diff = prediction - target;
        return diff * diff;
    }
    const float p = clamp_probability(prediction);
    if (loss_type == LossType::Focal) {
        if (target >= 0.5f) {
            const float modulating = std::pow(1.0f - p, focal_gamma_);
            return -(positive_class_weight_ * focal_alpha_) * modulating * std::log(p);
        }
        const float modulating = std::pow(p, focal_gamma_);
        return -(negative_class_weight_ * (1.0f - focal_alpha_)) * modulating * std::log(1.0f - p);
    }
    return -(positive_class_weight_ * target * std::log(p) +
             negative_class_weight_ * (1.0f - target) * std::log(1.0f - p));
}

float LstmNetwork::output_logit_gradient(float prediction, float target, LossType loss_type) const {
    if (loss_type == LossType::MSE) {
        return 2.0f * (prediction - target) * prediction * (1.0f - prediction);
    }

    const float p = clamp_probability(prediction);
    if (loss_type == LossType::Focal) {
        float dloss_dp = 0.0f;
        if (target >= 0.5f) {
            const float one_minus_p = 1.0f - p;
            const float modulating = std::pow(one_minus_p, focal_gamma_);
            const float modulating_grad =
                focal_gamma_ * std::pow(one_minus_p, focal_gamma_ - 1.0f);
            const float positive_scale = positive_class_weight_ * focal_alpha_;
            dloss_dp = positive_scale * (modulating_grad * std::log(p) - modulating / p);
        } else {
            const float modulating = std::pow(p, focal_gamma_);
            const float modulating_grad = focal_gamma_ * std::pow(p, focal_gamma_ - 1.0f);
            const float negative_scale = negative_class_weight_ * (1.0f - focal_alpha_);
            dloss_dp = negative_scale * (modulating / (1.0f - p) - modulating_grad * std::log(1.0f - p));
        }
        return dloss_dp * p * (1.0f - p);
    }

    return positive_class_weight_ * target * (prediction - 1.0f) +
           negative_class_weight_ * (1.0f - target) * prediction;
}

void LstmNetwork::clear_gradients(Gradients &grads) const {
    grads.dWf.assign(Wf_.size(), 0.0f);
    grads.dWi.assign(Wi_.size(), 0.0f);
    grads.dWg.assign(Wg_.size(), 0.0f);
    grads.dWo.assign(Wo_.size(), 0.0f);
    grads.dbf.assign(bf_.size(), 0.0f);
    grads.dbi.assign(bi_.size(), 0.0f);
    grads.dbg.assign(bg_.size(), 0.0f);
    grads.dbo.assign(bo_.size(), 0.0f);
    grads.dWy.assign(Wy_.size(), 0.0f);
    grads.dby = 0.0f;
}

void LstmNetwork::accumulate_gradients_from_cache(const ForwardCache &cache,
                                                  float target,
                                                  LossType loss_type,
                                                  Gradients &grads) const {
    const float dlogit = output_logit_gradient(cache.probability, target, loss_type);

    const StepCache &last = cache.steps.back();
    for (std::size_t r = 0; r < hidden_size_; ++r) {
        grads.dWy[r] += dlogit * last.h[r];
    }
    grads.dby += dlogit;

    std::vector<float> dh_next(hidden_size_, 0.0f);
    std::vector<float> dc_next(hidden_size_, 0.0f);
    std::vector<float> da_f(hidden_size_, 0.0f);
    std::vector<float> da_i(hidden_size_, 0.0f);
    std::vector<float> da_g(hidden_size_, 0.0f);
    std::vector<float> da_o(hidden_size_, 0.0f);
    std::vector<float> dz(z_size_, 0.0f);
    std::vector<float> dh_prev(hidden_size_, 0.0f);
    std::vector<float> dc_prev(hidden_size_, 0.0f);

    for (std::size_t r = 0; r < hidden_size_; ++r) {
        dh_next[r] = dlogit * Wy_[r];
    }

    if (execution_backend_ == ExecutionBackend::GPU) {
        ensure_gpu_runtime();
    }

    std::unique_ptr<GpuBuffer> tmp_buf;
    std::unique_ptr<GpuBuffer> z_buf;
    std::unique_ptr<GpuBuffer> daf_buf;
    std::unique_ptr<GpuBuffer> dai_buf;
    std::unique_ptr<GpuBuffer> dag_buf;
    std::unique_ptr<GpuBuffer> dao_buf;
    std::unique_ptr<GpuBuffer> dz_buf;
    std::unique_ptr<GpuKernel> grad_accumulate_kernel;
    std::unique_ptr<GpuKernel> backward_input_kernel;
    std::unique_ptr<GpuKernel> add_inplace_kernel;
    std::vector<float> dz_tail;
    if (execution_backend_ == ExecutionBackend::GPU) {
        tmp_buf = std::make_unique<GpuBuffer>(GpuBuffer::allocate(gpu_runtime_->context, z_size_ * sizeof(float)));
        z_buf = std::make_unique<GpuBuffer>(GpuBuffer::allocate(gpu_runtime_->context, z_size_ * sizeof(float)));
        daf_buf = std::make_unique<GpuBuffer>(GpuBuffer::allocate(gpu_runtime_->context, hidden_size_ * sizeof(float)));
        dai_buf = std::make_unique<GpuBuffer>(GpuBuffer::allocate(gpu_runtime_->context, hidden_size_ * sizeof(float)));
        dag_buf = std::make_unique<GpuBuffer>(GpuBuffer::allocate(gpu_runtime_->context, hidden_size_ * sizeof(float)));
        dao_buf = std::make_unique<GpuBuffer>(GpuBuffer::allocate(gpu_runtime_->context, hidden_size_ * sizeof(float)));
        dz_buf = std::make_unique<GpuBuffer>(GpuBuffer::allocate(gpu_runtime_->context, z_size_ * sizeof(float)));
        grad_accumulate_kernel = std::make_unique<GpuKernel>(gpu_runtime_->program, "dense_accumulate_grads", gpu_runtime_->context);
        backward_input_kernel = std::make_unique<GpuKernel>(gpu_runtime_->program, "dense_backward_input", gpu_runtime_->context);
        add_inplace_kernel = std::make_unique<GpuKernel>(gpu_runtime_->program, "elementwise_add_inplace", gpu_runtime_->context);
        dz_tail.assign(hidden_size_, 0.0f);
    }

    for (std::size_t t = cache.steps.size(); t-- > 0;) {
        const StepCache &step = cache.steps[t];
        std::fill(da_f.begin(), da_f.end(), 0.0f);
        std::fill(da_i.begin(), da_i.end(), 0.0f);
        std::fill(da_g.begin(), da_g.end(), 0.0f);
        std::fill(da_o.begin(), da_o.end(), 0.0f);
        std::fill(dz.begin(), dz.end(), 0.0f);
        std::fill(dh_prev.begin(), dh_prev.end(), 0.0f);
        std::fill(dc_prev.begin(), dc_prev.end(), 0.0f);

        for (std::size_t r = 0; r < hidden_size_; ++r) {
            const float tanh_c = std::tanh(step.c[r]);
            const float do_gate = dh_next[r] * tanh_c;
            const float dc = dh_next[r] * step.o[r] * (1.0f - tanh_c * tanh_c) + dc_next[r];
            const float df_gate = dc * step.c_prev[r];
            const float di_gate = dc * step.g[r];
            const float dg_gate = dc * step.i[r];

            dc_prev[r] = dc * step.f[r];

            da_f[r] = df_gate * step.f[r] * (1.0f - step.f[r]);
            da_i[r] = di_gate * step.i[r] * (1.0f - step.i[r]);
            da_o[r] = do_gate * step.o[r] * (1.0f - step.o[r]);
            da_g[r] = dg_gate * (1.0f - step.g[r] * step.g[r]);
        }

        if (execution_backend_ == ExecutionBackend::GPU) {
            z_buf->copy_from_host_async(step.z.data(), z_size_ * sizeof(float));
            daf_buf->copy_from_host_async(da_f.data(), hidden_size_ * sizeof(float));
            dai_buf->copy_from_host_async(da_i.data(), hidden_size_ * sizeof(float));
            dag_buf->copy_from_host_async(da_g.data(), hidden_size_ * sizeof(float));
            dao_buf->copy_from_host_async(da_o.data(), hidden_size_ * sizeof(float));
            dz_buf->copy_from_host_async(dz.data(), z_size_ * sizeof(float));

            auto accumulate = [&](GpuBuffer &gradW, GpuBuffer &gradb, GpuBuffer &da) {
                grad_accumulate_kernel->set_args({
                    KernelArg::buffer(gradW.get_buffer()),
                    KernelArg::buffer(gradb.get_buffer()),
                    KernelArg::buffer(z_buf->get_buffer()),
                    KernelArg::buffer(da.get_buffer()),
                    KernelArg::scalar_uint(static_cast<uint32_t>(z_size_)),
                });
                grad_accumulate_kernel->enqueue_1d(hidden_size_);
            };

            auto accumulate_dz = [&](GpuBuffer &weights, GpuBuffer &da) {
                backward_input_kernel->set_args({
                    KernelArg::buffer(weights.get_buffer()),
                    KernelArg::buffer(da.get_buffer()),
                    KernelArg::buffer(tmp_buf->get_buffer()),
                    KernelArg::scalar_uint(static_cast<uint32_t>(z_size_)),
                    KernelArg::scalar_uint(static_cast<uint32_t>(hidden_size_)),
                });
                backward_input_kernel->enqueue_1d(z_size_);

                add_inplace_kernel->set_args({
                    KernelArg::buffer(dz_buf->get_buffer()),
                    KernelArg::buffer(tmp_buf->get_buffer()),
                });
                add_inplace_kernel->enqueue_1d(z_size_);
            };

            accumulate(*gpu_runtime_->gradWf, *gpu_runtime_->gradbf, *daf_buf);
            accumulate(*gpu_runtime_->gradWi, *gpu_runtime_->gradbi, *dai_buf);
            accumulate(*gpu_runtime_->gradWg, *gpu_runtime_->gradbg, *dag_buf);
            accumulate(*gpu_runtime_->gradWo, *gpu_runtime_->gradbo, *dao_buf);

            accumulate_dz(*gpu_runtime_->Wf, *daf_buf);
            accumulate_dz(*gpu_runtime_->Wi, *dai_buf);
            accumulate_dz(*gpu_runtime_->Wg, *dag_buf);
            accumulate_dz(*gpu_runtime_->Wo, *dao_buf);

            dz_buf->copy_to_host_offset_async(
                dz_tail.data(),
                hidden_size_ * sizeof(float),
                input_size_ * sizeof(float));
            gpu_runtime_->context.get_queue().finish();
            std::fill(dh_prev.begin(), dh_prev.end(), 0.0f);
            for (std::size_t r = 0; r < hidden_size_; ++r) {
                dh_prev[r] = dz_tail[r];
            }
        } else {
            for (std::size_t r = 0; r < hidden_size_; ++r) {
                grads.dbf[r] += da_f[r];
                grads.dbi[r] += da_i[r];
                grads.dbo[r] += da_o[r];
                grads.dbg[r] += da_g[r];

                for (std::size_t k = 0; k < z_size_; ++k) {
                    const std::size_t idx = r * z_size_ + k;
                    grads.dWf[idx] += da_f[r] * step.z[k];
                    grads.dWi[idx] += da_i[r] * step.z[k];
                    grads.dWo[idx] += da_o[r] * step.z[k];
                    grads.dWg[idx] += da_g[r] * step.z[k];
                    dz[k] += Wf_[idx] * da_f[r] +
                             Wi_[idx] * da_i[r] +
                             Wo_[idx] * da_o[r] +
                             Wg_[idx] * da_g[r];
                }
            }
        }

        if (execution_backend_ != ExecutionBackend::GPU) {
            for (std::size_t r = 0; r < hidden_size_; ++r) {
                dh_prev[r] = dz[input_size_ + r];
            }
        }

        dh_next.swap(dh_prev);
        dc_next.swap(dc_prev);
    }
}

void LstmNetwork::apply_optimizer(std::vector<float> &params,
                                  const std::vector<float> &grads,
                                  ParamState &state,
                                  float learning_rate,
                                  float inv_batch,
                                  std::uint64_t step) {
    const float beta1_pow = pow_uint(adam_beta1_, step);
    const float beta2_pow = pow_uint(adam_beta2_, step);
    const float bias_corr1 = 1.0f - beta1_pow;
    const float bias_corr2 = 1.0f - beta2_pow;

    for (std::size_t i = 0; i < params.size(); ++i) {
        const float grad = grads[i] * inv_batch;
        if (optimizer_type_ == OptimizerType::SGD) {
            params[i] -= learning_rate * grad;
        } else if (optimizer_type_ == OptimizerType::Momentum) {
            state.momentum[i] = momentum_ * state.momentum[i] - learning_rate * grad;
            params[i] += state.momentum[i];
        } else if (optimizer_type_ == OptimizerType::AdamW) {
            state.adam_m[i] = adam_beta1_ * state.adam_m[i] + (1.0f - adam_beta1_) * grad;
            state.adam_v[i] = adam_beta2_ * state.adam_v[i] + (1.0f - adam_beta2_) * grad * grad;
            const float m_hat = state.adam_m[i] / bias_corr1;
            const float v_hat = state.adam_v[i] / bias_corr2;
            const float param_before = params[i];
            const float adam_step = m_hat / (std::sqrt(v_hat) + adam_epsilon_);
            params[i] = param_before - learning_rate * (adam_step + weight_decay_ * param_before);
        } else {
            state.adam_m[i] = adam_beta1_ * state.adam_m[i] + (1.0f - adam_beta1_) * grad;
            state.adam_v[i] = adam_beta2_ * state.adam_v[i] + (1.0f - adam_beta2_) * grad * grad;
            const float m_hat = state.adam_m[i] / bias_corr1;
            const float v_hat = state.adam_v[i] / bias_corr2;
            params[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + adam_epsilon_);
        }
    }
}

float LstmNetwork::predict_probability(const std::vector<std::vector<float>> &sequence) const {
    SequenceSample sample;
    sample.timesteps = sequence;
    return forward_with_cache(sample).probability;
}

float LstmNetwork::predict_probability(const SequenceSample &sample) const {
    return forward_with_cache(sample).probability;
}

float LstmNetwork::evaluate_cost(const std::vector<SequenceSample> &dataset, LossType loss_type) const {
    if (dataset.empty()) {
        throw std::invalid_argument("Sequence dataset is empty");
    }

    float total_loss = 0.0f;
    for (const auto &sample : dataset) {
        const float prediction = predict_probability(sample);
        total_loss += compute_loss(prediction, static_cast<float>(sample.label), loss_type);
    }
    return total_loss / static_cast<float>(dataset.size());
}

float LstmNetwork::train_one_epoch(const std::vector<SequenceSample> &dataset,
                                   float learning_rate,
                                   LossType loss_type,
                                   std::size_t batch_size,
                                   float timeout_sec,
                                   bool *timed_out) {
    if (dataset.empty()) {
        throw std::invalid_argument("Sequence dataset is empty");
    }
    if (learning_rate <= 0.0f) {
        throw std::invalid_argument("Learning rate must be positive");
    }
    if (batch_size == 0) {
        throw std::invalid_argument("Batch size must be positive");
    }

    if (timed_out != nullptr) {
        *timed_out = false;
    }

    if (execution_backend_ == ExecutionBackend::GPU) {
        ensure_gpu_runtime();
    }

    std::vector<std::size_t> indices(dataset.size());
    std::iota(indices.begin(), indices.end(), 0);
    static std::mt19937 rng(42u);
    std::shuffle(indices.begin(), indices.end(), rng);

    float total_loss = 0.0f;
    std::size_t processed_samples = 0;
    Gradients grads;
    const auto epoch_start = std::chrono::high_resolution_clock::now();

    for (std::size_t begin = 0; begin < indices.size(); begin += batch_size) {
        if (timeout_sec > 0.0f) {
            const auto now = std::chrono::high_resolution_clock::now();
            const float elapsed_sec =
                std::chrono::duration_cast<std::chrono::duration<float>>(now - epoch_start).count();
            if (elapsed_sec >= timeout_sec) {
                if (timed_out != nullptr) {
                    *timed_out = true;
                }
                break;
            }
        }

        const std::size_t end = std::min(begin + batch_size, indices.size());
        const std::size_t current_batch_size = end - begin;
        clear_gradients(grads);

        if (execution_backend_ == ExecutionBackend::GPU) {
            enqueue_fill_float_lstm(gpu_runtime_->context, gpu_runtime_->program, *gpu_runtime_->gradWf, 0.0f, Wf_.size());
            enqueue_fill_float_lstm(gpu_runtime_->context, gpu_runtime_->program, *gpu_runtime_->gradWi, 0.0f, Wi_.size());
            enqueue_fill_float_lstm(gpu_runtime_->context, gpu_runtime_->program, *gpu_runtime_->gradWg, 0.0f, Wg_.size());
            enqueue_fill_float_lstm(gpu_runtime_->context, gpu_runtime_->program, *gpu_runtime_->gradWo, 0.0f, Wo_.size());
            enqueue_fill_float_lstm(gpu_runtime_->context, gpu_runtime_->program, *gpu_runtime_->gradbf, 0.0f, bf_.size());
            enqueue_fill_float_lstm(gpu_runtime_->context, gpu_runtime_->program, *gpu_runtime_->gradbi, 0.0f, bi_.size());
            enqueue_fill_float_lstm(gpu_runtime_->context, gpu_runtime_->program, *gpu_runtime_->gradbg, 0.0f, bg_.size());
            enqueue_fill_float_lstm(gpu_runtime_->context, gpu_runtime_->program, *gpu_runtime_->gradbo, 0.0f, bo_.size());
        }

        for (std::size_t p = begin; p < end; ++p) {
            const SequenceSample &sample = dataset[indices[p]];
            const float target = static_cast<float>(sample.label);
            const ForwardCache cache = forward_with_cache(sample);
            const float prediction = cache.probability;
            total_loss += compute_loss(prediction, target, loss_type);
            ++processed_samples;
            accumulate_gradients_from_cache(cache, target, loss_type, grads);
        }

        const float inv_batch = 1.0f / static_cast<float>(current_batch_size);
        const std::uint64_t next_step = optimizer_step_ + 1;

        if (execution_backend_ == ExecutionBackend::GPU) {
            const float beta1_pow = pow_uint(adam_beta1_, next_step);
            const float beta2_pow = pow_uint(adam_beta2_, next_step);
            const float bias_corr1 = 1.0f - beta1_pow;
            const float bias_corr2 = 1.0f - beta2_pow;

            enqueue_lstm_gate_update(
                gpu_runtime_->context,
                gpu_runtime_->program,
                optimizer_type_,
                *gpu_runtime_->Wf,
                *gpu_runtime_->bf,
                *gpu_runtime_->gradWf,
                *gpu_runtime_->gradbf,
                *gpu_runtime_->velWf,
                *gpu_runtime_->velbf,
                *gpu_runtime_->adamMWf,
                *gpu_runtime_->adamVWf,
                *gpu_runtime_->adamMbf,
                *gpu_runtime_->adamVbf,
                learning_rate,
                momentum_,
                adam_beta1_,
                adam_beta2_,
                adam_epsilon_,
                weight_decay_,
                bias_corr1,
                bias_corr2,
                inv_batch,
                z_size_,
                hidden_size_);
            enqueue_lstm_gate_update(
                gpu_runtime_->context,
                gpu_runtime_->program,
                optimizer_type_,
                *gpu_runtime_->Wi,
                *gpu_runtime_->bi,
                *gpu_runtime_->gradWi,
                *gpu_runtime_->gradbi,
                *gpu_runtime_->velWi,
                *gpu_runtime_->velbi,
                *gpu_runtime_->adamMWi,
                *gpu_runtime_->adamVWi,
                *gpu_runtime_->adamMbi,
                *gpu_runtime_->adamVbi,
                learning_rate,
                momentum_,
                adam_beta1_,
                adam_beta2_,
                adam_epsilon_,
                weight_decay_,
                bias_corr1,
                bias_corr2,
                inv_batch,
                z_size_,
                hidden_size_);
            enqueue_lstm_gate_update(
                gpu_runtime_->context,
                gpu_runtime_->program,
                optimizer_type_,
                *gpu_runtime_->Wg,
                *gpu_runtime_->bg,
                *gpu_runtime_->gradWg,
                *gpu_runtime_->gradbg,
                *gpu_runtime_->velWg,
                *gpu_runtime_->velbg,
                *gpu_runtime_->adamMWg,
                *gpu_runtime_->adamVWg,
                *gpu_runtime_->adamMbg,
                *gpu_runtime_->adamVbg,
                learning_rate,
                momentum_,
                adam_beta1_,
                adam_beta2_,
                adam_epsilon_,
                weight_decay_,
                bias_corr1,
                bias_corr2,
                inv_batch,
                z_size_,
                hidden_size_);
            enqueue_lstm_gate_update(
                gpu_runtime_->context,
                gpu_runtime_->program,
                optimizer_type_,
                *gpu_runtime_->Wo,
                *gpu_runtime_->bo,
                *gpu_runtime_->gradWo,
                *gpu_runtime_->gradbo,
                *gpu_runtime_->velWo,
                *gpu_runtime_->velbo,
                *gpu_runtime_->adamMWo,
                *gpu_runtime_->adamVWo,
                *gpu_runtime_->adamMbo,
                *gpu_runtime_->adamVbo,
                learning_rate,
                momentum_,
                adam_beta1_,
                adam_beta2_,
                adam_epsilon_,
                weight_decay_,
                bias_corr1,
                bias_corr2,
                inv_batch,
                z_size_,
                hidden_size_);
        }

        if (execution_backend_ != ExecutionBackend::GPU) {
            apply_optimizer(Wf_, grads.dWf, state_Wf_, learning_rate, inv_batch, next_step);
            apply_optimizer(Wi_, grads.dWi, state_Wi_, learning_rate, inv_batch, next_step);
            apply_optimizer(Wg_, grads.dWg, state_Wg_, learning_rate, inv_batch, next_step);
            apply_optimizer(Wo_, grads.dWo, state_Wo_, learning_rate, inv_batch, next_step);
            apply_optimizer(bf_, grads.dbf, state_bf_, learning_rate, inv_batch, next_step);
            apply_optimizer(bi_, grads.dbi, state_bi_, learning_rate, inv_batch, next_step);
            apply_optimizer(bg_, grads.dbg, state_bg_, learning_rate, inv_batch, next_step);
            apply_optimizer(bo_, grads.dbo, state_bo_, learning_rate, inv_batch, next_step);
        }
        apply_optimizer(Wy_, grads.dWy, state_Wy_, learning_rate, inv_batch, next_step);

        const float grad_by = grads.dby * inv_batch;
        if (optimizer_type_ == OptimizerType::SGD) {
            by_ -= learning_rate * grad_by;
        } else if (optimizer_type_ == OptimizerType::Momentum) {
            state_by_momentum_ = momentum_ * state_by_momentum_ - learning_rate * grad_by;
            by_ += state_by_momentum_;
        } else if (optimizer_type_ == OptimizerType::AdamW) {
            const float beta1_pow = pow_uint(adam_beta1_, next_step);
            const float beta2_pow = pow_uint(adam_beta2_, next_step);
            const float bias_corr1 = 1.0f - beta1_pow;
            const float bias_corr2 = 1.0f - beta2_pow;
            state_by_adam_m_ = adam_beta1_ * state_by_adam_m_ + (1.0f - adam_beta1_) * grad_by;
            state_by_adam_v_ = adam_beta2_ * state_by_adam_v_ + (1.0f - adam_beta2_) * grad_by * grad_by;
            const float m_hat = state_by_adam_m_ / bias_corr1;
            const float v_hat = state_by_adam_v_ / bias_corr2;
            by_ -= learning_rate * m_hat / (std::sqrt(v_hat) + adam_epsilon_);
        } else {
            const float beta1_pow = pow_uint(adam_beta1_, next_step);
            const float beta2_pow = pow_uint(adam_beta2_, next_step);
            const float bias_corr1 = 1.0f - beta1_pow;
            const float bias_corr2 = 1.0f - beta2_pow;
            state_by_adam_m_ = adam_beta1_ * state_by_adam_m_ + (1.0f - adam_beta1_) * grad_by;
            state_by_adam_v_ = adam_beta2_ * state_by_adam_v_ + (1.0f - adam_beta2_) * grad_by * grad_by;
            const float m_hat = state_by_adam_m_ / bias_corr1;
            const float v_hat = state_by_adam_v_ / bias_corr2;
            by_ -= learning_rate * m_hat / (std::sqrt(v_hat) + adam_epsilon_);
        }

        optimizer_step_ = next_step;

        if (execution_backend_ != ExecutionBackend::GPU) {
            invalidate_gpu_parameter_cache();
        }
    }

    if (execution_backend_ == ExecutionBackend::GPU) {
        gpu_runtime_->context.get_queue().finish();
        Wf_ = gpu_runtime_->Wf->to_host<float>();
        Wi_ = gpu_runtime_->Wi->to_host<float>();
        Wg_ = gpu_runtime_->Wg->to_host<float>();
        Wo_ = gpu_runtime_->Wo->to_host<float>();
        bf_ = gpu_runtime_->bf->to_host<float>();
        bi_ = gpu_runtime_->bi->to_host<float>();
        bg_ = gpu_runtime_->bg->to_host<float>();
        bo_ = gpu_runtime_->bo->to_host<float>();
    }

    if (processed_samples == 0) {
        return 0.0f;
    }
    return total_loss / static_cast<float>(processed_samples);
}
