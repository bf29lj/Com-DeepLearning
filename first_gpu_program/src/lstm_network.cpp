#include "lstm_network.h"

#include <algorithm>
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

} // namespace

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

void LstmNetwork::set_class_weights(float positive_weight, float negative_weight) {
    if (positive_weight <= 0.0f || negative_weight <= 0.0f || !std::isfinite(positive_weight) ||
        !std::isfinite(negative_weight)) {
        throw std::invalid_argument("Class weights must be positive finite numbers");
    }
    positive_class_weight_ = positive_weight;
    negative_class_weight_ = negative_weight;
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
}

float LstmNetwork::sigmoid(float x) {
    const float clamped = std::clamp(x, -40.0f, 40.0f);
    return 1.0f / (1.0f + std::exp(-clamped));
}

LstmNetwork::ForwardCache LstmNetwork::forward_with_cache(const SequenceSample &sample) const {
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

LstmNetwork::ForwardCache LstmNetwork::forward_with_cache(const std::vector<std::vector<float>> &sequence) const {
    SequenceSample sample;
    sample.timesteps = sequence;
    return forward_with_cache(sample);
}

float LstmNetwork::compute_loss(float prediction, float target, LossType loss_type) const {
    if (loss_type == LossType::MSE) {
        const float diff = prediction - target;
        return diff * diff;
    }
    const float p = clamp_probability(prediction);
    return -(positive_class_weight_ * target * std::log(p) +
             negative_class_weight_ * (1.0f - target) * std::log(1.0f - p));
}

float LstmNetwork::output_logit_gradient(float prediction, float target, LossType loss_type) const {
    if (loss_type == LossType::MSE) {
        return 2.0f * (prediction - target) * prediction * (1.0f - prediction);
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

        for (std::size_t r = 0; r < hidden_size_; ++r) {
            dh_prev[r] = dz[input_size_ + r];
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
    return forward_with_cache(sequence).probability;
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
                                   std::size_t batch_size) {
    if (dataset.empty()) {
        throw std::invalid_argument("Sequence dataset is empty");
    }
    if (learning_rate <= 0.0f) {
        throw std::invalid_argument("Learning rate must be positive");
    }
    if (batch_size == 0) {
        throw std::invalid_argument("Batch size must be positive");
    }

    std::vector<std::size_t> indices(dataset.size());
    std::iota(indices.begin(), indices.end(), 0);
    static std::mt19937 rng(42u);
    std::shuffle(indices.begin(), indices.end(), rng);

    float total_loss = 0.0f;
    Gradients grads;

    for (std::size_t begin = 0; begin < indices.size(); begin += batch_size) {
        const std::size_t end = std::min(begin + batch_size, indices.size());
        const std::size_t current_batch_size = end - begin;
        clear_gradients(grads);

        for (std::size_t p = begin; p < end; ++p) {
            const SequenceSample &sample = dataset[indices[p]];
            const float target = static_cast<float>(sample.label);
            const ForwardCache cache = forward_with_cache(sample);
            const float prediction = cache.probability;
            total_loss += compute_loss(prediction, target, loss_type);
            accumulate_gradients_from_cache(cache, target, loss_type, grads);
        }

        const float inv_batch = 1.0f / static_cast<float>(current_batch_size);
        const std::uint64_t next_step = optimizer_step_ + 1;

        apply_optimizer(Wf_, grads.dWf, state_Wf_, learning_rate, inv_batch, next_step);
        apply_optimizer(Wi_, grads.dWi, state_Wi_, learning_rate, inv_batch, next_step);
        apply_optimizer(Wg_, grads.dWg, state_Wg_, learning_rate, inv_batch, next_step);
        apply_optimizer(Wo_, grads.dWo, state_Wo_, learning_rate, inv_batch, next_step);
        apply_optimizer(bf_, grads.dbf, state_bf_, learning_rate, inv_batch, next_step);
        apply_optimizer(bi_, grads.dbi, state_bi_, learning_rate, inv_batch, next_step);
        apply_optimizer(bg_, grads.dbg, state_bg_, learning_rate, inv_batch, next_step);
        apply_optimizer(bo_, grads.dbo, state_bo_, learning_rate, inv_batch, next_step);
        apply_optimizer(Wy_, grads.dWy, state_Wy_, learning_rate, inv_batch, next_step);

        const float grad_by = grads.dby * inv_batch;
        if (optimizer_type_ == OptimizerType::SGD) {
            by_ -= learning_rate * grad_by;
        } else if (optimizer_type_ == OptimizerType::Momentum) {
            state_by_momentum_ = momentum_ * state_by_momentum_ - learning_rate * grad_by;
            by_ += state_by_momentum_;
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
    }

    return total_loss / static_cast<float>(dataset.size());
}
