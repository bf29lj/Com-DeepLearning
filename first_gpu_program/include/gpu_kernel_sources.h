#pragma once

#include <string>

namespace gpu {

// Core inference kernels used by configurable 1D MLP forward execution.
constexpr const char *CORE_KERNEL_SOURCE = R"CLC(
    __kernel void dense_forward(
        __global const float *weights,
        __global const float *biases,
        __global const float *input,
        __global float *output,
        const uint input_size)
    {
        const uint neuron = get_global_id(0);
        float sum = biases[neuron];
        const uint offset = neuron * input_size;
        for (uint i = 0; i < input_size; ++i) {
            sum += weights[offset + i] * input[i];
        }
        output[neuron] = sum;
    }

    __kernel void relu_activation(__global float *values)
    {
        const uint index = get_global_id(0);
        values[index] = fmax(values[index], 0.0f);
    }

    __kernel void sigmoid_activation(__global float *values)
    {
        const uint index = get_global_id(0);
        values[index] = 1.0f / (1.0f + exp(-values[index]));
    }

    __kernel void tanh_activation(__global float *values)
    {
        const uint index = get_global_id(0);
        values[index] = tanh(values[index]);
    }

    __kernel void leaky_relu_activation(__global float *values)
    {
        const uint index = get_global_id(0);
        const float x = values[index];
        values[index] = x > 0.0f ? x : 0.01f * x;
    }

    __kernel void gelu_activation(__global float *values)
    {
        const uint index = get_global_id(0);
        const float x = values[index];
        const float c = 0.044715f;
        const float s = 0.7978845608f;
        const float u = s * (x + c * x * x * x);
        values[index] = 0.5f * x * (1.0f + tanh(u));
    }
)CLC";

// Common math/loss kernels for training or metric computation.
constexpr const char *MATH_KERNEL_SOURCE = R"CLC(
    __kernel void relu_derivative(
        __global const float *activated,
        __global float *derivative)
    {
        const uint idx = get_global_id(0);
        derivative[idx] = activated[idx] > 0.0f ? 1.0f : 0.0f;
    }

    __kernel void sigmoid_derivative(
        __global const float *activated,
        __global float *derivative)
    {
        const uint idx = get_global_id(0);
        const float a = activated[idx];
        derivative[idx] = a * (1.0f - a);
    }

    __kernel void tanh_derivative(
        __global const float *activated,
        __global float *derivative)
    {
        const uint idx = get_global_id(0);
        const float a = activated[idx];
        derivative[idx] = 1.0f - a * a;
    }

    __kernel void leaky_relu_derivative(
        __global const float *activated,
        __global float *derivative)
    {
        const uint idx = get_global_id(0);
        derivative[idx] = activated[idx] > 0.0f ? 1.0f : 0.01f;
    }

    __kernel void gelu_derivative(
        __global const float *input,
        __global float *derivative)
    {
        const uint idx = get_global_id(0);
        const float x = input[idx];
        const float c = 0.044715f;
        const float s = 0.7978845608f;
        const float x2 = x * x;
        const float x3 = x2 * x;
        const float u = s * (x + c * x3);
        const float t = tanh(u);
        const float sech2 = 1.0f - t * t;
        const float du_dx = s * (1.0f + 3.0f * c * x2);
        derivative[idx] = 0.5f * (1.0f + t) + 0.5f * x * sech2 * du_dx;
    }

    __kernel void mse_loss_elementwise(
        __global const float *prediction,
        __global const float *target,
        __global float *losses)
    {
        const uint idx = get_global_id(0);
        const float d = prediction[idx] - target[idx];
        losses[idx] = d * d;
    }

    __kernel void mse_loss_derivative(
        __global const float *prediction,
        __global const float *target,
        __global float *grad,
        const float inv_n)
    {
        const uint idx = get_global_id(0);
        grad[idx] = 2.0f * (prediction[idx] - target[idx]) * inv_n;
    }

    __kernel void bce_loss_elementwise(
        __global const float *prediction,
        __global const float *target,
        __global float *losses,
        const float epsilon,
        const float positive_weight,
        const float negative_weight)
    {
        const uint idx = get_global_id(0);
        const float p = clamp(prediction[idx], epsilon, 1.0f - epsilon);
        const float y = target[idx];
        losses[idx] = -(positive_weight * y * log(p) + negative_weight * (1.0f - y) * log(1.0f - p));
    }

    __kernel void bce_loss_derivative(
        __global const float *prediction,
        __global const float *target,
        __global float *grad,
        const float epsilon,
        const float inv_n,
        const float positive_weight,
        const float negative_weight)
    {
        const uint idx = get_global_id(0);
        const float p = clamp(prediction[idx], epsilon, 1.0f - epsilon);
        const float y = target[idx];
        grad[idx] = (-(positive_weight * y / p) + (negative_weight * (1.0f - y) / (1.0f - p))) * inv_n;
    }

    __kernel void elementwise_multiply_inplace(
        __global float *values,
        __global const float *factors)
    {
        const uint idx = get_global_id(0);
        values[idx] *= factors[idx];
    }

    __kernel void elementwise_add_inplace(
        __global float *values,
        __global const float *addends)
    {
        const uint idx = get_global_id(0);
        values[idx] += addends[idx];
    }

    __kernel void fill_float(
        __global float *values,
        const float value,
        const uint n)
    {
        const uint idx = get_global_id(0);
        if (idx < n) {
            values[idx] = value;
        }
    }

    __kernel void concat_input_hidden(
        __global const float *input,
        __global const float *hidden,
        __global float *z,
        const uint input_size)
    {
        const uint idx = get_global_id(0);
        if (idx < input_size) {
            z[idx] = input[idx];
        } else {
            z[idx] = hidden[idx - input_size];
        }
    }

    __kernel void lstm_cell_update(
        __global const float *f,
        __global const float *i,
        __global const float *g,
        __global const float *o,
        __global const float *c_prev,
        __global float *c_out,
        __global float *h_out)
    {
        const uint idx = get_global_id(0);
        const float c_val = f[idx] * c_prev[idx] + i[idx] * g[idx];
        c_out[idx] = c_val;
        h_out[idx] = o[idx] * tanh(c_val);
    }

    __kernel void dense_accumulate_grads(
        __global float *grad_weights,
        __global float *grad_biases,
        __global const float *input,
        __global const float *grad_output,
        const uint input_size)
    {
        const uint out = get_global_id(0);
        const float delta = grad_output[out];
        grad_biases[out] += delta;

        const uint offset = out * input_size;
        for (uint in = 0; in < input_size; ++in) {
            grad_weights[offset + in] += delta * input[in];
        }
    }

    __kernel void dense_backward_input(
        __global const float *weights,
        __global const float *grad_output,
        __global float *grad_input,
        const uint input_size,
        const uint output_size)
    {
        const uint in = get_global_id(0);
        float sum = 0.0f;
        for (uint out = 0; out < output_size; ++out) {
            sum += weights[out * input_size + in] * grad_output[out];
        }
        grad_input[in] = sum;
    }

    __kernel void dense_sgd_update(
        __global float *weights,
        __global float *biases,
        __global const float *grad_weights,
        __global const float *grad_biases,
        const float learning_rate,
        const float inv_batch,
        const uint input_size)
    {
        const uint out = get_global_id(0);
        const float delta = grad_biases[out] * inv_batch;
        biases[out] -= learning_rate * delta;

        const uint offset = out * input_size;
        for (uint in = 0; in < input_size; ++in) {
            const float g = grad_weights[offset + in] * inv_batch;
            weights[offset + in] -= learning_rate * g;
        }
    }

    __kernel void dense_momentum_update(
        __global float *weights,
        __global float *biases,
        __global float *velocity_weights,
        __global float *velocity_biases,
        __global const float *grad_weights,
        __global const float *grad_biases,
        const float learning_rate,
        const float momentum,
        const float inv_batch,
        const uint input_size)
    {
        const uint out = get_global_id(0);
        const float grad_b = grad_biases[out] * inv_batch;
        velocity_biases[out] = momentum * velocity_biases[out] - learning_rate * grad_b;
        biases[out] += velocity_biases[out];

        const uint offset = out * input_size;
        for (uint in = 0; in < input_size; ++in) {
            const float grad_w = grad_weights[offset + in] * inv_batch;
            velocity_weights[offset + in] =
                momentum * velocity_weights[offset + in] - learning_rate * grad_w;
            weights[offset + in] += velocity_weights[offset + in];
        }
    }

    __kernel void dense_adam_update(
        __global float *weights,
        __global float *biases,
        __global float *adam_m_weights,
        __global float *adam_v_weights,
        __global float *adam_m_biases,
        __global float *adam_v_biases,
        __global const float *grad_weights,
        __global const float *grad_biases,
        const float learning_rate,
        const float beta1,
        const float beta2,
        const float epsilon,
        const float weight_decay,
        const float bias_corr1,
        const float bias_corr2,
        const float inv_batch,
        const uint input_size)
    {
        const uint out = get_global_id(0);

        const float grad_b = grad_biases[out] * inv_batch;
        const float m_b = beta1 * adam_m_biases[out] + (1.0f - beta1) * grad_b;
        const float v_b = beta2 * adam_v_biases[out] + (1.0f - beta2) * grad_b * grad_b;
        adam_m_biases[out] = m_b;
        adam_v_biases[out] = v_b;
        const float m_hat_b = m_b / bias_corr1;
        const float v_hat_b = v_b / bias_corr2;
        biases[out] -= learning_rate * m_hat_b / (sqrt(v_hat_b) + epsilon);

        const uint offset = out * input_size;
        for (uint in = 0; in < input_size; ++in) {
            const float weight_before = weights[offset + in];
            const float grad_w = grad_weights[offset + in] * inv_batch;
            const float m_w = beta1 * adam_m_weights[offset + in] + (1.0f - beta1) * grad_w;
            const float v_w = beta2 * adam_v_weights[offset + in] + (1.0f - beta2) * grad_w * grad_w;
            adam_m_weights[offset + in] = m_w;
            adam_v_weights[offset + in] = v_w;
            const float m_hat_w = m_w / bias_corr1;
            const float v_hat_w = v_w / bias_corr2;
            weights[offset + in] = weight_before - learning_rate * (
                m_hat_w / (sqrt(v_hat_w) + epsilon) + weight_decay * weight_before);
        }
    }
)CLC";

// Builds a combined source string so all kernels can be compiled once.
inline std::string build_default_kernel_source() {
    std::string source;
    source.reserve(8192);
    source += CORE_KERNEL_SOURCE;
    source += "\n";
    source += MATH_KERNEL_SOURCE;
    return source;
}

}  // namespace gpu