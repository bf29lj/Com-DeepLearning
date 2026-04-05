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
        __global const float *input,
        __global const float *grad_output,
        const float learning_rate,
        const uint input_size)
    {
        const uint out = get_global_id(0);
        const float delta = grad_output[out];
        biases[out] -= learning_rate * delta;

        const uint offset = out * input_size;
        for (uint in = 0; in < input_size; ++in) {
            weights[offset + in] -= learning_rate * delta * input[in];
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