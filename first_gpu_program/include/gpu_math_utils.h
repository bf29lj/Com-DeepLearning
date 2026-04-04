#pragma once

#include "gpu_adapter.h"

namespace gpu {

// Parallel activation derivative operators for 1D buffers.
class ActivationDerivativeOps {
public:
    // Computes ReLU'(x) from activated values.
    static void relu_derivative(GpuProgram &program, GpuContext &ctx, GpuBuffer &activated, GpuBuffer &out) {
        run_two_input_one_output(program, ctx, "relu_derivative", activated, out);
    }

    // Computes Sigmoid'(x) from sigmoid outputs.
    static void sigmoid_derivative(GpuProgram &program, GpuContext &ctx, GpuBuffer &activated, GpuBuffer &out) {
        run_two_input_one_output(program, ctx, "sigmoid_derivative", activated, out);
    }

private:
    static void run_two_input_one_output(
        GpuProgram &program,
        GpuContext &ctx,
        const char *kernel_name,
        GpuBuffer &input,
        GpuBuffer &out)
    {
        GpuKernel kernel(program, kernel_name, ctx);
        const size_t n = out.element_count() / sizeof(float);
        kernel.set_args({
            KernelArg::buffer(input.get_buffer()),
            KernelArg::buffer(out.get_buffer()),
        });
        kernel.enqueue_1d(n);
    }
};

// Parallel loss and loss-derivative operators for 1D buffers.
class LossOps {
public:
    // Element-wise MSE loss values.
    static void mse_loss(
        GpuProgram &program,
        GpuContext &ctx,
        GpuBuffer &prediction,
        GpuBuffer &target,
        GpuBuffer &losses)
    {
        run_two_input_loss(program, ctx, "mse_loss_elementwise", prediction, target, losses);
    }

    // Element-wise d(MSE)/d(prediction), scaled by inv_n.
    static void mse_derivative(
        GpuProgram &program,
        GpuContext &ctx,
        GpuBuffer &prediction,
        GpuBuffer &target,
        GpuBuffer &grad,
        float inv_n)
    {
        GpuKernel kernel(program, "mse_loss_derivative", ctx);
        const size_t n = grad.element_count() / sizeof(float);
        kernel.set_args({
            KernelArg::buffer(prediction.get_buffer()),
            KernelArg::buffer(target.get_buffer()),
            KernelArg::buffer(grad.get_buffer()),
            KernelArg::scalar_float(inv_n),
        });
        kernel.enqueue_1d(n);
    }

    // Element-wise BCE loss values with epsilon clamp.
    static void bce_loss(
        GpuProgram &program,
        GpuContext &ctx,
        GpuBuffer &prediction,
        GpuBuffer &target,
        GpuBuffer &losses,
        float epsilon = 1e-7f)
    {
        GpuKernel kernel(program, "bce_loss_elementwise", ctx);
        const size_t n = losses.element_count() / sizeof(float);
        kernel.set_args({
            KernelArg::buffer(prediction.get_buffer()),
            KernelArg::buffer(target.get_buffer()),
            KernelArg::buffer(losses.get_buffer()),
            KernelArg::scalar_float(epsilon),
        });
        kernel.enqueue_1d(n);
    }

    // Element-wise d(BCE)/d(prediction), scaled by inv_n.
    static void bce_derivative(
        GpuProgram &program,
        GpuContext &ctx,
        GpuBuffer &prediction,
        GpuBuffer &target,
        GpuBuffer &grad,
        float inv_n,
        float epsilon = 1e-7f)
    {
        GpuKernel kernel(program, "bce_loss_derivative", ctx);
        const size_t n = grad.element_count() / sizeof(float);
        kernel.set_args({
            KernelArg::buffer(prediction.get_buffer()),
            KernelArg::buffer(target.get_buffer()),
            KernelArg::buffer(grad.get_buffer()),
            KernelArg::scalar_float(epsilon),
            KernelArg::scalar_float(inv_n),
        });
        kernel.enqueue_1d(n);
    }

private:
    static void run_two_input_loss(
        GpuProgram &program,
        GpuContext &ctx,
        const char *kernel_name,
        GpuBuffer &prediction,
        GpuBuffer &target,
        GpuBuffer &losses)
    {
        GpuKernel kernel(program, kernel_name, ctx);
        const size_t n = losses.element_count() / sizeof(float);
        kernel.set_args({
            KernelArg::buffer(prediction.get_buffer()),
            KernelArg::buffer(target.get_buffer()),
            KernelArg::buffer(losses.get_buffer()),
        });
        kernel.enqueue_1d(n);
    }
};

}  // namespace gpu