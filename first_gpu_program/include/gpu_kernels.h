#pragma once

#include "gpu_adapter.h"
#include "gpu_kernel_sources.h"

#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace gpu {

// Kernel parameter type descriptors
struct BufferParam {
    static constexpr bool is_buffer = true;
    static constexpr const char *type_name = "buffer";
};

struct ScalarParam {
    static constexpr bool is_buffer = false;
    static constexpr const char *type_name = "scalar";
};

// Specific scalar types
struct UintParam : public ScalarParam {
    using value_type = uint32_t;
};

struct FloatParam : public ScalarParam {
    using value_type = float;
};

// Kernel trait definitions with compile-time parameter verification
template <typename... ParamTypes>
struct KernelTraits;

// dense_forward kernel: weights, biases, input, output, input_size
struct DenseForwardKernel {
    using Params = std::tuple<BufferParam, BufferParam, BufferParam, BufferParam, UintParam>;
    static constexpr const char *name = "dense_forward";
    static constexpr size_t param_count = 5;

    // Compile-time verification: ensure arguments match expected types
    template <typename T1, typename T2, typename T3, typename T4, typename T5>
    static constexpr bool verify_params() {
        return std::is_same_v<T1, BufferParam> && std::is_same_v<T2, BufferParam> &&
               std::is_same_v<T3, BufferParam> && std::is_same_v<T4, BufferParam> &&
               std::is_same_v<T5, UintParam>;
    }
};

// relu_activation kernel: values
struct ReluActivationKernel {
    using Params = std::tuple<BufferParam>;
    static constexpr const char *name = "relu_activation";
    static constexpr size_t param_count = 1;

    template <typename T1>
    static constexpr bool verify_params() {
        return std::is_same_v<T1, BufferParam>;
    }
};

// sigmoid_activation kernel: values
struct SigmoidActivationKernel {
    using Params = std::tuple<BufferParam>;
    static constexpr const char *name = "sigmoid_activation";
    static constexpr size_t param_count = 1;

    template <typename T1>
    static constexpr bool verify_params() {
        return std::is_same_v<T1, BufferParam>;
    }
};

// Type-safe kernel argument builder
template <typename KernelDef>
class TypedKernelCall {
public:
    TypedKernelCall(const GpuProgram &prog, GpuContext &ctx)
        : kernel_(prog, KernelDef::name, ctx) {}

    // Variadic setter that verifies parameter count and types at compile time
    template <typename... Args>
    void set_typed_args(Args &&...args) {
        static_assert(sizeof...(Args) == KernelDef::param_count,
                      "Argument count mismatch with kernel definition");
        static_assert(KernelDef::template verify_params<std::decay_t<Args>...>(),
                      "Argument types do not match kernel definition");

        std::vector<KernelArg> kernel_args;
        collect_args(kernel_args, std::forward<Args>(args)...);
        kernel_.set_args(kernel_args);
    }

    // Enqueue element-wise operation (1 work item per element)
    void enqueue_element_wise(size_t element_count, size_t local_size = 0) {
        kernel_.enqueue_1d(element_count, local_size);
    }

    // Enqueue vector operation (1 work item per vector element)
    void enqueue_vector_wise(size_t vector_count, size_t local_size = 0) {
        kernel_.enqueue_1d(vector_count, local_size);
    }

    GpuKernel *get_kernel() { return &kernel_; }

private:
    GpuKernel kernel_;

    void collect_args(std::vector<KernelArg> &) {}

    template <typename FirstArg, typename... RestArgs>
    void collect_args(std::vector<KernelArg> &args, FirstArg &&first, RestArgs &&...rest) {
        if constexpr (std::is_same_v<std::decay_t<FirstArg>, BufferParam>) {
            // This is a placeholder; actual buffer passed through get_buffer() separately
            throw GpuException("Use KernelArg::buffer() for buffer parameters");
        } else if constexpr (std::is_same_v<std::decay_t<FirstArg>, UintParam>) {
            args.push_back(KernelArg::scalar_uint(static_cast<uint32_t>(first)));
        } else if constexpr (std::is_same_v<std::decay_t<FirstArg>, FloatParam>) {
            args.push_back(KernelArg::scalar_float(static_cast<float>(first)));
        }

        if constexpr (sizeof...(RestArgs) > 0) {
            collect_args(args, std::forward<RestArgs>(rest)...);
        }
    }
};

// Simplified wrapper for element-wise operations
template <typename KernelDef>
class ElementwiseKernelOp {
public:
    ElementwiseKernelOp(const GpuProgram &prog, GpuContext &ctx)
        : typed_call_(prog, ctx) {}

    template <typename... Args>
    void apply(GpuBuffer &output, Args &&...scalar_args) {
        static_assert(KernelDef::param_count == 1, "Elementwise must have exactly 1 buffer param");

        std::vector<KernelArg> args = {KernelArg::buffer(output.get_buffer())};
        typed_call_.get_kernel()->set_args(args);
        typed_call_.enqueue_element_wise(output.element_count() / sizeof(float));
    }

private:
    TypedKernelCall<KernelDef> typed_call_;
};

// Simplified wrapper for dense operations
template <typename KernelDef>
class DenseKernelOp {
public:
    DenseKernelOp(const GpuProgram &prog, GpuContext &ctx) : typed_call_(prog, ctx) {}

    void forward(GpuBuffer &weights, GpuBuffer &biases, GpuBuffer &input, GpuBuffer &output,
                 uint32_t input_size) {
        static_assert(std::is_same_v<KernelDef, DenseForwardKernel>,
                      "This method only works with DenseForwardKernel");

        std::vector<KernelArg> args = {
            KernelArg::buffer(weights.get_buffer()),
            KernelArg::buffer(biases.get_buffer()),
            KernelArg::buffer(input.get_buffer()),
            KernelArg::buffer(output.get_buffer()),
            KernelArg::scalar_uint(input_size),
        };

        typed_call_.get_kernel()->set_args(args);
        typed_call_.enqueue_vector_wise(output.element_count() / sizeof(float));
    }

private:
    TypedKernelCall<KernelDef> typed_call_;
};

// Factory function to create pre-validated GPU program
inline GpuProgram create_gpu_program(GpuContext &ctx) {
    return GpuProgram::build(ctx, build_default_kernel_source());
}

}  // namespace gpu