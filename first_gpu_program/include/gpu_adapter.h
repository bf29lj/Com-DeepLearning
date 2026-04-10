#pragma once

#include <boost/compute.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace compute = boost::compute;

// Exception for GPU-related errors
class GpuException : public std::runtime_error {
public:
    explicit GpuException(const std::string &message) : std::runtime_error(message) {}
};

// Wraps GPU device, context and command queue
class GpuContext {
public:
    static GpuContext create_default();

    compute::device get_device() const { return device_; }
    compute::context get_context() const { return context_; }
    compute::command_queue &get_queue() { return queue_; }

    std::string get_device_info() const;

private:
    GpuContext(compute::device device,
               compute::context context,
               compute::command_queue queue)
        : device_(device), context_(context), queue_(queue) {}

    compute::device device_;
    compute::context context_;
    compute::command_queue queue_;
};

// Wraps OpenCL program compilation and caching
class GpuProgram {
public:
    static GpuProgram build(const GpuContext &ctx, const std::string &source);

    compute::program get_program() const { return program_; }
    std::string get_build_log() const { return build_log_; }

private:
    GpuProgram(compute::program program, const std::string &build_log = "")
        : program_(program), build_log_(build_log) {}

    compute::program program_;
    std::string build_log_;
};

// Unified parameter structure for kernel arguments
struct KernelArg {
    enum class Type {
        GlobalBuffer,
        LocalBuffer,
        Scalar,
    };

    Type type;
    void *data;  // For Buffer: compute::buffer, For Scalar: const float*, int*, etc
    size_t size; // For Buffer: byte size, For Scalar: element size

    static KernelArg buffer(compute::buffer &buf) {
        return {Type::GlobalBuffer, &buf, 0};
    }

    static KernelArg scalar_uint(uint32_t val) {
        static thread_local uint32_t storage;
        storage = val;
        return {Type::Scalar, &storage, sizeof(uint32_t)};
    }

    static KernelArg scalar_float(float val) {
        static thread_local float storage;
        storage = val;
        return {Type::Scalar, &storage, sizeof(float)};
    }
};

// Wraps kernel execution with standardized parameter handling
class GpuKernel {
public:
    GpuKernel(const GpuProgram &prog, const std::string &kernel_name, GpuContext &ctx)
        : kernel_(prog.get_program().create_kernel(kernel_name)), context_(ctx) {}

    // Set arguments in order
    void set_args(const std::vector<KernelArg> &args);

    // Enqueue 1D range kernel
    void enqueue_1d(size_t global_size, size_t local_size = 0);

    compute::event get_last_event() const { return last_event_; }

private:
    compute::kernel kernel_;
    GpuContext &context_;
    compute::event last_event_;
};

// Wraps GPU buffer (compute::vector) with unified host<->device copy interface
class GpuBuffer {
public:
    // Allocate uninitialized on device
    static GpuBuffer allocate(const GpuContext &ctx, size_t element_count);

    // Create from host data
    template <typename T>
    static GpuBuffer from_host(GpuContext &ctx, const std::vector<T> &host_data) {
        GpuBuffer buf = allocate(ctx, host_data.size() * sizeof(T));
        buf.copy_from_host(host_data.data(), host_data.size() * sizeof(T));
        return buf;
    }

    // Copy to host
    void copy_to_host(void *host_ptr, size_t bytes);
    void copy_to_host_offset(void *host_ptr, size_t bytes, size_t offset_bytes);
    void copy_to_host_async(void *host_ptr, size_t bytes);
    void copy_to_host_offset_async(void *host_ptr, size_t bytes, size_t offset_bytes);

    template <typename T>
    std::vector<T> to_host() {
        size_t byte_count = element_count_;
        size_t element_count = byte_count / sizeof(T);
        std::vector<T> result(element_count);
        copy_to_host(result.data(), byte_count);
        return result;
    }

    // Copy from host
    void copy_from_host(const void *host_ptr, size_t bytes);
    void copy_from_host_async(const void *host_ptr, size_t bytes);

    // Get underlying compute::buffer for kernel setup
    compute::buffer &get_buffer() {
        if (!vec_) throw GpuException("Buffer is empty");
        return const_cast<compute::buffer &>(vec_->get_buffer());
    }

    size_t element_count() const { return element_count_; }

    // Move constructor
    GpuBuffer(GpuBuffer &&other) noexcept = default;
    GpuBuffer &operator=(GpuBuffer &&other) noexcept = default;

private:
    GpuBuffer(std::unique_ptr<compute::vector<uint8_t>> vec, size_t count, GpuContext *ctx)
        : vec_(std::move(vec)), element_count_(count), context_(ctx) {}

    std::unique_ptr<compute::vector<uint8_t>> vec_;
    size_t element_count_;
    GpuContext *context_;
};