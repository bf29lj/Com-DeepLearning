#include "gpu_adapter.h"

#include <iostream>
#include <sstream>

// GpuContext implementation
GpuContext GpuContext::create_default() {
    try {
        compute::device device = compute::system::default_device();
        compute::context context(device);
        compute::command_queue queue(context, device);
        return GpuContext(device, context, queue);
    } catch (const std::exception &e) {
        throw GpuException(std::string("Failed to create GPU context: ") + e.what());
    }
}

std::string GpuContext::get_device_info() const {
    std::ostringstream oss;
    oss << "Device: " << device_.name() << "\n";
    oss << "Global memory: "
        << device_.get_info<CL_DEVICE_GLOBAL_MEM_SIZE>() / 1024 / 1024 << " MB\n";
    oss << "Max compute units: " << device_.get_info<CL_DEVICE_MAX_COMPUTE_UNITS>() << "\n";
    return oss.str();
}

// GpuProgram implementation
GpuProgram GpuProgram::build(const GpuContext &ctx, const std::string &source) {
    try {
        compute::program program = compute::program::build_with_source(source, ctx.get_context());
        return GpuProgram(program, "");
    } catch (const std::exception &e) {
        std::string build_log = "Build error: ";
        build_log += e.what();
        throw GpuException(build_log);
    }
}

// GpuKernel implementation
void GpuKernel::set_args(const std::vector<KernelArg> &args) {
    for (size_t i = 0; i < args.size(); ++i) {
        const KernelArg &arg = args[i];
        switch (arg.type) {
            case KernelArg::Type::GlobalBuffer: {
                compute::buffer *buf = static_cast<compute::buffer *>(arg.data);
                kernel_.set_arg(i, *buf);
                break;
            }
            case KernelArg::Type::Scalar: {
                if (arg.size == sizeof(uint32_t)) {
                    uint32_t val = *static_cast<uint32_t *>(arg.data);
                    kernel_.set_arg(i, val);
                } else if (arg.size == sizeof(float)) {
                    float val = *static_cast<float *>(arg.data);
                    kernel_.set_arg(i, val);
                } else {
                    throw GpuException("Unsupported scalar type size: " + std::to_string(arg.size));
                }
                break;
            }
            case KernelArg::Type::LocalBuffer:
                // Not implemented yet
                throw GpuException("LocalBuffer not yet implemented");
        }
    }
}

void GpuKernel::enqueue_1d(size_t global_size, size_t local_size) {
    last_event_ = context_.get_queue().enqueue_1d_range_kernel(kernel_, 0, global_size, local_size);
}

// GpuBuffer implementation
GpuBuffer GpuBuffer::allocate(const GpuContext &ctx, size_t element_count) {
    try {
        auto vec = std::make_unique<compute::vector<uint8_t>>(element_count, ctx.get_context());
        return GpuBuffer(std::move(vec), element_count, const_cast<GpuContext *>(&ctx));
    } catch (const std::exception &e) {
        throw GpuException(std::string("Failed to allocate GPU buffer: ") + e.what());
    }
}

void GpuBuffer::copy_to_host(void *host_ptr, size_t bytes) {
    if (bytes > element_count_) {
        throw GpuException("Copy size exceeds buffer capacity");
    }
    try {
        compute::copy(vec_->begin(), vec_->begin() + bytes,
                      static_cast<uint8_t *>(host_ptr), context_->get_queue());
        context_->get_queue().finish();
    } catch (const std::exception &e) {
        throw GpuException(std::string("Failed to copy buffer to host: ") + e.what());
    }
}

void GpuBuffer::copy_from_host(const void *host_ptr, size_t bytes) {
    if (bytes > element_count_) {
        throw GpuException("Copy size exceeds buffer capacity");
    }
    try {
        compute::copy(static_cast<const uint8_t *>(host_ptr),
                      static_cast<const uint8_t *>(host_ptr) + bytes,
                      vec_->begin(), context_->get_queue());
        context_->get_queue().finish();
    } catch (const std::exception &e) {
        throw GpuException(std::string("Failed to copy buffer from host: ") + e.what());
    }
}