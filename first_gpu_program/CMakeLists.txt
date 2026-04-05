cmake_minimum_required(VERSION 3.15)
project(first_gpu_program VERSION 0.1.0 LANGUAGES C CXX)

add_executable(first_gpu_program
	src/main.cpp
	src/config_io.cpp
	src/training_pipeline.cpp
	src/gpu_adapter.cpp
	src/defect_dataset.cpp
	src/mlp_network.cpp
)

find_package(Boost REQUIRED)  # Boost.Compute is header-only
target_include_directories(first_gpu_program PRIVATE
	${Boost_INCLUDE_DIRS}
	${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link OpenCL library
find_package(OpenCL REQUIRED)
target_link_libraries(first_gpu_program PRIVATE OpenCL::OpenCL)

# Optional: Enable C++17
target_compile_features(first_gpu_program PRIVATE cxx_std_17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)