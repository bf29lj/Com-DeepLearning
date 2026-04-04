#include "defect_dataset.h"
#include "gpu_adapter.h"
#include "mlp_network.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <vector>

namespace {

std::filesystem::path resolve_dataset_path() {
    const std::filesystem::path candidates[] = {
        "data/processed/train.csv",
        "../data/processed/train.csv",
        "../../data/processed/train.csv",
    };

    for (const auto &candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }

    throw std::runtime_error("Failed to locate dataset file under data/processed/train.csv");
}

}  // namespace

int main() {
    try {
        const LossType selected_loss = LossType::BCE;
        const float learning_rate = 0.01f;

        GpuContext ctx = GpuContext::create_default();
        std::cout << ctx.get_device_info();
        std::cout << "\n";

        const auto dataset = ManufacturingDefectDataset::load_csv(resolve_dataset_path());
        std::cout << "Loaded samples: " << dataset.size() << "\n";
        std::cout << "Feature count: " << dataset.feature_count() << "\n\n";

        std::vector<OperationConfig> operations = {
            OperationConfig::linear(dataset.feature_count(), 8),
            OperationConfig::relu(),
            OperationConfig::linear(8, 4),
            OperationConfig::relu(),
            OperationConfig::linear(4, 1),
            OperationConfig::sigmoid(),
        };
        MlpNetwork network(ctx, std::move(operations));

        const DefectSample &first_sample = dataset.sample(0);
        std::vector<float> input = first_sample.features;

        auto start = std::chrono::high_resolution_clock::now();
        auto output = network.forward(input);
        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "Input: ";
        for (float v : input) {
            std::cout << v << " ";
        }
        std::cout << "\n";

        std::cout << "Label: " << static_cast<int>(first_sample.label) << "\n";

        std::cout << "Output: ";
        for (float v : output) {
            std::cout << v << " ";
        }
        std::cout << "\n";

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Inference time: " << duration.count() << " us\n";

        const float before_cost = network.evaluate_cost(dataset, selected_loss);
        const float train_epoch_cost = network.train_one_epoch(dataset, learning_rate, selected_loss);
        const float after_cost = network.evaluate_cost(dataset, selected_loss);

        const auto cost_start = std::chrono::high_resolution_clock::now();
        const float average_cost = network.evaluate_cost(dataset, selected_loss);
        const auto cost_end = std::chrono::high_resolution_clock::now();
        const auto cost_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cost_end - cost_start);

        std::cout << "Loss function: " << (selected_loss == LossType::BCE ? "BCE" : "MSE") << "\n";
        std::cout << "Learning rate: " << learning_rate << "\n";
        std::cout << "Cost before training: " << before_cost << "\n";
        std::cout << "Average epoch training cost: " << train_epoch_cost << "\n";
        std::cout << "Cost after one epoch: " << after_cost << "\n";
        std::cout << "Average dataset cost: " << average_cost << "\n";
        std::cout << "Cost evaluation time: " << cost_duration.count() << " ms\n";

        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}