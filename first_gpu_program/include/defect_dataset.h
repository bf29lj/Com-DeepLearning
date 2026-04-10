#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

struct DefectSample {
    // One preprocessed 1D feature vector.
    std::vector<float> features;
    // Binary label: 0 = low defect, 1 = high defect.
    uint8_t label = 0;
};

// CSV dataset loader for processed manufacturing defect splits.
// Expected format: feature columns first, label column last.
class ManufacturingDefectDataset {
public:
    ManufacturingDefectDataset() = default;
    using ProgressCallback = std::function<void(std::size_t, std::size_t)>;

    // Loads all rows from a processed CSV file into memory.
    static ManufacturingDefectDataset load_csv(const std::filesystem::path &csv_path,
                                               const ProgressCallback &progress_callback = {});

    const std::vector<std::string> &feature_names() const { return feature_names_; }
    std::size_t feature_count() const { return feature_names_.size(); }
    std::size_t size() const { return samples_.size(); }

    const DefectSample &sample(std::size_t index) const { return samples_.at(index); }
    const std::vector<DefectSample> &samples() const { return samples_; }

private:
    std::vector<std::string> feature_names_;
    std::vector<DefectSample> samples_;
};