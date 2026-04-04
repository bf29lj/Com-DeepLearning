#include "defect_dataset.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {

std::vector<std::string> split_csv_line(const std::string &line) {
    std::vector<std::string> result;
    std::stringstream stream(line);
    std::string cell;
    while (std::getline(stream, cell, ',')) {
        result.push_back(cell);
    }
    return result;
}

float parse_float(const std::string &text, std::size_t row_index, const std::string &column_name) {
    try {
        return std::stof(text);
    } catch (const std::exception &) {
        throw std::runtime_error("Invalid numeric value at row " + std::to_string(row_index + 1) +
                                 ", column '" + column_name + "': " + text);
    }
}

}  // namespace

ManufacturingDefectDataset ManufacturingDefectDataset::load_csv(const std::filesystem::path &csv_path) {
    std::ifstream input(csv_path);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open dataset: " + csv_path.string());
    }

    std::string header_line;
    if (!std::getline(input, header_line)) {
        throw std::runtime_error("Dataset is empty: " + csv_path.string());
    }

    const std::vector<std::string> header = split_csv_line(header_line);
    if (header.size() < 2) {
        throw std::runtime_error("Dataset must contain at least one feature and one label column");
    }

    ManufacturingDefectDataset dataset;
    dataset.feature_names_.assign(header.begin(), header.end() - 1);

    std::string line;
    std::size_t row_index = 0;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }

        const std::vector<std::string> cells = split_csv_line(line);
        if (cells.size() != header.size()) {
            throw std::runtime_error("CSV column count mismatch at row " + std::to_string(row_index + 1));
        }

        DefectSample sample;
        sample.features.reserve(header.size() - 1);
        for (std::size_t column = 0; column + 1 < header.size(); ++column) {
            sample.features.push_back(parse_float(cells[column], row_index, header[column]));
        }
        sample.label = static_cast<uint8_t>(parse_float(cells.back(), row_index, header.back()));
        dataset.samples_.push_back(std::move(sample));
        ++row_index;
    }

    if (dataset.samples_.empty()) {
        throw std::runtime_error("Dataset contains no samples: " + csv_path.string());
    }

    return dataset;
}