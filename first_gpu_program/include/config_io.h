#pragma once

#include "training_pipeline.h"

#include <filesystem>
#include <string>

void load_training_config_file(const std::filesystem::path &path, TrainingConfig &config);
void save_training_config_file(const std::filesystem::path &path, const TrainingConfig &config);
