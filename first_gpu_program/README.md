# GPU 加速深度学习项目（制造缺陷检测）

## 📋 项目概述

本项目是一个利用 **GPU 加速**进行神经网络训练的深度学习应用，主要用于 **制造业缺陷检测**任务。项目集成了 OpenCL（开放计算语言）用于 GPU 计算加速，使用 MLP（多层感知机）网络进行二分类预测。

### 核心功能

- 🚀 **GPU 加速训练**：基于 OpenCL 的 GPU 加速计算
- 🧠 **神经网络实现**：完整的 MLP 网络前向传播和反向传播
- 📊 **灵活配置系统**：支持 INI 配置文件和命令行参数
- 📈 **性能评估**：支持 PR 曲线分析和阈值扫描
- 💾 **模型持久化**：模型权重的保存和加载功能
- 📉 **学习率调度**：支持学习率衰减策略

---

## 🎯 项目结构

```
first_gpu_program/
├── CMakeLists.txt              # CMake 构建配置
├── README.md                   # 项目文档
├── src/                        # 源代码目录
│   ├── main.cpp               # 主程序入口，命令行参数处理
│   ├── training_pipeline.cpp  # 训练流程管理
│   ├── gpu_adapter.cpp        # GPU 适配层（OpenCL 包装）
│   ├── defect_dataset.cpp     # 数据集加载和处理
│   └── mlp_network.cpp        # 多层感知机网络实现
├── include/                    # 头文件目录
│   ├── training_pipeline.h    # 训练流程接口
│   ├── gpu_adapter.h          # GPU 适配接口
│   ├── gpu_kernels.h          # GPU 核函数声明
│   ├── gpu_kernel_sources.h   # GPU 核心计算代码
│   ├── gpu_math_utils.h       # GPU 数学工具函数
│   ├── mlp_network.h          # MLP 网络接口
│   ├── defect_dataset.h       # 数据集接口
│   └── config_io.h            # 配置 I/O 接口
├── configs/                    # 配置文件目录
│   └── training_filled.ini     # 训练配置示例
├── data/                       # 数据目录
│   ├── raw/                   # 原始数据
│   │   └── manufacturing_defect_dataset.csv
│   └── processed/             # 处理后的数据
│       ├── train.csv          # 训练集
│       ├── val.csv            # 验证集
│       ├── test.csv           # 测试集
│       └── scaler.json        # 数据标准化参数
├── scripts/                    # 脚本目录
│   └── preprocess_manufacturing_defects.py  # 数据预处理脚本
└── build/                      # 构建输出目录（CMake）
```

---

## 📦 依赖安装

### 系统要求

- **操作系统**：Windows 10/11 或 Linux
- **编译器**：C++17 或更高版本
  - Windows：MSVC 2019+ 或 MinGW
  - Linux：GCC 7.0+ 或 Clang
- **GPU**：支持 OpenCL 的显卡（NVIDIA、AMD 或 Intel）

### 依赖库

#### 1. **Boost** 库（用于 Boost.Compute）

**Windows**：

```bash
# 使用 vcpkg（推荐）
vcpkg install boost:x64-windows

# 或使用 binary 包
# 从 https://www.boost.org/ 下载并设置环境变量 BOOST_ROOT
```

**Linux**：

```bash
# Ubuntu/Debian
sudo apt-get install libboost-all-dev

# Fedora
sudo dnf install boost-devel

# macOS
brew install boost
```

#### 2. **OpenCL** 库

**Windows**：

```bash
# 使用 vcpkg
vcpkg install opencl:x64-windows

# 或从 GPU 制造商官网下载驱动：
# - NVIDIA: CUDA Toolkit (包含 OpenCL)
# - AMD: ROCm 或 AMD GPU 驱动
# - Intel: Intel Graphics Compute Runtime
```

**Linux**：

```bash
# Ubuntu/Debian
sudo apt-get install opencl-headers ocl-icd-opencl-dev

# Fedora
sudo dnf install opencl-headers ocl-icd-devel

# macOS（通常已内置）
# 可选安装 Intel Graphics Compute Runtime
brew install intel-compute-runtime
```

#### 3. **CMake**（3.15+）

```bash
# Windows（使用 scoop 或 chocolatey）
choco install cmake
# 或 scoop install cmake

# Linux
sudo apt-get install cmake

# macOS
brew install cmake
```

### 安装步骤

**方式一：使用 vcpkg（推荐 Windows 用户）**

```bash
# 克隆 vcpkg
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

# 安装依赖
.\vcpkg install boost:x64-windows opencl:x64-windows
.\vcpkg integrate install

# 在 CMake 中使用
# 设置 CMAKE_TOOLCHAIN_FILE 为 vcpkg.cmake 文件
```

**方式二：手动安装（Linux）**

```bash
# Ubuntu/Debian
sudo apt-get install -y libboost-all-dev opencl-headers ocl-icd-opencl-dev cmake

# 验证安装
clinfo  # 检查 OpenCL 设备
```

---

## 🔨 构建与编译

### Windows（使用 CMake + Visual Studio）

```bash
# 进入项目目录
cd first_gpu_program

# 创建 build 目录
mkdir build
cd build

# 配置 CMake
cmake .. -G "Visual Studio 16 2019" -A x64

# 构建项目
cmake --build . --config Release

# 或使用 Visual Studio 打开生成的 .sln 文件
start first_gpu_program.sln
```

### Linux / macOS

```bash
# 进入项目目录
cd first_gpu_program

# 创建 build 目录
mkdir -p build
cd build

# 配置 CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# 构建项目
cmake --build . -j $(nproc)

# 或使用 make
make -j$(nproc)
```

### 构建输出

- **可执行文件位置**：
  - Windows：`build/Release/first_gpu_program.exe`
  - Linux/macOS：`build/first_gpu_program`

---

## ⚙️ 配置文件

### INI 配置文件格式

配置文件位置：`configs/training_filled.ini`

```ini
[training]
# 训练参数
epochs = 100
batch_size = 32
learning_rate = 0.01
loss_function = bce
backend = gpu

[model]
# 网络架构（通常从数据或硬编码确定）
input_dim = 24
hidden_layers = 2
hidden_dim = 64
output_dim = 1

[dataset]
# 数据集路径
train_path = data/processed/train.csv
val_path = data/processed/val.csv
test_path = data/processed/test.csv

[io]
# 输出配置
save_model = false
model_path = model_weights.bin
save_metrics = false
metrics_path = metrics.csv
```

### 命令行参数

运行程序时可以使用命令行参数来覆盖配置文件设置：

```bash
./first_gpu_program [选项]
```

#### 主要参数

| 参数                       | 说明                   | 示例                                     |
| -------------------------- | ---------------------- | ---------------------------------------- |
| `--config <path>`        | 导入训练配置 INI 文件  | `--config configs/training_filled.ini` |
| `--export-config <path>` | 导出合并后的配置到 INI | `--export-config output.ini`           |
| `--export-only`          | 仅导出配置，不执行训练 | 无参数                                   |
| `--dataset <path>`       | 数据集 CSV 路径        | `--dataset data/processed/train.csv`   |
| `--backend <cpu\|gpu>`    | 执行后端（默认：gpu）  | `--backend gpu`                        |
| `--load-model <path>`    | 加载模型权重           | `--load-model weights.bin`             |
| `--save-model <path>`    | 保存模型权重           | `--save-model output_weights.bin`      |
| `--epochs <int>`         | 训练轮次（默认：20）   | `--epochs 50`                          |
| `--batch-size <int>`     | 批次大小（默认：1）    | `--batch-size 32`                      |
| `--lr <float>`           | 学习率（默认：0.01）   | `--lr 0.001`                           |
| `--loss <bce\|mse>`       | 损失函数（默认：bce）  | `--loss bce`                           |
| `--threshold <float>`    | 分类阈值（默认：0.5）  | `--threshold 0.6`                      |
| `--results-csv <path>`   | 保存每个 epoch 的指标  | `--results-csv metrics.csv`            |
| `--pr-csv <path>`        | 保存 PR 曲线扫描结果   | `--pr-csv pr_curve.csv`                |
| `--auto-class-weights`   | 自动计算 BCE 类权重    | 无参数                                   |
| `--eval-only`            | 仅评估，跳过训练       | 无参数                                   |
| `--help`                 | 显示帮助信息           | 无参数                                   |

---

## 🚀 运行示例

### 1. 基础训练（使用默认参数）

```bash
cd first_gpu_program
./build/first_gpu_program
```

### 2. 使用配置文件训练

```bash
./build/first_gpu_program --config configs/training_filled.ini
```

### 3. 自定义参数训练

```bash
./build/first_gpu_program \
  --dataset data/processed/train.csv \
  --backend gpu \
  --epochs 50 \
  --batch-size 32 \
  --lr 0.001 \
  --save-model trained_model.bin \
  --results-csv training_metrics.csv
```

### 4. 使用保存的模型进行评估

```bash
./build/first_gpu_program \
  --load-model trained_model.bin \
  --dataset data/processed/test.csv \
  --eval-only \
  --pr-csv pr_curve_results.csv
```

### 5. 导出配置文件

```bash
./build/first_gpu_program \
  --config configs/training_filled.ini \
  --export-config merged_config.ini \
  --export-only
```

### 6. CPU 后端训练（调试用）

```bash
./build/first_gpu_program \
  --backend cpu \
  --epochs 10 \
  --batch-size 16
```

---

## 📊 数据预处理

项目包含数据预处理脚本用于从原始数据生成训练/验证/测试集。

### 预处理脚本

**位置**：`scripts/preprocess_manufacturing_defects.py`

**依赖**：

```bash
pip install pandas scikit-learn numpy
```

**运行方式**：

```bash
python scripts/preprocess_manufacturing_defects.py \
  --input data/raw/manufacturing_defect_dataset.csv \
  --output-dir data/processed/ \
  --train-ratio 0.7 \
  --val-ratio 0.15
```

**输出文件**：

- `data/processed/train.csv` - 训练集（70%）
- `data/processed/val.csv` - 验证集（15%）
- `data/processed/test.csv` - 测试集（15%）
- `data/processed/scaler.json` - 标准化参数（用于推理时的特征缩放）

---

## 🎓 使用工作流

### 工作流 1：完整训练管道

```bash
# 1. 数据预处理
python scripts/preprocess_manufacturing_defects.py

# 2. 编译项目
cd first_gpu_program
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
cd ..

# 3. 执行训练
./build/first_gpu_program \
  --backend gpu \
  --epochs 100 \
  --batch-size 32 \
  --lr 0.01 \
  --save-model model_final.bin \
  --results-csv training_log.csv

# 4. 评估模型
./build/first_gpu_program \
  --load-model model_final.bin \
  --eval-only \
  --pr-csv pr_curve.csv
```

### 工作流 2：超参数调优

```bash
# 尝试不同的学习率
for lr in 0.001 0.01 0.1; do
  ./build/first_gpu_program \
    --lr $lr \
    --epochs 50 \
    --save-model model_lr_${lr}.bin \
    --results-csv metrics_lr_${lr}.csv
done
```

### 工作流 3：迁移学习

```bash
# 加载预训练模型，继续训练
./build/first_gpu_program \
  --load-model pretrained_model.bin \
  --epochs 20 \
  --lr 0.001 \
  --save-model finetuned_model.bin
```

---

## 📈 输出说明

### 标准输出

训练过程中会打印每个 epoch 的统计信息：

```
Epoch 1/100 - Loss: 0.6234 | Train Acc: 0.62 | Val Acc: 0.60
Epoch 2/100 - Loss: 0.5891 | Train Acc: 0.68 | Val Acc: 0.65
...
Training completed in 45.23 seconds
```

### CSV 输出文件

**训练指标（--results-csv）**：

```csv
epoch,train_loss,train_accuracy,val_loss,val_accuracy
1,0.6234,0.62,0.6189,0.60
2,0.5891,0.68,0.5756,0.65
...
```

**PR 曲线（--pr-csv）**：

```csv
threshold,precision,recall,f1_score
0.1,0.45,0.95,0.61
0.2,0.52,0.88,0.65
...
```

---

## 🐛 常见问题

### Q1：编译错误 - 找不到 Boost 库

**解决方案**：

```bash
# Windows 用户（设置 CMake 变量）
cmake .. -G "Visual Studio 16 2019" -DBOOST_ROOT=C:\path\to\boost

# Linux 用户
sudo apt-get install libboost-all-dev
```

### Q2：运行错误 - 找不到 OpenCL 设备

**解决方案**：

- 检查 GPU 驱动是否安装：
  ```bash
  clinfo  # 列出可用的 OpenCL 设备
  ```
- 安装最新的 GPU 驱动程序
- 如果无可用 GPU，使用 `--backend cpu` 切换到 CPU 模式

### Q3：数据集路径错误

**解决方案**：

- 确保数据文件存在于指定路径
- 使用绝对路径而不是相对路径
- 检查文件权限

### Q4：模型训练很慢

**原因与解决方案**：

- 检查是否正确使用 GPU：`--backend gpu`
- 增加批次大小：`--batch-size 64`
- 使用 Release 构建而非 Debug 构建
- 确认 GPU 驱动和 OpenCL 环境正确配置

### Q5：导出配置文件失败

**解决方案**：

- 检查输出目录是否存在且可写
- 使用绝对路径作为输出位置
- 确保足够的磁盘空间

---

## 🔧 开发与调试

### 使用 VS Code 调试

**launch.json 配置示例**：

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "C++ Debug",
      "type": "cppdbg",
      "request": "launch",
      "program": "${workspaceFolder}/build/first_gpu_program",
      "args": ["--backend", "cpu", "--epochs", "5"],
      "stopAtEntry": false,
      "cwd": "${workspaceFolder}",
      "environment": [],
      "externalConsole": false,
      "MIMode": "gdb",
      "setupCommands": [
        {
          "description": "为 gdb 启用整洁打印",
          "text": "-enable-pretty-printing",
          "ignoreFailures": true
        }
      ],
      "preLaunchTask": "C/C++: 构建活动文件"
    }
  ]
}
```

### 构建模式

- **Debug 构建**（带调试符号）：

  ```bash
  cmake .. -DCMAKE_BUILD_TYPE=Debug
  ```
- **Release 构建**（优化性能）：

  ```bash
  cmake .. -DCMAKE_BUILD_TYPE=Release
  ```

---

## 📚 核心代码模块说明

### 1. **gpu_adapter.h/cpp** - GPU 适配层

处理 OpenCL 初始化、设备查询、内存管理和核函数调用。

### 2. **mlp_network.h/cpp** - 神经网络实现

实现 MLP 的前向传播、反向传播和权重更新。

### 3. **defect_dataset.h/cpp** - 数据集管理

加载 CSV 数据，处理数据归一化和批处理。

### 4. **training_pipeline.h/cpp** - 训练流程

协调数据加载、训练循环、评估和模型保存。

### 5. **gpu_kernels.h / gpu_kernel_sources.h** - GPU 核函数

包含用 OpenCL C 编写的并行计算核函数代码。

---

## 📝 许可证与贡献

本项目为学习用途。欢迎提交问题和改进建议。

---

## 📞 联系与支持

如有问题或建议，请检查以下资源：

- **NVIDIA OpenCL 文档**：https://www.khronos.org/opencl/
- **Boost.Compute 文档**：https://www.boost.org/doc/libs/release/libs/compute/
- **CMake 文档**：https://cmake.org/documentation/

---

**最后更新**：2026 年

**项目版本**：v0.1.0
