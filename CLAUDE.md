# whisper.cpp 项目架构文档

## 1. 整体目录结构

```
whisper.cpp/
├── include/           # 公共头文件
│   └── whisper.h      # 主头文件 - C API定义
├── src/               # 核心源代码
│   ├── whisper.cpp    # 主要实现文件 (约330KB)
│   ├── whisper-arch.h # 模型架构定义
│   ├── coreml/        # CoreML后端支持
│   └── openvino/      # OpenVINO后端支持
├── ggml/              # GGML机器学习库 (子模块)
│   ├── include/       # GGML头文件
│   ├── src/           # GGML源文件
│   └── cmake/         # GGML CMake配置
├── examples/          # 示例程序集合
├── bindings/          # 多语言绑定
│   ├── go/
│   ├── java/
│   ├── javascript/
│   └── ruby/
├── tests/             # 测试代码和测试数据
├── cmake/             # CMake构建配置文件
├── scripts/           # 辅助脚本
├── models/            # 模型文件目录
├── samples/           # 音频样本
├── CMakeLists.txt    # 主构建文件
└── Makefile          # 简化构建Makefile
```

## 2. 主要模块和职责

### 核心模块 (src/)

**whisper.cpp** - 主要实现文件，包含以下关键结构：

- **whisper_mel**: Mel频谱数据结构
- **whisper_filters**: 音频滤波器配置
- **whisper_vocab**: 词汇表和标记映射
- **whisper_segment**: 转录结果片段
- **whisper_batch**: 批处理支持
- **whisper_hparams**: 模型超参数
- **whisper_layer_encoder**: 编码器层
- **whisper_layer_decoder**: 解码器层
- **whisper_model**: 模型结构
- **whisper_context**: 推理上下文
- **whisper_state**: 推理状态
- **whisper_vad_context**: 语音活动检测(VAD)

### GGML库 (ggml/)

作为核心依赖库，提供：

- **ggml.h**: 基础张量操作
- **ggml-cpu.h**: CPU后端
- **ggml-cuda.h**: NVIDIA CUDA GPU支持
- **ggml-metal.h**: Apple Metal GPU支持
- **ggml-vulkan.h**: Vulkan GPU支持
- **ggml-opencl.h**: OpenCL支持
- **ggml-sycl.h**: Intel SYCL支持

### 示例程序 (examples/)

- **cli/**: 命令行转录工具 (whisper-cli)
- **stream/**: 实时流式转录
- **server/**: HTTP服务器 (REST API)
- **bench/**: 性能基准测试
- **command/**: 语音命令识别
- **quantize/**: 模型量化工具
- **whisper.wasm/**: WebAssembly版本
- **talk-llama/**: 与LLM集成

## 3. 核心源代码文件位置和功能

### 头文件 (include/)

**whisper.h** - 公共C API接口，包含：
- 上下文初始化函数 (whisper_init_*)
- 音频处理函数 (whisper_pcm_to_mel)
- 编码/解码函数 (whisper_encode, whisper_decode)
- 完整转录函数 (whisper_full)
- 语音活动检测 (whisper_vad_*)
- 语言检测 (whisper_lang_*)

### 实现文件 (src/)

**whisper.cpp** - 约330KB的单文件实现：
- 模型加载和初始化
- Mel频谱转换
- 编码器推理
- 解码器推理
- 采样策略实现
- 语法约束生成
- VAD实现

**whisper-arch.h** - 模型架构定义：
- 张量名称映射
- 编码器/解码器结构

## 4. 示例程序结构

### 通用模块 (examples/common.*)

所有示例共享的工具：
- **common.h/cpp**: 通用工具函数
- **common-ggml.h/cpp**: GGML相关工具
- **common-whisper.h/cpp**: Whisper特定工具
- **grammar-parser.h/cpp**: 语法解析器

### 典型示例工作流程

1. **加载模型**: whisper_init_from_file()
2. **准备音频**: whisper_pcm_to_mel() 或 whisper_set_mel()
3. **执行推理**: whisper_full() 或 whisper_encode() + whisper_decode()
4. **获取结果**: whisper_full_get_segment_text()

## 5. 构建系统配置

### CMake构建 (CMakeLists.txt)

主要构建选项：

```cmake
# 计算后端选项
WHISPER_USE_SYSTEM_GGML  # 使用系统GGML库
GGML_CUDA                 # NVIDIA CUDA支持
GGML_METAL                # Apple Metal支持
GGML_VULKAN               # Vulkan GPU支持
GGML_OPENCL               # OpenCL支持
GGML_SYCL                 # Intel SYCL支持

# 特殊支持
WHISPER_COREML            # CoreML支持
WHISPER_OPENVINO          # OpenVINO支持
WHISPER_FFMPEG            # FFmpeg支持

# 构建选项
WHISPER_BUILD_TESTS       # 构建测试
WHISPER_BUILD_EXAMPLES    # 构建示例
WHISPER_BUILD_SERVER      # 构建服务器
```

### 构建命令

```bash
# CMake构建
cmake -B build
cmake --build build -j --config Release

# Makefile简化构建
make base.en  # 下载模型+构建+测试
```

### 平台特定配置 (cmake/)

- arm64-apple-clang.cmake (Apple Silicon)
- arm64-windows-llvm.cmake (Windows ARM)
- x64-windows-llvm.cmake (Windows x64)
- riscv64-spacemit-linux-gnu-gcc.cmake (RISC-V)

## 6. 头文件和实现文件的组织方式

### 组织模式

1. **单一主头文件**: include/whisper.h
   - C语言风格的API
   - 完整的函数声明
   - 文档注释

2. **单一主实现文件**: src/whisper.cpp
   - 包含所有实现细节
   - 命名空间: 匿名命名空间用于内部函数
   - 结构体定义和函数实现

3. **模块化子目录**
   - src/coreml/: CoreML后端
   - src/openvino/: OpenVINO后端

4. **GGML依赖**
   - 作为子目录集成
   - 通过CMake的add_subdirectory引入

5. **示例代码共享**
   - 公共库 (common.*) 提供复用代码
   - 静态库链接方式

### 语言绑定 (bindings/)

- **JavaScript/TypeScript**: Node.js和Web环境
- **Go**: Golang绑定
- **Java**: Android和桌面Java
- **Ruby**: Ruby绑定

## 总结

whisper.cpp 是一个高度模块化但实现集中的项目：

- **核心哲学**: 单一高效的实现文件 + 清晰的C API
- **平台支持**: 多后端 (CPU, CUDA, Metal, Vulkan, OpenCL, SYCL, CoreML, OpenVINO)
- **易于集成**: 简洁的C API设计，零运行时内存分配
- **丰富的示例**: 从简单CLI到复杂服务器应用
- **多语言支持**: 通过官方绑定支持主流编程语言

这个架构使得whisper.cpp既适合作为嵌入式库集成到各种应用中，也能作为独立的语音识别工具使用。
