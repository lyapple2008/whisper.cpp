# whisper.cpp CLI 学习笔记

> 记录阅读 `examples/cli/cli.cpp` 过程中遇到的关键函数、机制与设计细节。
> 每条主题独立成节，统一在下方「📑 目录」中建立索引。

---

## 📑 目录

| 主题 | 源码位置 | 状态 |
|------|---------|------|
| [ggml_backend_load_all() 是什么作用](#ggml_backend_load_all-是什么作用) | `examples/cli/cli.cpp:964` | ✅ 已整理 |
| [文件形式的参数获取机制（@rsp file）](#文件形式的参数获取机制rsp-file) | `examples/cli/cli.cpp:975` | ✅ 已整理 |
| [whisper.cpp 模型加载](whisper.cpp模型加载.html) | `src/whisper.cpp` | ✅ 已整理 |
| [音频对齐 DTW](#音频对齐-dtw) | TBD | ⏳ 待整理 |
| [Flash Attention 是什么](#flash-attention-是什么) | TBD | ⏳ 待研究 |

---

<a id="ggml_backend_load_all-是什么作用"></a>

## ggml_backend_load_all() 是什么作用

### 一句话总结

`ggml_backend_load_all()` 是 GGML 的**运行时后端插件加载器**——在程序启动时，尝试通过 `dlopen` / `LoadLibrary` 动态加载系统中所有可用的计算后端（CUDA、Metal、Vulkan、CPU 等），把它们注册到全局的 backend registry 中，供后续推理使用。

源码位置：`ggml/src/ggml-backend-reg.cpp:555-557`

```cpp
// ggml/src/ggml-backend-reg.cpp
void ggml_backend_load_all() {
    ggml_backend_load_all_from_path(nullptr);
}
```

它本身只是个壳，真正的工作在 `ggml_backend_load_all_from_path`：

```cpp
void ggml_backend_load_all_from_path(const char * dir_path) {
    // ...
    // 硬编码遍历 15 种已知后端，每种都尝试动态加载
    ggml_backend_load_best("blas",     silent, dir_path);
    ggml_backend_load_best("zendnn",   silent, dir_path);
    ggml_backend_load_best("cann",     silent, dir_path);
    ggml_backend_load_best("cuda",     silent, dir_path);
    ggml_backend_load_best("hip",      silent, dir_path);
    ggml_backend_load_best("metal",    silent, dir_path);
    ggml_backend_load_best("rpc",      silent, dir_path);
    ggml_backend_load_best("sycl",     silent, dir_path);
    ggml_backend_load_best("vulkan",   silent, dir_path);
    ggml_backend_load_best("virtgpu",  silent, dir_path);
    ggml_backend_load_best("opencl",   silent, dir_path);
    ggml_backend_load_best("hexagon",  silent, dir_path);
    ggml_backend_load_best("musa",     silent, dir_path);
    ggml_backend_load_best("openvino", silent, dir_path);
    ggml_backend_load_best("cpu",      silent, dir_path);

    // 环境变量 GGML_BACKEND_PATH：用于加载额外的第三方后端
    const char * backend_path = std::getenv("GGML_BACKEND_PATH");
    if (backend_path) {
        ggml_backend_load(backend_path);
    }
}
```

### 它具体在干什么

#### 1. 动态库扫描与加载

`ggml_backend_load_best(name, ...)` 按下面这套**文件命名规则**在多个目录里找匹配的动态库：

| 平台 | 文件名模式 | 示例 |
|------|----------|------|
| Windows | `ggml-<name>-*.dll` | `ggml-cuda-12.dll` |
| Linux / macOS | `libggml-<name>-*.so` | `libggml-cuda.so` |

搜索路径按顺序：
1. 编译时定义的 `GGML_BACKEND_DIR` 宏
2. 可执行文件所在目录
3. 当前工作目录

#### 2. 底层用 `dlopen` / `LoadLibrary`

文件：`ggml/src/ggml-backend-dl.cpp`

```cpp
#ifdef _WIN32
dl_handle * dl_load_library(const fs::path & path) {
    HMODULE handle = LoadLibraryW(path.wstring().c_str());
    return handle;
}
void * dl_get_sym(dl_handle * handle, const char * name) {
    return (void *) GetProcAddress(handle, name);
}
#else
dl_handle * dl_load_library(const fs::path & path) {
    return dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
}
void * dl_get_sym(dl_handle * handle, const char * name) {
    return dlsym(handle, name);
}
#endif
```

#### 3. 验证 ABI 并注册

加载每个候选库后，`load_backend` 会通过 `dlsym` / `GetProcAddress` 解析两个必须的符号（见 `ggml/src/ggml-backend-impl.h:235-265`）：

- **`ggml_backend_score()`** — 返回 0 表示"本机不支持，跳过"；数值越大表示该实现越优
- **`ggml_backend_init()`** — 返回一个 `ggml_backend_reg_t`，其 `api_version` 必须等于 `GGML_BACKEND_API_VERSION`（当前值为 2）

通过校验后，后端被加入全局 registry，后续 `whisper_init_from_file_with_params()` 就能在创建 context 时把模型权重分配到对应的设备上。

### 在 whisper.cpp CLI 中的位置

`examples/cli/cli.cpp:964` 把它放在 `main()` 的**第一行**，先于任何 `whisper_*` 调用：

```cpp
int main(int argc, char ** argv) {
    // mars-todo: 这里是做什么的？
    ggml_backend_load_all();      // ← 在这里
    // ... 解析参数 ...
    struct whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);
    // ...
}
```

这种放置是有意为之的——`whisper_init_from_file_with_params` 内部要根据 `cparams.use_gpu` / `gpu_device` 等参数选择一个 backend 来加载模型张量。`ggml_backend_load_all()` **必须**在 context 创建前完成所有 backend 的注册。

### 关键注意点

#### 不是所有编译配置下都有效

| 编译配置 | 实际行为 |
|---------|---------|
| `GGML_BACKEND_DL=ON` + 后端构建为 `MODULE` 库 | **真正发挥作用**——后端只以动态库形式存在，必须靠它 dlopen |
| `GGML_BACKEND_DL=OFF`（默认） | 大部分退化为空操作——找不到 `.so` / `.dll`，但静态链接进来的后端由全局构造函数直接注册，不依赖此函数 |
| 编译时未启用任何 `GGML_USE_*` 宏 | 几乎完全空操作 |

#### 静态注册的并行路径

`ggml-backend-reg.cpp:111-167` 的 `ggml_backend_registry` 构造函数会根据编译期宏把已链接进二进制的后端注册到同一份 registry 中：

```cpp
ggml_backend_registry() {
#ifdef GGML_USE_CUDA
    register_backend(ggml_backend_cuda_reg());
#endif
#ifdef GGML_USE_METAL
    register_backend(ggml_backend_metal_reg());
#endif
    // ... Vulkan / SYCL / OpenCL / BLAS / CPU 等
}
```

而 `GGML_DISABLE_VULKAN` 这个环境变量**只**影响静态注册路径，**不会**让 `ggml_backend_load_all()` 跳过 Vulkan。

#### 自定义后端支持

通过设置 `GGML_BACKEND_PATH=/path/to/your/libggml-custom.so` 可以在不重新编译的情况下加载第三方后端——这是这个 API 在设计上预留的扩展点。

### 一句话回答

`ggml_backend_load_all()` 在程序启动时尝试 `dlopen` 一组硬编码的后端动态库（CUDA / Metal / Vulkan / CPU 等），把找到的、ABI 兼容的、硬件支持的实现注册到 GGML 的后端注册表里，供 `whisper_init_*` 选用。**它必须放在 `main()` 开头、在创建 whisper context 之前调用**——在 macOS 项目里它可能确实是空操作（取决于 CMake 时是否开了 `GGML_BACKEND_DL`），但保留它能保证在不同构建配置下都正确初始化。

---

<a id="文件形式的参数获取机制rsp-file"></a>

## 文件形式的参数获取机制（@rsp file）

### 一句话总结

`@rsp file` 是 whisper.cpp CLI 实现的一个**响应文件（response file）机制**——当用户以 `@args.txt` 形式传入单个参数时，程序会按行读取该文件，把每行内容当作一个独立的命令行参数，重新构造 `argv` 后再交给 `whisper_params_parse()` 解析，从而绕过 shell 命令行长度的限制。

源码位置：`examples/cli/cli.cpp:975-1002`

```cpp
// examples/cli/cli.cpp:975-1002
// If the only argument starts with "@", read arguments line-by-line
// from the given file.
std::vector<std::string> vec_args;
if (argc == 2 && argv != nullptr && argv[1] != nullptr && argv[1][0] == '@') {
    // Save the name of the executable.
    vec_args.push_back(argv[0]);

    // Open the response file.
    char const * rspfile = argv[1] + sizeof(char);
    std::ifstream fin(rspfile);
    if (fin.is_open() == false) {
        fprintf(stderr, "error: response file '%s' not found\n", rspfile);
        return 1;
    }

    // Read the entire response file.
    std::string line;
    while (std::getline(fin, line)) {
        vec_args.push_back(line);
    }

    // Use the contents of the response file as the command-line arguments.
    argc = static_cast<int>(vec_args.size());
    argv = static_cast<char **>(alloca(argc * sizeof (char *)));
    for (int i = 0; i < argc; ++i) {
        argv[i] = const_cast<char *>(vec_args[i].c_str());
    }
}
```

### 触发条件与执行流程

#### 1. 严格的条件判断

```cpp
argc == 2 && argv != nullptr && argv[1] != nullptr && argv[1][0] == '@'
```

四个条件缺一不可：

| 条件 | 含义 |
|------|------|
| `argc == 2` | 必须是"程序 + 一个 `@` 参数"的形式。多个普通参数 + 一个 `@xxx` 不触发此机制 |
| `argv != nullptr` | 防御性检查 |
| `argv[1] != nullptr` | 防御性检查（理论上 `argv[argc]` 必为 `nullptr`，但显式判断更稳） |
| `argv[1][0] == '@'` | 参数首字符必须是 `@` |

> 这意味着 `./whisper-cli -f @somefile.wav` 不会被识别为 rsp 文件，`@somefile.wav` 会被当作 `-f` 的值原样传入。

#### 2. 跳过 `@` 符号

```cpp
char const * rspfile = argv[1] + sizeof(char);
```

- `sizeof(char)` 永远等于 1，写法上等价于 `argv[1] + 1`
- 目的是跳过首字符 `@`，拿到真正的文件路径
- 例如 `@args.txt` 会被处理成 `args.txt`

#### 3. 逐行读取

```cpp
std::string line;
while (std::getline(fin, line)) {
    vec_args.push_back(line);
}
```

- `std::getline` 会**自动剥离换行符**
- 文件中**每一行就是一条独立的命令行参数**
- 没有 token 切分、没有引号处理——按行整行原样塞入

#### 4. 重建 argv 数组

```cpp
argc = static_cast<int>(vec_args.size());
argv = static_cast<char **>(alloca(argc * sizeof (char *)));
for (int i = 0; i < argc; ++i) {
    argv[i] = const_cast<char *>(vec_args[i].c_str());
}
```

- `vec_args[0]` 保留为原始的 `argv[0]`（程序名），保持 `main` 语义一致
- `alloca` 在**栈上分配**指针数组，函数返回时自动释放
- 用 `const_cast` 去掉 `c_str()` 的 `const`——因为 `argv` 类型是 `char**`

**关键设计**：重建后**就地覆盖**了 `main` 的实参 `argc` 和 `argv`，后续 `whisper_params_parse(argc, argv, params)` 就能像处理普通命令行参数一样处理文件内容——**对后面的解析代码完全透明、零侵入**。

### 使用示例

写一个 `args.txt`：

```
-m
models/ggml-base.en.bin
-f
samples/jfk.wav
-l
auto
```

执行：

```bash
./whisper-cli @args.txt
```

等价于：

```bash
./whisper-cli -m models/ggml-base.en.bin -f samples/jfk.wav -l auto
```

### 为什么需要这个机制

#### 1. 绕过 shell 命令行长度的限制

| 平台 | 限制 |
|------|------|
| Windows `CreateProcess` | 约 32K 字符 |
| Linux `ARG_MAX` | 通常 2MB 左右（具体看实现） |
| macOS | 受 `kern.argmax` sysctl 限制 |

长音频批量处理时，参数拼接可能轻松突破这些限制。

#### 2. 调用方式可持久化

把常用的参数组合存到 rsp 文件里：

- 方便版本管理（rsp 文件可入库）
- 方便 CI 自动化（脚本里只需要 `whisper-cli @prod.rsp`）
- 方便不同场景切换（`@dev.rsp` / `@prod.rsp`）

#### 3. 批量处理友好

一行一个音频文件路径放进 rsp 文件，简化调用脚本——与 `examples/cli/cli.cpp` 现有的"多文件输入"机制（`std::vector<std::string> fname_inp`）天然契合。

#### 4. 与主流工具对齐

这是编译器领域的事实标准：

- MSVC `cl.exe @args.rsp`
- GCC `gcc @args.txt`
- Clang `clang @args.txt`
- Rust `rustc @args`
- CMake `cmake -E environment` 链路

whisper.cpp 引入这个机制后，与上述工具链的"调用约定"对齐，用户迁移成本极低。

### 关键注意点

#### 不会跳过空行/注释

- `std::getline` 不会自动跳过空行
- 文件里**没有 `#` 注释识别机制**——`# xxx` 会被当作一个以 `#` 开头的参数传给 `whisper_params_parse`，最终在 `else` 分支被当作"unknown argument"报错
- 每行必须写**一个完整的 token**；带空格的路径整行原样传入，rsp 文件内部不做引号转义

#### alloca 的使用安全性

- 在 `main` 顶部使用一次，函数退出自动回收，**无内存泄漏**
- `alloca` 在循环中反复使用会持续占用栈空间，但这里不会
- `vec_args` 中每个 `std::string` 内部走堆分配，**栈上只有指针数组**

#### ABI 兼容性 / 生命周期

- 重建后的 `argv[i]` 指向 `vec_args[i]` 的内部 buffer
- `vec_args` 在 if block 结束后**依然存活**——因为它声明在 `if` 之外，整个 `main` 作用域内都可见
- `whisper_params_parse` 调用前 `vec_args` 不会被析构，所以**指针有效**

#### 行为对比：与 `-f` 参数的差别

| 调用形式 | 解析结果 |
|---------|---------|
| `./whisper-cli -f audio.wav` | `params.fname_inp` 包含 `audio.wav` |
| `./whisper-cli @audio.wav` | 文件不存在则报错退出；存在则按行展开 |

**注意**：`@audio.wav` 在这里**不是**"读取 audio.wav 作为输入"——它读取的是**参数文件**，不是音频文件。这是从 MSVC rsp 机制沿用过来的语义，初次使用者容易混淆。

### 一句话回答

`@rsp file` 机制让 whisper.cpp CLI 接受 `@filename` 形式的单个参数，自动从文件中按行读取参数并重建 argv——**对后续参数解析代码完全透明**，仅在 `argc == 2` 且参数以 `@` 开头时触发，是一种轻量级的"长命令行参数"解决方案，与主流编译器（MSVC / GCC / Clang）的 rsp 机制语义一致。

---

<a id="音频对齐-dtw"></a>

## 音频对齐 DTW

> ⏳ **TODO**：待整理

**需要调研的问题**：
- whisper.cpp 中 DTW（Dynamic Time Warping）在哪里实现？
- CLI 通过哪些参数开启音频对齐（如 token / word 时间戳）？
- 输出格式（segment 时间戳、word 时间戳）与原始 token 序列的对应关系
- 与 `--diarize`、`--tinydiarize` 等参数的关系

---

<a id="flash-attention-是什么"></a>

## Flash Attention 是什么

> ⏳ **TODO**：待研究

**需要调研的问题**：
- **本质定义**：Flash Attention 是什么？由谁提出（论文 Tri Dao et al., 2022）？核心思想一句话概括
- **解决的问题**：
  - 标准 Attention 的时间复杂度 `O(N²)` 和空间复杂度 `O(N²)` 在长序列上为什么成为瓶颈？
  - 显存瓶颈（HBM 带宽墙）具体表现：attention 实际是 memory-bound 还是 compute-bound？
  - 训练大模型时 attention 占了多少显存/时间？
- **相对原始 Attention 的修改**：
  - **Tiling（分块）**：把 Q/K/V 分成小块（block），在 SRAM/片上缓存中完成完整的 QKᵀ → softmax → PV 流程，避免写回 HBM
  - **Recomputation（重计算）**：前向时不存储完整的 `N×N` attention matrix，反向时按 block 重新计算——用算力换显存
  - **数值稳定**：如何在分块 softmax 中避免 `exp` 溢出（在线 softmax / 维护 `m`、`l` 统计量）？
  - **IO 感知**：HBM 读写的次数从 `O(N²)` 降到 `O(N)` 的推导细节
- **关键变体**：
  - Flash Attention v1 / v2 / v3 各自的改进点（v2 引入 work partitioning、causal masking 优化；v3 引入 async warp specialization、FP8）
  - 与其他高效 attention（Paged Attention、Linear Attention、Multi-Query Attention）的区别
- **在 whisper.cpp / GGML 中的体现**：
  - ggml 中是否启用了 Flash Attention？对应编译宏/参数是什么（如 `-DGGML_USE_FLASH_ATTN`、CLI 是否有 `--flash-attn` 开关）？
  - Whisper 的 encoder 长度（1500 个 mel 帧）和 decoder 长度典型场景下，Flash Attention 能带来多大收益？
  - Metal / CUDA / CPU 后端对 Flash Attention 的支持现状

---

<!--
📝 添加新主题的模板：

## [主题标题]

### 一句话总结
...

### ... 小节（自由组织）

---

记得同时在「📑 目录」表格中新增一行。
-->
