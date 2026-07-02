# Whisper Mel Filter 系数：生成、保存与推理使用

本文档说明 OpenAI Whisper 及 whisper.cpp 中 **Mel 滤波器组（Mel filterbank）** 系数的完整生命周期：离线如何生成、如何持久化、推理时如何加载并在 Log Mel Spectrogram 计算中使用。

---

## 1. 概述

Whisper 不直接在时域波形上运行 Transformer，而是先将 PCM 转为 **Log Mel Spectrogram**。其中 Mel 频带投影依赖一个固定的 **滤波器矩阵**：

```
功率谱 P[k]  (k = 0 … n_fft-1)
    ↓  矩阵乘法（Mel filterbank）
Mel 能量 S[j]  (j = 0 … n_mels-1)
    ↓  log₁₀ + 动态范围归一化
Log Mel Spectrogram → Encoder 输入
```

Mel 滤波器系数具有以下特点：

- **离线预计算**，不在推理时动态生成
- **与 STFT 参数绑定**：`sr=16000`, `n_fft=400`, `n_mels=80` 或 `128`
- **随模型版本变化**：large-v3 起 `n_mels` 从 80 增至 128
- **量化不影响**：Q5_0 等量化只作用于 Transformer 权重，Mel 滤波器始终为 float32

---

## 2. 系数生成（离线）

### 2.1 生成工具：librosa

OpenAI 官方使用 [librosa](https://librosa.org/) 的 `librosa.filters.mel()` 生成三角 Mel 滤波器组。官方仓库 **不包含** 独立的生成脚本，仅在 `whisper/audio.py` 的 docstring 中记录了生成方式：

```python
import librosa
import numpy as np

np.savez_compressed(
    "mel_filters.npz",
    mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
    mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
)
```

### 2.2 参数含义

| 参数 | 值 | 对应 Whisper 常量 |
|------|-----|-------------------|
| `sr` | 16000 | `SAMPLE_RATE` / `WHISPER_SAMPLE_RATE` |
| `n_fft` | 400 | `N_FFT` / `WHISPER_N_FFT` |
| `n_mels` | 80 或 128 | `model.dims.n_mels` / `hparams.n_mels` |

### 2.3 输出矩阵形状

librosa 返回的矩阵形状为 **`(n_mels, n_fft // 2 + 1)`** = **`(n_mels, 201)`**：

- 行：Mel 频带索引 `j`（0 → n_mels-1）
- 列：功率谱 bin 索引 `k`（0 → 200，含 DC 与 Nyquist）

每个 Mel 频带对应一组在 FFT 频率轴上的三角权重，用于将线性频率功率谱映射到 Mel 刻度。

### 2.4 按模型选择 n_mels

| 模型 | n_mels | npz 中的 key |
|------|--------|--------------|
| tiny / base / small / medium / large-v1 / large-v2 | 80 | `mel_80` |
| large-v3 / large-v3-turbo | 128 | `mel_128` |

---

## 3. 保存格式

Mel 滤波器系数经历 **两级持久化**：OpenAI 的 `.npz` 资源文件，以及 whisper.cpp 的 `.bin` 模型文件。

### 3.1 OpenAI 官方：`mel_filters.npz`

**路径**（官方仓库）：

```
whisper/assets/mel_filters.npz
```

**内容**：

```
mel_80   → float32 数组，shape (80, 201)
mel_128  → float32 数组，shape (128, 201)
```

该文件随 pip 包分发（`MANIFEST.in` 包含 `whisper/assets/*`），推理时直接加载，**无需 librosa 运行时依赖**。

### 3.2 whisper.cpp：`ggml-model.bin` 内嵌段

转换脚本 `models/convert-pt-to-ggml.py` 从官方 npz 读取并写入 GGML 模型：

```python
# 读取
n_mels = hparams["n_mels"]
with np.load(dir_whisper / "whisper" / "assets" / "mel_filters.npz") as f:
    filters = torch.from_numpy(f[f"mel_{n_mels}"])

# 写入 .bin（位于 hparams 之后、vocab 之前）
fout.write(struct.pack("i", filters.shape[0]))   # n_mel
fout.write(struct.pack("i", filters.shape[1]))   # n_fft (= 201)
for i in range(filters.shape[0]):
    for j in range(filters.shape[1]):
        fout.write(struct.pack("f", filters[i][j]))
```

**GGML 模型文件中 filters 段的二进制布局**：

```
int32   n_mel          (80 或 128)
int32   n_fft          (201)
float32 data[n_mel × n_fft]   行主序：data[j * n_fft + k]
```

同等逻辑也存在于 `models/convert-h5-to-ggml.py`（HuggingFace 格式转换）。

---

## 4. 推理时加载

### 4.1 OpenAI Whisper（Python）

`whisper/audio.py` 中 `mel_filters()` 从 npz 加载，带 LRU 缓存：

```python
@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int) -> torch.Tensor:
    assert n_mels in {80, 128}
    filters_path = os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")
    with np.load(filters_path, allow_pickle=False) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)
```

调用链：

```
whisper.load_model("large-v3")
    → model.dims.n_mels = 128
log_mel_spectrogram(audio, n_mels=model.dims.n_mels)
    → filters = mel_filters(device, n_mels)
```

### 4.2 whisper.cpp（C++）

#### 数据结构

```cpp
struct whisper_filters {
    int32_t n_mel;   // 80 或 128
    int32_t n_fft;   // 201
    std::vector<float> data;  // [n_mel × n_fft]
};

struct whisper_model {
    whisper_hparams hparams;
    whisper_filters filters;  // 成员
    // ...
};
```

#### 模型加载

在 `whisper_model_load()` 中，读取 hparams 之后、vocab 之前：

```cpp
// src/whisper.cpp — load mel filters
auto & filters = wctx.model.filters;

read_safe(loader, filters.n_mel);
read_safe(loader, filters.n_fft);

filters.data.resize(filters.n_mel * filters.n_fft);
loader->read(loader->context, filters.data.data(), filters.data.size() * sizeof(float));
BYTESWAP_FILTERS(filters);
```

加载完成后，`ctx->model.filters` 常驻内存，直到 `whisper_free()`。

#### 入口 API

| API | 说明 |
|-----|------|
| `whisper_init_from_file()` | 加载 .bin，含 filters |
| `whisper_init_from_buffer()` | 从内存 buffer 加载 |
| `whisper_pcm_to_mel()` | 使用 `ctx->model.filters` 计算 Mel |

---

## 5. 推理时使用

### 5.1 OpenAI Whisper

在 `log_mel_spectrogram()` 中，STFT 得到功率谱后做矩阵乘法：

```python
window = torch.hann_window(N_FFT).to(audio.device)
stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
magnitudes = stft[..., :-1].abs() ** 2          # 功率谱

filters = mel_filters(audio.device, n_mels)     # (n_mels, 201)
mel_spec = filters @ magnitudes                   # (n_mels, n_frames)

log_spec = torch.clamp(mel_spec, min=1e-10).log10()
log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
log_spec = (log_spec + 4.0) / 4.0
```

等价于：对每个时间帧，每个 Mel 频带 `j` 做 dot product：

```
S[j] = Σ_k  P[k] × filters[j, k]
```

### 5.2 whisper.cpp

#### 调用入口

```cpp
// whisper_pcm_to_mel_with_state()
log_mel_spectrogram(*state, samples, n_samples,
    WHISPER_SAMPLE_RATE,    // 16000
    WHISPER_N_FFT,          // 400
    WHISPER_HOP_LENGTH,     // 160
    ctx->model.filters.n_mel,
    n_threads,
    ctx->model.filters,     // ← 滤波器矩阵
    false,
    state->mel);
```

#### Worker 线程中的点积

每帧 STFT 得到功率谱 `fft_out[k]`（201 个 bin）后：

```cpp
for (int j = 0; j < mel.n_mel; j++) {
    double sum = 0.0;
    for (int k = 0; k < n_fft; k++) {
        sum += fft_out[k] * filters.data[j * n_fft + k];
    }
    sum = log10(std::max(sum, 1e-10));
    mel.data[j * mel.n_len + i] = sum;
}
```

存储布局：**行 = Mel 频带 `j`，列 = 时间帧 `i`**，即 `data[j * n_len + i]`。

#### 后续流向

```
ctx->model.filters
    ↓ log_mel_spectrogram()
state->mel.data
    ↓ whisper_encode_internal() 切片拷贝
Encoder 输入张量 mel [2×n_ctx, n_mels]
    ↓ Conv1d + Transformer
音频特征 embedding
```

---

## 6. 完整数据流

```
┌─────────────────────────────────────────────────────────────────┐
│  离线（一次性）                                                   │
│                                                                 │
│  librosa.filters.mel(sr=16000, n_fft=400, n_mels=80|128)        │
│      ↓ np.savez_compressed                                      │
│  whisper/assets/mel_filters.npz  (mel_80 / mel_128)             │
│      ↓ convert-pt-to-ggml.py                                    │
│  ggml-model.bin  (filters 段: n_mel, n_fft, float32[])          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  推理时                                                          │
│                                                                 │
│  [Python]  mel_filters()  ← 加载 mel_filters.npz              │
│  [C++]     whisper_model_load()  ← 读取 .bin 中 filters 段       │
│      ↓                                                          │
│  功率谱 |X[k]|²  ×  filters[j,k]  →  Mel 能量 S[j]              │
│      ↓ log₁₀ + clamp + normalize                                │
│  Log Mel Spectrogram  →  Encoder                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. 关键源码索引

| 环节 | 项目 | 文件 | 位置 |
|------|------|------|------|
| 生成说明 | OpenAI Whisper | `whisper/audio.py` | `mel_filters()` docstring |
| npz 资源 | OpenAI Whisper | `whisper/assets/mel_filters.npz` | — |
| 运行时加载 | OpenAI Whisper | `whisper/audio.py` | `mel_filters()` |
| 推理使用 | OpenAI Whisper | `whisper/audio.py` | `log_mel_spectrogram()` |
| 写入 .bin | whisper.cpp | `models/convert-pt-to-ggml.py` | L219-286 |
| 加载 .bin | whisper.cpp | `src/whisper.cpp` | `whisper_model_load()` L1576-1586 |
| 推理使用 | whisper.cpp | `src/whisper.cpp` | `log_mel_spectrogram_worker_thread()` L3139-3157 |
| 公共 API | whisper.cpp | `src/whisper.cpp` | `whisper_pcm_to_mel_with_state()` L3892 |

---

## 8. 常见问题

### Q: 推理时会重新计算 Mel 滤波器吗？

**不会。** Python 从 npz 加载，C++ 从 .bin 加载，均为只读使用。

### Q: 不同模型大小的 filters 有何区别？

STFT 参数（16 kHz / 400 / 160）相同；仅 **`n_mels` 和矩阵行数** 不同（80 vs 128）。large-v3 的 filters 矩阵比旧模型大约 60%。

### Q: 量化模型（Q5_0）的 filters 会变吗？

**不会。** 量化只压缩 Encoder/Decoder 权重；Mel 滤波器段仍以 float32 原样存储和读取。

### Q: 如何本地复现生成？

```python
import librosa, numpy as np

mel_80  = librosa.filters.mel(sr=16000, n_fft=400, n_mels=80)
mel_128 = librosa.filters.mel(sr=16000, n_fft=400, n_mels=128)
np.savez_compressed("mel_filters.npz", mel_80=mel_80, mel_128=mel_128)
```

生成后替换 `whisper/assets/mel_filters.npz`，或在转换 GGML 模型时让 `convert-pt-to-ggml.py` 读取新的 npz。

---

## 9. 相关文档

- [log-mel-spectrogram.html](./log-mel-spectrogram.html) — Log Mel Spectrogram 完整计算流程与可视化
- [whisper.cpp模型加载.html](./whisper.cpp模型加载.html) — GGML 模型加载机制
