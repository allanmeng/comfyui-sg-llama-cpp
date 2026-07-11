中文 | [English](README.md)

# comfyui-sg-llama-cpp（Intel GPU SYCL 加速版）

ComfyUI 自定义节点，llama-cpp-python 的封装插件，**专为 Intel GPU（Arc / Iris Xe）SYCL 后端加速优化**。支持视觉模型，通过 llama.cpp 从提示词生成文本响应。

![截图](assets/node_preview.png)

- 加载和使用 GGUF 模型（包括视觉模型）
- 通过 llama.cpp 生成文本提示词
- 支持多模态输入（多图/批量）
- 高级采样（Min-P、Presence/Frequency 惩罚）
- JSON Schema 结构化输出支持（json_object）
- 内存管理选项
- 与 ComfyUI 工作流集成
- **支持 llama-cpp-python 0.3.39+ MTMD 重写**（GenericMTMDChatHandler）
- **支持混合架构模型**（Qwen3.5 等，通过 `ctx_checkpoints`）

## 安装

1. 安装 SYCL 加速版 llama-cpp-python whl 包。

   > **重要**：JamePeng 的 Release 中**不包含** SYCL 构建。Intel GPU SYCL 加速请从以下地址下载 whl：
   > ```
   > https://github.com/allanmeng/llama-cpp-python-sycl-windows
   > ```
   > 如果你不需要 SYCL（例如使用 CUDA/CPU），可以使用通用构建：https://github.com/JamePeng/llama-cpp-python/releases

2. 将本仓库克隆到 ComfyUI 的 custom_nodes 目录：
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/allanmeng/comfyui-sg-llama-cpp
   ```

3. **仅 SYCL 用户**：安装 whl 后，删除 `bin/` 目录以防止 DLL 分裂加载崩溃：
   ```bat
   rmdir /s /q "你的Python路径\Lib\site-packages\llama_cpp\bin"
   ```

4. 重启 ComfyUI。

## 更新内容（0.3.39+ 适配）

本 Fork 适配了 **llama-cpp-python 0.3.39+**，该版本引入了重大 MTMD（Multi-Modal Token Decomposition）重写：

- **`clip_model_path` 已废弃** → 改为 `mmproj_path` 直接传给 `Lama()`
- **移除手动创建 handler** → `Llama()` 现在通过 `chat_handler_kwargs` 内部自动创建视觉 handler
- **`GenericMTMDChatHandler`** 替代模型特定 handler，成为主要视觉 handler
- **参数过滤** 对 `GenericMTMDChatHandler` 与 `MTMDChatHandler` 取并集进行过滤
- **新增 `ctx_checkpoints` 选项**（默认 `0`，混合架构模型如 Qwen3.5 Transformer+Mamba 必须设置）
- **移除无效 UI 参数**：`vision_enable_thinking`、`vision_force_reasoning`、`vision_add_vision_id`（`GenericMTMDChatHandler` 不接受）
- **`vision_image_min_tokens` 默认值** 从 `-1` 改为 `1024`（Qwen-VL 最低要求）

### SYCL 环境注意事项（Windows）

如果你使用 **Intel Arc GPU + SYCL 后端**，安装 llama-cpp-python 后必须删除 `bin/` 目录，否则会导致 DLL 分裂加载崩溃：

```bat
rmdir /s /q "你的Python路径\Lib\site-packages\llama_cpp\bin"
```

`bin/` 目录只包含部分 DLL（14 个中的 7 个），会导致 Windows DLL 加载器混合加载路径，引发 SYCL 运行时初始化冲突（access violation）。`lib/` 目录包含完整的 DLL 集合。详见 [ISSUE_JAMEPENG.md](ISSUE_JAMEPENG.md) 的完整分析。

## 节点参考

### LlamaCPPModelLoader
加载 GGUF 模型文件并准备使用。

**输入**
- **必填**：
  - `model_name`：选择要加载的 GGUF 模型文件。
- **可选**：
  - `chat_format`：聊天模板（默认：`llama-2`）。0.3.39+ 视觉模型通过 `GenericMTMDChatHandler` 自动检测。
  - `mmproj_model_name`：多模态投影模型（默认：`None`）。

**输出**
- `MODEL`：加载的 Llama 模型对象。

### LlamaCPPOptions
配置模型的高级参数。

**输入**
- **可选**：
  - `n_gpu_layers`：卸载到 GPU 的层数（默认：`-1` 表示全部）。
  - `n_ctx`：上下文窗口大小（默认：`2048`）。
  - `n_threads`：CPU 线程数（默认：`-1` 自动）。
  - `n_threads_batch`：批处理线程数（默认：`-1` 自动）。
  - `n_batch`：批大小（默认：`2048`）。
  - `n_ubatch`：微批大小（默认：`512`）。
  - `main_gpu`：主 GPU ID（默认：`0`）。
  - `offload_kqv`：将 K/Q/V 卸载到 GPU（默认：`启用`）。
  - `numa`：NUMA 支持（默认：`禁用`）。
  - `use_mmap`：内存映射（默认：`启用`）。
  - `use_mlock`：内存锁定（默认：`禁用`）。
  - `use_direct_io`：直接 I/O（仅 Linux，默认：`禁用`）。
  - `verbose`：详细日志（默认：`禁用`）。
  - `ctx_checkpoints`：上下文检查点（默认：`0` 禁用；`-1` 使用默认值；**混合架构模型如 Qwen3.5 必须设置**）。
  - `vision_use_gpu`：视觉 handler 启用 GPU（默认：`启用`）。
  - `vision_image_min_tokens`：最小图像 token 数（默认：`1024`，Qwen-VL 推荐；`-1` 使用默认值）。
  - `vision_image_max_tokens`：最大图像 token 数（默认：`-1` 使用默认值）。

**输出**
- `OPTIONS`：配置字典。

### LlamaCPPEngine
主要的生成节点。

**输入**
- **必填**：
  - `model`：来自 `LlamaCPPModelLoader` 的模型。
  - `prompt`：文本提示词。
- **可选**：
  - `images`：视觉模型的输入图像（支持批量）。
  - `options`：来自 `LlamaCPPOptions` 的选项。
  - `system_prompt`：系统指令（默认：空）。
  - `memory_cleanup`：生成后清理内存的策略（默认：`close`）。
  - `response_format`：`text` 或 `json_object`（默认：`text`）。
  - `json_schema`：JSON Schema（仅在选择 `json_object` 时可用）。
  - `max_tokens`：最大生成 token 数（默认：`512`）。
  - `temperature`：采样温度（默认：`0.2`）。
  - `top_p`：核采样（默认：`0.95`）。
  - `top_k`：Top-k 采样（默认：`40`）。
  - `min_p`：Min-p 采样（默认：`0.05`）。
  - `repeat_penalty`：重复惩罚（默认：`1.1`）。
  - `present_penalty`：Presence 惩罚（默认：`0.0`）。
  - `frequency_penalty`：Frequency 惩罚（默认：`0.0`）。
  - `seed`：随机种子（默认：`1`）。

**输出**
- `RESPONSE`：生成的文本。

### LlamaCPPMemoryCleanup
手动释放资源的工具节点。

**输入**
- **必填**：
  - `memory_cleanup`：清理模式（`close`、`backend_free`、`full_cleanup`、`persistent`）。
- **可选**：
  - `passthrough`：任意输入透传（允许链式连接）。

**输出**
- `PASSTHROUGH`：原样透传的输入。

## 模型目录规则

插件会**递归扫描**以下目录中的 `.gguf` 文件（包括子目录）：

| 来源 | 路径 | 自动检测 |
|------|------|:---:|
| ComfyUI `text_encoders` | `ComfyUI/models/text_encoders/` + `ComfyUI/models/clip/` | 是 |
| 专属 LLM 目录 | `ComfyUI/models/LLM/` | 是 |
| 自定义目录 | `config.json` 中指定的任意路径 | 否 |

所有扫描到的文件会按文件名自动分离到两个下拉菜单：
- **模型下拉菜单**：`.gguf` 文件（排除文件名含 `mmproj` 的文件）
- **mmproj 下拉菜单**：文件名含 `mmproj` 的 `.gguf` 文件

### 推荐目录结构

```
ComfyUI/models/LLM/
├── Qwen3.5-4B-Q4_K_M.gguf          ← LLM 模型
├── Qwen2.5-VL-3B-instruct-q4_k_m.gguf
├── mmproj/                          ← 视觉投影文件（任意名称均可）
│   └── Qwen2.5-VL-3B-Instruct-mmproj-f16.gguf
└── GGUF/                            ← 可自由使用子目录
    └── ...
```

`mmproj` 文件只要文件名包含 "mmproj" 关键词就会出现在 mmproj 下拉菜单中，存放位置不限。

### 自定义目录（config.json）

在插件目录下创建 `config.json` 可添加额外搜索路径：

```json
{
  "model_folders": [
    "D:\\AI\\LLM\\models",
    "/home/user/models"
  ]
}
```

- `config.json` 是**可选的** — 没有也能正常工作
- 路径可以是绝对路径或相对路径
- 同时支持 Windows 和 Unix 风格路径
- 不存在的路径会自动过滤
- 参见 `config.example.json` 获取更多示例

## 依赖要求

- llama-cpp-python >= 0.3.39
  - **Intel GPU（SYCL）**：https://github.com/allanmeng/llama-cpp-python-sycl-windows
  - **其他后端（CUDA/CPU）**：https://github.com/JamePeng/llama-cpp-python/releases
- SYCL 安装后需删除 `llama_cpp/bin/`（参见安装步骤 3）

## 许可证

本项目基于 GNU AGPLv3 许可证 — 详见 [LICENSE](LICENSE) 文件。

## 仓库地址

- **Fork（SYCL 优化版）**：https://github.com/allanmeng/comfyui-sg-llama-cpp
- **SYCL whl 构建**：https://github.com/allanmeng/llama-cpp-python-sycl-windows
- **上游**：https://github.com/sebagallo/comfyui-sg-llama-cpp
