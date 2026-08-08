[中文](README_CN.md) | English

# comfyui-sg-llama-cpp (Intel GPU SYCL Accelerated)

ComfyUI custom node that acts as a llama-cpp-python wrapper, **specifically optimized for Intel GPU (Arc / Iris Xe) with SYCL backend acceleration**. Supports vision models and allows generating text responses from prompts using llama.cpp.

![Screenshot](assets/node_preview.png)

- Load and use GGUF models (including vision models)
- Generate text prompts using llama.cpp
- Support for multi-modal inputs (multiple images/batches)
- Advanced sampling (Min-P, Presence/Frequency penalties)
- JSON Schema support for structured output (json_object)
- Memory management options
- Integration with ComfyUI workflows
- **llama-cpp-python 0.3.39+ MTMD rewrite support** (GenericMTMDChatHandler)
- **Hybrid model support** (Qwen3.5 etc. via `ctx_checkpoints`)

## Installation

1. Install the SYCL-accelerated llama-cpp-python wheel.

   > **Important**: JamePeng's releases do **not** include SYCL builds. For Intel GPU SYCL acceleration, download the whl from:
   > ```
   > https://github.com/allanmeng/llama-cpp-python-sycl-windows
   > ```
   > If you don't need SYCL (e.g., using CUDA/CPU), you can use the generic builds from https://github.com/JamePeng/llama-cpp-python/releases

2. Clone this repository into your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/allanmeng/comfyui-sg-llama-cpp
   ```

3. Restart ComfyUI.

### Upgrading from 0.3.38 or Earlier

If you are upgrading llama-cpp-python from version **0.3.38 or earlier** to **0.3.39+**, do NOT use `pip install --upgrade` — old files may linger and cause conflicts. Follow this procedure instead:

```bat
# 1. Completely remove the old version
pip uninstall llama-cpp-python -y

# 2. Install the new wheel
pip install llama_cpp_python-0.3.41-<your-build>.whl
```

> **Why uninstall first?** A simple `--upgrade` may leave stale DLLs from the old version in `lib/`, which can conflict with the new version's files. Uninstalling first ensures a clean state.

## What's New (0.3.39+ Adaptation)

This fork adapts the plugin for **llama-cpp-python 0.3.39+**, which introduced a major MTMD (Multi-Modal Token Decomposition) rewrite:

- **`clip_model_path` deprecated** → replaced by `mmproj_path` passed directly to `Llama()`
- **Manual handler creation removed** → `Llama()` now internally creates the vision handler via `chat_handler_kwargs`
- **`GenericMTMDChatHandler`** replaces model-specific handlers as the primary vision handler
- **Parameter filtering** inspects `GenericMTMDChatHandler` union `MTMDChatHandler` to filter valid kwargs
- **`ctx_checkpoints` option** added (default `0`, required for hybrid models like Qwen3.5 Transformer+Mamba)
- **Removed invalid UI params**: `vision_enable_thinking`, `vision_force_reasoning`, `vision_add_vision_id` (not accepted by `GenericMTMDChatHandler`)
- **`vision_image_min_tokens` default** changed from `-1` to `1024` (Qwen-VL minimum requirement)

## Node Reference

### LlamaCPPModelLoader
Loads GGUF model files and prepares them for use.

**Inputs**
- **Required**:
  - `model_name`: Select the GGUF model file to load.
- **Optional**:
  - `chat_format`: Chat template to use (default: `llama-2`). In 0.3.39+, vision models auto-detect via `GenericMTMDChatHandler`.
  - `mmproj_model_name`: Multi-modal projector model for vision (default: `None`).

**Outputs**
- `MODEL`: The loaded Llama model object.

### LlamaCPPOptions
Configures advanced parameters for the model.

**Inputs**
- **Optional**:
  - `n_gpu_layers`: Number of layers to offload to GPU (default: `-1` for all).
  - `n_ctx`: Context window size (default: `2048`).
  - `n_threads`: CPU threads to use (default: `-1` for auto).
  - `n_threads_batch`: Threads for batch processing (default: `-1` for auto).
  - `n_batch`: Batch size (default: `2048`).
  - `n_ubatch`: Micro-batch size (default: `512`).
  - `main_gpu`: Main GPU ID (default: `0`).
  - `offload_kqv`: Offload K/Q/V to GPU (default: `Enabled`).
  - `numa`: NUMA support (default: `Disabled`).
  - `use_mmap`: Memory mapping (default: `Enabled`).
  - `use_mlock`: Memory locking (default: `Disabled`).
  - `use_direct_io`: Enable direct I/O for library (Linux only, default: `Disabled`).
  - `verbose`: Verbose logging (default: `Disabled`).
  - `ctx_checkpoints`: Context checkpoints (default: `0` to disable; `-1` for default; **required for hybrid models like Qwen3.5**).
  - `vision_use_gpu`: Enable GPU for vision handler (default: `Enabled`).
  - `vision_image_min_tokens`: Minimum image tokens (default: `1024`, recommended for Qwen-VL; `-1` for default).
  - `vision_image_max_tokens`: Maximum image tokens (default: `-1` for default).

**Outputs**
- `OPTIONS`: A configuration dictionary.

### LlamaCPPEngine
The main generation node.

**Inputs**
- **Required**:
  - `model`: The model from `LlamaCPPModelLoader`.
  - `prompt`: The text prompt.
- **Optional**:
  - `images`: Input image(s) for vision models (supports batches).
  - `options`: Options from `LlamaCPPOptions`.
  - `system_prompt`: System instruction (default: empty).
  - `memory_cleanup`: Strategy to clean memory after generation (default: `close`).
  - `response_format`: `text` or `json_object` (default: `text`).
  - `json_schema`: JSON Schema to enforce (available only when `json_object` is selected).
  - `max_tokens`: Max new tokens (default: `512`).
  - `temperature`: Randomness (default: `0.2`).
  - `top_p`: Nucleus sampling (default: `0.95`).
  - `top_k`: Top-k sampling (default: `40`).
  - `min_p`: Min-p sampling (default: `0.05`).
  - `repeat_penalty`: Penalty for repetition (default: `1.1`).
  - `present_penalty`: Penalty for presence of tokens (default: `0.0`).
  - `frequency_penalty`: Penalty for frequency of tokens (default: `0.0`).
  - `seed`: Random seed (default: `1`).

**Outputs**
- `RESPONSE`: The generated text.

### LlamaCPPMemoryCleanup
Utility to manually free resources.

**Inputs**
- **Required**:
  - `memory_cleanup`: Cleanup mode (`close`, `backend_free`, `full_cleanup`, `persistent`).
- **Optional**:
  - `passthrough`: Any input to pass through (allows chaining).

**Outputs**
- `PASSTHROUGH`: The input passed through unmodified.

## Model Directory Rules

The plugin scans the following directories for `.gguf` files **recursively** (including subdirectories):

| Source | Path(s) | Auto-detected |
|--------|---------|:---:|
| ComfyUI `text_encoders` | `ComfyUI/models/text_encoders/` + `ComfyUI/models/clip/` | Yes |
| Dedicated LLM folder | `ComfyUI/models/LLM/` | Yes |
| Custom folders | Any path in `config.json` | No |

All scanned files are automatically separated into two dropdowns by filename:
- **Model dropdown**: `.gguf` files (excluding those with `mmproj` in the name)
- **mmproj dropdown**: `.gguf` files with `mmproj` in the filename

### Recommended Layout

```
ComfyUI/models/LLM/
├── Qwen3.5-4B-Q4_K_M.gguf          ← LLM model
├── Qwen2.5-VL-3B-instruct-q4_k_m.gguf
├── mmproj/                          ← Vision projectors (or any name)
│   └── Qwen2.5-VL-3B-Instruct-mmproj-f16.gguf
└── GGUF/                            ← Can use subdirectories freely
    └── ...
```

`mmproj` files only need the keyword "mmproj" in their filename to appear in the mmproj dropdown — their folder location doesn't matter.

### Custom Folders (config.json)

Create `config.json` in the plugin directory for additional search paths:

```json
{
  "model_folders": [
    "D:\\AI\\LLM\\models",
    "/home/user/models"
  ]
}
```

- The `config.json` file is **optional** — the node works without it
- Paths can be absolute or relative
- Both Windows and Unix style paths are supported
- Non-existent paths are automatically filtered out
- See `config.example.json` for additional examples

## Requirements

- llama-cpp-python >= 0.3.39
  - **Intel GPU (SYCL)**: https://github.com/allanmeng/llama-cpp-python-sycl-windows
  - **Other backends (CUDA/CPU)**: https://github.com/JamePeng/llama-cpp-python/releases
## License

This project is licensed under the GNU AGPLv3 License - see the [LICENSE](LICENSE) file for details.

## Repository

- **Fork (SYCL optimized)**: https://github.com/allanmeng/comfyui-sg-llama-cpp
- **SYCL whl builds**: https://github.com/allanmeng/llama-cpp-python-sycl-windows
- **Upstream**: https://github.com/sebagallo/comfyui-sg-llama-cpp
