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

## What's New (v2.3.0)

This fork adapts the plugin for **llama-cpp-python 0.3.39+** (MTMD rewrite) and ships additional fixes/tuning on top:

### v2.3.0 (2026-10-17)
- **`use_mmap` / `use_mlock` / `use_direct_io` were dead switches on 0.4.x** — they now show as `[DEPRECATED in 0.4.x]`, still work on 0.3.x, and are skipped when the build supports `load_mode`.
- **New 0.4.x knobs exposed**: `load_mode`, `lazy_mode`, `flash_attn_type`, `vision_flash_attn` (all default `auto`).
- **KV cache quantization**: `type_k` / `type_v` (`--cache-type-k/-v`) can cut the KV cache allocation by ~45% (`q8_0`) or ~70% (`q4_0`). A quantized V cache requires Flash Attention — that invalid combination is rejected with a clear message. **Measured: a net loss on small hybrid models** (`q8_0` = −7.4% decode speed for ~114 MiB saved on Qwen3.5-4B), so the default stays `auto`; it pays off only for large dense VLMs or very long contexts.
- **Parameter whitelist against the installed `Llama()` signature** — one code path for 0.3.x and 0.4.x, unknown params are dropped instead of leaking into `**kwargs`.
- **Vision token bounds no longer abort the run** — an inconsistent `vision_image_max_tokens` is dropped with a warning instead of raising.
- **New Options inputs are appended, never inserted mid-list** — ComfyUI stores widget values positionally, so mid-list insertions corrupted existing workflows.
- Benchmarked on B580/0.4.3: the four new knobs gave **no measurable speed-up** (`clip_encode` spread ≤1.4%); defaults stay `auto`.

### v2.2.0 (2026-08-22)
- **`checkpoint_interval` / `checkpoint_on_device` options** on `LlamaCPPOptions` for hybrid models (defaults `4096` / `Disabled`). Verified via 2×2 benchmark (host/device × 8/16 checkpoints): no impact on load, encode or token throughput — defaults are optimal for most users.
- **`think_display` toggle on `LlamaCPPEngine`** — `show`/`hide` to keep or strip the `<think>…</think>` reasoning block. Handles malformed tags (missing opening, multiple closing, dangling open).
- **`think` stripping robustness** — keeps only text after the *last* `</think>` (engines may emit `</think>…reasoning…<//think>answer` with no opening tag).
- **`ctx_checkpoints` default changed `0` → `-1`** — `0` triggers a buggy fast-path in llama-cpp-python 0.3.48 that crashes on large images; `-1` uses the llama_cpp default (16) and avoids the crash.
- **Large-image `n_ctx` exhaustion → reactive English hint** (no forced override): if the first decode fails with "memory slot for batch", the engine tells you to raise `n_ctx` (8192 known-good for large images).

### 0.3.39+ Adaptation
- **`clip_model_path` deprecated** → replaced by `mmproj_path` passed directly to `Llama()`
- **Manual handler creation removed** → `Llama()` now internally creates the vision handler via `chat_handler_kwargs`
- **`GenericMTMDChatHandler`** replaces model-specific handlers as the primary vision handler
- **Parameter filtering** inspects `GenericMTMDChatHandler` union `MTMDChatHandler` to filter valid kwargs
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
  - `n_ctx`: Context window size (default: `8192`; a large-image prefill needs 4000+ tokens of headroom).
  - `n_threads`: CPU threads to use (default: `-1` for auto).
  - `n_threads_batch`: Threads for batch processing (default: `-1` for auto).
  - `n_batch`: Batch size (default: `2048`).
  - `n_ubatch`: Micro-batch size (default: `512`).
  - `main_gpu`: Main GPU ID (default: `0`).
  - `offload_kqv`: Offload K/Q/V to GPU (default: `Enabled`).
  - `numa`: NUMA support (default: `Disabled`).
  - `use_mmap`: Memory mapping (default: `Enabled`). **DEPRECATED in llama-cpp-python 0.4.x** — ignored there, use `load_mode`; still honored by 0.3.x.
  - `use_mlock`: Memory locking (default: `Disabled`). **DEPRECATED in 0.4.x** — use `load_mode=mlock`.
  - `use_direct_io`: Enable direct I/O for library (Linux only, default: `Disabled`). **DEPRECATED in 0.4.x** — use `load_mode=direct_io`.
  - `verbose`: Verbose logging (default: `Disabled`). When enabled, the resolved 0.4.x tuning parameters are echoed to the console.
  - `ctx_checkpoints`: Context checkpoints (default: `-1` for default; `0` disables — **do not use `0` with llama-cpp-python 0.3.48, it triggers a buggy fast-path that crashes on large images**; required for hybrid models like Qwen3.5).
  - `checkpoint_interval`: Hybrid checkpoint token interval (default: `4096`).
  - `checkpoint_on_device`: Store hybrid checkpoint payloads in VRAM (default: `Disabled`; ~50 MiB per checkpoint, no throughput impact — benchmarks show defaults are optimal).
  - `vision_use_gpu`: Enable GPU for vision handler (default: `Enabled`).
  - `vision_image_min_tokens`: Minimum image tokens (default: `1024`, recommended for Qwen-VL; `-1` for default).
  - `vision_image_max_tokens`: Maximum image tokens (default: `-1` for default). Must be `-1` or `>= vision_image_min_tokens`.
  - `load_mode` (**0.4.x**): Model load strategy — `auto` / `none` / `mmap` / `mlock` / `mmap_mlock` / `direct_io` (default: `auto`).
  - `lazy_mode` (**0.4.x**): Lazy tensor loading, requires mmap — `auto` / `off` / `on` (default: `auto`).
  - `flash_attn_type` (**0.4.x**): Flash Attention for the main model — `auto` / `disabled` / `enabled` (default: `auto`).
  - `vision_flash_attn` (**0.4.x**): Flash Attention inside the vision encoder — `auto` / `enabled` / `disabled` (default: `auto` = handler default).
  - `type_k` (**0.4.x**): KV cache data type for K (`--cache-type-k`) — `auto` / `f16` / `q8_0` / `q5_1` / `q5_0` / `q4_1` / `q4_0` / `iq4_nl` / `bf16` (default: `auto` = f16). Quantizing shrinks the KV cache allocation (reported by `llama_kv_cache: size = ...`, **not** by the compute-buffer line). Measured on B580/Qwen3.5-4B: `q8_0` costs **−7.4%** decode speed for ~114 MiB saved, so it is only worth it for large dense VLMs or very long contexts.
  - `type_v` (**0.4.x**): KV cache data type for V (`--cache-type-v`), same options. **A quantized V cache requires Flash Attention** — it is rejected when `flash_attn_type='disabled'`.

> The four `0.4.x` knobs are appended at the end of the input list on purpose: ComfyUI
> maps saved workflow widget values by position. On a 0.3.x build they are ignored
> automatically (detected from the installed `Llama()` signature).

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
  - `memory_cleanup`: Strategy to clean memory after generation (default: `full_cleanup`).
  - `response_format`: `text` or `json_object` (default: `text`).
  - `json_schema`: JSON Schema to enforce (available only when `json_object` is selected).
  - `max_tokens`: Max new tokens (default: `2048`).
  - `temperature`: Randomness (default: `0.1`).
  - `top_p`: Nucleus sampling (default: `0.9`).
  - `top_k`: Top-k sampling (default: `20`).
  - `min_p`: Min-p sampling (default: `0.05`).
  - `repeat_penalty`: Penalty for repetition (default: `1.1`).
  - `present_penalty`: Penalty for presence of tokens (default: `0.0`).
  - `frequency_penalty`: Penalty for frequency of tokens (default: `0.0`).
  - `seed`: Random seed (default: `1`).
  - `think_display`: `show` or `hide` (default: `hide`). `hide` strips the
    `<think>…</think>` reasoning block so only the final answer is returned.
    Handles malformed tags (missing opening, multiple closing, dangling open).

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

## Batch Captioning (LoRA dataset tagging)

Two optional utility nodes turn the plugin into a batch captioning pipeline (the common
LoRA dataset workflow: describe a folder of images, one `.txt` per image). They live under
`LlamaCPP/Dataset` and never touch the inference path — ignore them if you only generate
single-image prompts.

```
Llama CPP Image Folder Loader ── images ──┐
                              └─ paths  ──┤
                                          └─► Llama CPP Engine ── RESPONSE ──► Llama CPP Caption Saver
                                                   (runs once per image,          (writes one .txt per image,
                                                    one caption each)             next to the source image)
```

No third-party nodes are required, and images are never re-encoded or renamed.

### LlamaCPPImageFolderLoader

Scans a folder and emits a **list**, so a non-list node downstream (the Engine) runs once
per image. Emitting a tensor batch instead would put every image into a single request and
run out of context on the first try.

**Inputs**
- **Required**:
  - `folder_path`: Absolute path is used as-is; a relative path resolves under `ComfyUI/input`.
  - `extensions`: Comma separated, default `png,jpg,jpeg,webp,bmp`.
  - `recursive`: Scan sub-folders (default `true`).
  - `sort_order`: `natural` (2.png before 10.png) / `name` / `modified` / `none`.
  - `max_images`: `0` = all. Use a small value to validate a workflow first.
  - `start_index`: Skip the first N images after sorting, to resume an interrupted run.
  - `max_side`: Downscale so the longest side is at most this many pixels (`0` = keep
    original). This is the cheapest way to cut vision tokens and base64 size; `1024` is a
    good captioning default.

**Outputs**
- `images` (list): One `[1, H, W, C]` tensor per file.
- `paths` (list): Absolute file paths, parallel to `images`.
- `count`: Number of loaded images.

Images are EXIF-rotated to upright, and transparent pixels are composited on white instead
of black. Unreadable files are skipped with a warning instead of aborting the run.

### LlamaCPPCaptionSaver

Writes one `.txt` per caption. By default the file lands **next to the source image with the
same stem** (`abc.png` → `abc.txt`), which is exactly what kohya-style trainers expect.

**Inputs**
- **Required**:
  - `text`: Caption(s) from `LlamaCPPEngine.RESPONSE`.
  - `output_dir`: Empty = write next to each source image. Relative paths resolve under
    `ComfyUI/input`.
  - `skip_on_error`: Drop captions that are the Engine's error string
    (`Error generating response: ...`). Default `true` — otherwise a failed image silently
    becomes a garbage caption.
  - `skip_empty`: Drop blank captions (default `true`).
- **Optional**:
  - `paths`: `paths` output of the loader. Leave unconnected to write numbered files
    (`caption_00000.txt`, ...) into `output_dir` instead.

**Outputs**
- `saved` / `skipped`: Counters for verification.

Files are written as UTF-8 with `\n` line endings, and every write is echoed to the console
(`[llama_cpp dataset] caption saved: ...`) so a batch run can be audited afterwards.

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
  ],
  "exclude_patterns": [
    "wan22",
    "krea2"
  ]
}
```

- The `config.json` file is **optional** — the node works without it
- Paths can be absolute or relative
- Both Windows and Unix style paths are supported
- Non-existent paths are automatically filtered out
- `exclude_patterns` is **optional**: case-insensitive substrings matched against the model's path relative to its search folder. Any GGUF whose path contains one of them is hidden from **both** dropdowns (models and mmproj). Useful when a folder is shared with a diffusion / video GGUF loader — llama.cpp cannot run those files, so they only clutter the list.
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
