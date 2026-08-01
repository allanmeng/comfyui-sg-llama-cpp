# Changelog

## [Unreleased / v1.7.0]

### Added
- **`load_mode` option** replaces deprecated `use_mmap` / `use_mlock` / `use_direct_io` (llama-cpp-python >= 0.3.45)
  - Options: `mmap` (default), `none`, `mlock`, `mmap_mlock`, `direct_io`
  - Automatic runtime detection: passes `load_mode` on 0.3.45+, falls back to legacy params on older whl (0.3.39 ~ 0.3.44)

### Changed
- `LlamaCPPOptions` schema: removed `use_mmap` / `use_mlock` / `use_direct_io` inputs, added `load_mode` combo

### Compatibility
- Plugin works with llama-cpp-python **0.3.39 ~ 0.3.45** without modification
- **0.3.45** recommended: includes the SYCL single-device sync fix (ggml-org/llama.cpp PR #25741), Windows DLL loading guard fixes, modernized embeddings API

---

## 0.3.39+ Adaptation (earlier)

- `clip_model_path` deprecated → `mmproj_path` passed directly to `Llama()`
- Manual vision handler creation removed → `Llama()` internally creates handler via `chat_handler_kwargs`
- `GenericMTMDChatHandler` replaces model-specific handlers
- `ctx_checkpoints` option added (required for hybrid models like Qwen3.5)
- Auto-detect `ComfyUI/models/LLM/` folder, recursive GGUF scanning
- Chinese README added
