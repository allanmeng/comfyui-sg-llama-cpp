# Changelog

All notable changes to this fork are documented here.

## [2.3.0] — 2026-10-17

### Fixed
- **`use_mmap` / `use_mlock` / `use_direct_io` were dead switches on llama-cpp-python 0.4.x.**
  0.4.x removed them from the load path (they only trigger a deprecation notice when
  `verbose`) and replaced them with `load_mode` / `lazy_mode`. The three Options toggles
  were silently ignored, so users had no working control over how weights were loaded.
  They are now marked `[DEPRECATED in 0.4.x]`, still honored on 0.3.x, and skipped
  entirely when the installed build supports `load_mode`.
- **Vision token bounds no longer abort the whole run.**
  `GenericMTMDChatHandler` raises `image_max_tokens cannot be less than image_min_tokens`
  when `0 < image_max_tokens < image_min_tokens` — easy to hit when a stale workflow
  carries an unrelated number in that widget. The plugin now drops the offending bound
  (falling back to the handler default) and prints a warning instead of failing.
- **New Options inputs are appended, never inserted mid-list.**
  ComfyUI stores workflow widget values *positionally*, so inserting a new input in the
  middle shifted the stored value of every widget after it (a Combo could render as a
  boolean, an Int could inherit a neighbour's number). All new inputs are appended at
  the end of the list to keep existing workflows intact.

### Added
- **`load_mode` / `lazy_mode` / `flash_attn_type` on `LlamaCPPOptions`** (llama-cpp-python 0.4.x).
- **`vision_flash_attn` on `LlamaCPPOptions`** (0.4.x, tri-state `auto`/`enabled`/`disabled`;
  `auto` keeps the handler default `None`).
- **`type_k` / `type_v` KV cache quantization on `LlamaCPPOptions`** (0.4.x; maps to
  llama.cpp `--cache-type-k` / `--cache-type-v`, i.e. a `ggml_type` value). Defaults
  `auto` = the library default `f16`, so behaviour is unchanged unless opted in.
  Quantizing shrinks the KV cache proportionally to bits per element, which matters most
  at large `n_ctx`: for a Qwen3.5-4B at `n_ctx=8192` the f16 KV cache is ~256 MiB, so
  `q8_0` saves ~114 MiB and `q4_0` ~181 MiB — 22–35% of a ~513 MiB compute buffer.
  - **A quantized V cache requires Flash Attention.** `llama-context.cpp` refuses to
    build the context (`"V cache quantization requires flash_attn"`) when
    `flash_attn_type` is `disabled`; the plugin now rejects that combination up front
    with an actionable message instead of an opaque context-creation failure.
  - The type's block size must divide the head dimension (32 divides the 256 of
    Qwen3.5); unsupported combinations are reported by llama.cpp itself.
- **Parameter whitelist against the installed `Llama()` signature**, so a single code path
  serves both 0.3.x and 0.4.x: parameters unknown to the installed build are dropped
  instead of leaking into `**kwargs`. Detection is by signature, not by version string.
- **`verbose=true` now echoes the resolved tuning parameters**, so it is possible to
  confirm from the console that a Combo actually reached `Llama()`, e.g.
  `[llama_cpp] resolved tuning params: {'load_mode': 0, 'lazy_mode': None, ...}`.

### Changed — tuned defaults
Defaults retuned for vision/description workflows (a fresh node now starts from a
configuration that was validated on B580). Existing workflows keep their own saved
values; only new nodes pick these up.

| Node | Parameter | Old | New |
|---|---|---|---|
| Options | `n_ctx` | `2048` | **`8192`** |
| Engine | `memory_cleanup` | `close` | **`full_cleanup`** |
| Engine | `max_tokens` | `512` | **`2048`** |
| Engine | `temperature` | `0.2` | **`0.1`** |
| Engine | `top_p` | `0.95` | **`0.9`** |
| Engine | `top_k` | `40` | **`20`** |
| Engine | `think_display` | `show` | **`hide`** |

- `n_ctx=8192` in particular: the old `2048` default could not hold a large-image
  prefill (4000+ vision tokens) plus the first decode slot, which is the crash mode
  documented in 2.2.0. `8192` is the tested floor.
- Deliberately **not** changed: `verbose` (stays `False` — verbose floods the console
  with hundreds of `find_slot` lines per image), `checkpoint_interval` (stays `4096`,
  the llama-cpp-python default; a smaller interval was never benchmarked), and
  `checkpoint_on_device` (stays `False` — the 2×2 benchmark showed no speed gain,
  +50 MiB VRAM, and one fewer rollback slot).

### Notes
- Benchmarked on Intel Arc B580 (SYCL, llama-cpp-python 0.4.3, Qwen3.5-4B, 1792×2304 /
  4032 vision tokens): none of the four new knobs changed `clip_encode`, prefill speed or
  token throughput beyond run-to-run noise (`clip_encode` spread ≤1.4%, ~11.0 s in all
  configurations). `vision_flash_attn=enabled` was not faster than `disabled`.
  Defaults therefore stay `auto` — these are exposed for tuning and future hardware, not
  as a speed-up.

## [2.2.0] — 2026-08-22
> Renumbered from an incorrect `1.7.0` label: the latest release tag was `v2.1.0`,
> so this release continues the 2.x line.

### Fixed
- **`think` content stripping (hide mode) for malformed tags.**
  The previous 3-pass regex only stripped up to the *first* `</think>` tag. Some
  engines emit a malformed block with **two** `</think>` tags and no opening
  `<think>` (e.g. `</think>…reasoning…<//think>answer`), so the reasoning text and
  the trailing `</think>` survived. The logic now keeps **only the text after the
  last `</think>`**, which is the real final answer. Verified against well-formed,
  double-close, single-close, dangling-open and no-think inputs.
- **Hybrid model crash on large images (`ctx_checkpoints`).**
  Default changed from `0` to `-1`. A value of `0` triggers a buggy fast-path in
  llama-cpp-python 0.3.48 that crashes on large images; `-1` lets llama_cpp use its
  default (16) and avoids the crash.

### Added
- **`think_display` control on `LlamaCPPEngine`.**
  Moved from `LlamaCPPOptions` to the Engine node. Combo `show` / `hide`
  (default `show`). `hide` strips the `<think>…</think>` reasoning block so only the
  final answer is returned. Pure output post-processing — no model reload required.
- **`checkpoint_interval` and `checkpoint_on_device` options on `LlamaCPPOptions`.**
  Expose the hybrid checkpoint tuning knobs (defaults: 4096 / False, matching
  llama-cpp-python). Verified via 2×2 benchmark (host/device × 8/16 checkpoints)
  that these knobs do not affect load time, encode time or token throughput —
  defaults are optimal for most users; advanced users may enable
  `checkpoint_on_device` to move the 50 MiB checkpoint payload to VRAM.

### Changed
- **Large-image `n_ctx` exhaustion is now a reactive hint, not a forced override.**
  The plugin no longer silently raises `n_ctx`. If a generation fails with a
  "memory slot for batch" error (the hybrid recursive memory backend ran out of
  KV/decoding slots on the first decode token), the engine prints an English hint
  telling the user to increase `n_ctx` in the Options node (8192 is known-good for
  large images). This respects `n_ctx` as a user-controlled knob.

---

## [0.3.39+] — 2026-07

### Added
- Adapted for **llama-cpp-python 0.3.39+** MTMD rewrite (`GenericMTMDChatHandler`).
- `mmproj_path` replaces deprecated `clip_model_path`.
- `ctx_checkpoints` option for hybrid models (Qwen3.5 etc.).
- Parameter filtering via `GenericMTMDChatHandler` ∪ `MTMDChatHandler`.

See git history for earlier changes.
