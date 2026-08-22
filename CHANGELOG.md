# Changelog

All notable changes to this fork are documented here.

## [Unreleased] — 2026-08-22

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
