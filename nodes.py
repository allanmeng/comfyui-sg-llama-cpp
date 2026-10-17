import os
import gc
import inspect
import torch
import sys
import json
import re
from llama_cpp import Llama, llama_backend_free
from llama_cpp.llama_chat_format import LlamaChatCompletionHandlerRegistry
try:
    from llama_cpp.llama_chat_format import MTMDChatHandler
except ImportError:
    # Support for older llama-cpp-python versions
    from llama_cpp.llama_chat_format import Llava15ChatHandler
    MTMDChatHandler = Llava15ChatHandler

# In llama-cpp-python >= 0.3.39, Llama() uses GenericMTMDChatHandler internally
# for vision models. We inspect it to filter chat_handler_kwargs properly.
try:
    from llama_cpp.llama_chat_format import GenericMTMDChatHandler as _VisionHandlerBase
except ImportError:
    _VisionHandlerBase = MTMDChatHandler

import folder_paths
from comfy_api.latest import io
from typing import Dict, Any, List
from .utils import image_to_data_uri

# --- llama-cpp-python 0.4.x capability detection & enum mappings -------------
# 0.4.x replaced the legacy `use_mmap` / `use_mlock` / `use_direct_io` booleans
# with `load_mode` + `lazy_mode`. On 0.4.x those old booleans are silently
# ignored (llama.py only prints a deprecation notice when verbose), so the
# Options toggles would otherwise be dead switches. Feature-detect on the actual
# Llama() signature instead of sniffing the version string, so 0.3.x installs
# keep their previous behaviour untouched.
_LLAMA_INIT_PARAMS = set(inspect.signature(Llama.__init__).parameters.keys())
_SUPPORTS_LOAD_MODE = "load_mode" in _LLAMA_INIT_PARAMS
_LEGACY_LOAD_KEYS = frozenset({"use_mmap", "use_mlock", "use_direct_io"})

# Combo label -> llama.cpp enum value. These mirror llama_cpp.llama_cpp's
# llama_load_mode / llama_lazy_mode / llama_flash_attn_type public enums.
_LOAD_MODE_VALUES = {
    "auto": -1, "none": 0, "mmap": 1, "mlock": 2, "mmap_mlock": 3, "direct_io": 4,
}
_LAZY_MODE_VALUES = {"off": 0, "auto": 1, "on": 2}
_FLASH_ATTN_VALUES = {"auto": -1, "disabled": 0, "enabled": 1}

# KV cache element types -> ggml_type values (ggml/include/ggml.h). llama.cpp
# defaults both K and V to GGML_TYPE_F16; quantizing them shrinks the KV cache
# proportionally to the bits per element.
_KV_TYPE_VALUES = {
    "f32": 0, "f16": 1, "q4_0": 2, "q4_1": 3, "q5_0": 6, "q5_1": 7,
    "q8_0": 8, "iq4_nl": 20, "bf16": 30,
}
# The subset that ggml_is_quantized() reports as quantized. A quantized V cache
# requires flash attention (see the validation in LlamaCPPEngine.execute).
_KV_QUANTIZED_VALUES = frozenset(
    _KV_TYPE_VALUES[k] for k in ("q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "iq4_nl")
)

# Options keys whose Combo string value must be converted to an int enum value.
_ENUM_OPTIONS = {
    "load_mode": _LOAD_MODE_VALUES,
    "lazy_mode": _LAZY_MODE_VALUES,
    "flash_attn_type": _FLASH_ATTN_VALUES,
    "type_k": _KV_TYPE_VALUES,
    "type_v": _KV_TYPE_VALUES,
}

# Config file path
CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config.json')

def load_config() -> Dict[str, Any]:
    """Load config from config.json, return empty dict if not found or invalid."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def get_user_model_folders() -> List[str]:
    """Get user-specified model folders from config."""
    config = load_config()
    return config.get('model_folders', [])

def get_merged_model_folders() -> List[str]:
    """Merge ComfyUI text_encoders folders, auto-detected LLM folder, and user folders."""
    try:
        comfy_folders = folder_paths.get_folder_paths("text_encoders")
        # Auto-detect ComfyUI\models\LLM\ folder
        llm_folder = os.path.join(folder_paths.models_dir, "LLM")
    except:
        comfy_folders = []
        llm_folder = None

    user_folders = get_user_model_folders()
    all_folders = comfy_folders + user_folders

    # Add auto-detected LLM folder if it exists and not already in the list
    if llm_folder and os.path.exists(llm_folder) and llm_folder not in all_folders:
        all_folders.append(llm_folder)

    # Filter out non-existent paths
    return [f for f in all_folders if os.path.exists(f)]

def scan_gguf_models_in_folders() -> List[str]:
    """Scan merged folders recursively for GGUF model files. Returns relative paths (e.g. 'GGUF/model.gguf')."""
    folders = get_merged_model_folders()
    model_list = []
    seen = set()
    for folder in folders:
        try:
            for root, dirs, files in os.walk(folder):
                # Skip hidden directories (e.g. .cache)
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                rel_dir = os.path.relpath(root, folder)
                for f in files:
                    if f.lower().endswith('.gguf'):
                        # Build relative path from the base folder
                        rel_path = f if rel_dir == '.' else os.path.join(rel_dir, f)
                        if rel_path not in seen:
                            model_list.append(rel_path)
                            seen.add(rel_path)
        except:
            pass  # Skip inaccessible folders
    return model_list

def find_model_path(model_name: str) -> str:
    """Find full path to model in merged folders. Handles relative paths from recursive scan."""
    folders = get_merged_model_folders()
    for folder in folders:
        path = os.path.join(folder, model_name)
        if os.path.exists(path):
            return path
    return None

# Global LLM instance for persistence
_global_llm = None

def _cleanup_global_llm(mode: str):
    """Helper function to cleanup the global LLM based on mode."""
    global _global_llm
    if mode == "persistent":
        return  # No cleanup

    # Common cleanup for all non-persistent modes
    if _global_llm is not None:
        # Clean up chat_handler if it exists (for vision models)
        if hasattr(_global_llm, "chat_handler") and _global_llm.chat_handler is not None:
            try:
                chat_handler = _global_llm.chat_handler
                
                # Primary Fix: Close ExitStack if present (used in newer llama-cpp-python versions)
                # This triggers the callback to mtmd_free -> mtmd_cpp.mtmd_free
                if hasattr(chat_handler, "_exit_stack") and chat_handler._exit_stack is not None:
                    try:
                        chat_handler._exit_stack.close()
                    except Exception:
                        pass

                # Secondary Fix: Manual cleanup of attributes as fallback
                # Attempt to find and close the CLIP/mmproj model embedded in the handler
                clip_attrs = ["clip_model", "_clip_model", "mmproj", "_mmproj", "clf"]
                
                for attr_name in clip_attrs:
                    if hasattr(chat_handler, attr_name):
                        attr = getattr(chat_handler, attr_name)
                        if attr is not None:
                            # Try to close/free the model if a method exists
                            if hasattr(attr, "close"):
                                try:
                                    attr.close()
                                except Exception:
                                    pass
                            elif hasattr(attr, "__del__"):
                                try:
                                    attr.__del__()
                                except Exception:
                                    pass
                                    
                            # Explicitly remove the attribute to break reference cycles
                            setattr(chat_handler, attr_name, None)
                            
                _global_llm.chat_handler.close()
            except Exception:
                pass  # Ignore cleanup errors for chat_handler
            
            # Remove the handler callback/reference from the LLM if possible/accessible
            if hasattr(_global_llm, "chat_handler"):
                del _global_llm.chat_handler
            # _global_llm might not have chat_handler attribute in strict sense if it was just in kwargs,
            # but we are cleaning up the attribute we accessed.
            
            gc.collect()
            gc.collect()
        try:
            _global_llm.close()
        except AttributeError:
            pass  # close() method may not be available in all versions
        del _global_llm
        _global_llm = None
        gc.collect()

    # Backend free cleanup for backend_free and full_cleanup modes
    if mode in ["backend_free", "full_cleanup"]:
        try:
            llama_backend_free()
            gc.collect()
        except (NameError, AttributeError):
            pass  # llama_backend_free may not be available

    # Full torch cleanup for full_cleanup mode
    if mode == "full_cleanup":
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()


# Mapping of chat formats to vision chat handlers (dynamically discovered)
import llama_cpp.llama_chat_format as lcf

VISION_HANDLERS = {}
for name, obj in inspect.getmembers(lcf):
    if inspect.isclass(obj) and issubclass(obj, MTMDChatHandler):
        vision_name = f"vision-{name.lower().replace('chathandler', '')}"
        VISION_HANDLERS[vision_name] = obj


class LlamaCPPModelLoader(io.ComfyNode):
    """Loads a GGUF model and optional mmproj for vision."""
    @classmethod
    def define_schema(cls) -> io.Schema:
        model_list = scan_gguf_models_in_folders()

        # Filter models based on criteria (case-insensitive)
        model_name_list = [f for f in model_list if 'mmproj' not in f.lower() and 'draft' not in f.lower()]
        mmproj_list = [f for f in model_list if 'mmproj' in f.lower()]

        # Add "None" option to make mmproj truly optional
        if mmproj_list:
            mmproj_list.insert(0, "None")
        else:
            mmproj_list = ["None"]

        # Available chat formats from llama-cpp-python
        base_chat_formats = sorted(list(LlamaChatCompletionHandlerRegistry()._chat_handlers.keys()))
        vision_formats = list(VISION_HANDLERS.keys())
        chat_formats = sorted(base_chat_formats + vision_formats)

        return io.Schema(
            node_id="LlamaCPPModelLoader",
            display_name="Llama CPP Model Loader",
            category="LlamaCPP",
            inputs=[
                io.Combo.Input("model_name", options=model_name_list if model_name_list else ["No GGUF models found"], tooltip="Select GGUF model file"),
                io.Combo.Input("chat_format", options=chat_formats, default="llama-2", tooltip="Chat format template", optional=True),
                io.Combo.Input("mmproj_model_name", options=mmproj_list, default="None", tooltip="Multi-modal projector model for vision (select 'None' to disable)", optional=True),
            ],
            outputs=[
                io.Custom("LLAMA_MODEL").Output(display_name="MODEL")
            ]
        )

    @classmethod
    def execute(cls, model_name: str, chat_format: str = "llama-2", mmproj_model_name: str = "None") -> io.NodeOutput:
        try:
            model_path = find_model_path(model_name)

            if model_path is None:
                raise FileNotFoundError(f"Model file not found: {model_name}")

            if not model_name.lower().endswith('.gguf'):
                raise ValueError(f"Selected file is not a GGUF model: {model_name}")

            model_info = {
                "model_path": model_path,
                "chat_format": chat_format,
            }

            # Handle mmproj if provided and not "None"
            if mmproj_model_name and mmproj_model_name != "None":
                mmproj_model_path = find_model_path(mmproj_model_name)
                if mmproj_model_path is None:
                    raise FileNotFoundError(f"Multi-modal projector model not found: {mmproj_model_name}")
                model_info["mmproj_model_path"] = mmproj_model_path

            # Return model info dict
            return io.NodeOutput(model_info)

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")


class LlamaCPPOptions(io.ComfyNode):
    """Configures Llama-cpp-python initialization parameters."""
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LlamaCPPOptions",
            display_name="Llama CPP Options",
            category="LlamaCPP",
            inputs=[
                io.Int.Input("n_gpu_layers", default=-1, min=-1, max=100, tooltip="Number of GPU layers to use", optional=True),
                io.Int.Input("n_ctx", default=8192, min=-1, max=262144, tooltip="Context window size (0 for max, -1 for default). 8192 is the tested floor for large images: a big image prefill can consume 4000+ tokens, leaving too little headroom for the first decode slot at 2048/4096.", optional=True),
                io.Int.Input("n_threads", default=-1, min=-1, max=256, tooltip="Number of threads (-1 for auto)", optional=True),
                io.Int.Input("n_threads_batch", default=-1, min=-1, max=256, tooltip="Number of threads per batch (-1 for auto)", optional=True),
                io.Int.Input("n_batch", default=2048, min=-1, max=16384, tooltip="Batch size (-1 for default)", optional=True),
                io.Int.Input("n_ubatch", default=512, min=-1, max=16384, tooltip="Micro batch size (-1 for default)", optional=True),
                io.Int.Input("main_gpu", default=0, min=-1, max=100, tooltip="GPU ID for main device (-1 for default)", optional=True),
                io.Boolean.Input("offload_kqv", default=True, tooltip="Enable offloading of K/Q/V tensors to GPU", optional=True),
                io.Boolean.Input("numa", default=False, tooltip="Enable NUMA affinity", optional=True),
                io.Boolean.Input("use_mmap", default=True, tooltip="[DEPRECATED in 0.4.x] Ignored by llama-cpp-python 0.4.x — use load_mode instead. Still honored by 0.3.x.", optional=True),
                io.Boolean.Input("use_mlock", default=False, tooltip="[DEPRECATED in 0.4.x] Ignored by llama-cpp-python 0.4.x — use load_mode='mlock' instead. Still honored by 0.3.x.", optional=True),
                io.Boolean.Input("use_direct_io", default=False, tooltip="[DEPRECATED in 0.4.x] Ignored by llama-cpp-python 0.4.x — use load_mode='direct_io' instead. Still honored by 0.3.x.", optional=True),
                io.Boolean.Input("verbose", default=False, tooltip="Enable verbose logging", optional=True),
                io.Int.Input("ctx_checkpoints", default=-1, min=-1, max=1024, tooltip="Context checkpoints (-1 = use llama-cpp-python default 16; 0 disables). In llama-cpp-python >= 0.3.48, 0 forces hybrid vision models down a 'Bypassing rollback' fast-path that fails at first decode on large images (e.g. 4000+ vision tokens). Keep -1 for vision.", optional=True),
                io.Int.Input("checkpoint_interval", default=4096, min=1, max=262144, tooltip="Hybrid model checkpoint token interval (llama-cpp-python default 4096). How often context checkpoints are created for rollback on hybrid (Mamba) models.", optional=True),
                io.Boolean.Input("checkpoint_on_device", default=False, tooltip="Store hybrid/recurrent checkpoint tensor payloads in device (GPU) buffers via LLAMA_STATE_SEQ_FLAGS_ON_DEVICE instead of host memory. Faster restores, but uses VRAM. Default False.", optional=True),
                io.Boolean.Input("vision_use_gpu", default=True, tooltip="Vision: Enable GPU for vision handler", optional=True),
                io.Int.Input("vision_image_min_tokens", default=1024, min=-1, max=16384, tooltip="Vision: Minimum image tokens (1024+ recommended for Qwen-VL, -1 for default)", optional=True),
                io.Int.Input("vision_image_max_tokens", default=-1, min=-1, max=16384, tooltip="Vision: Maximum image tokens (-1 for default)", optional=True),
                # --- llama-cpp-python 0.4.x tuning knobs ------------------------
                # Deliberately appended at the END of the input list. ComfyUI maps
                # saved workflow widget values POSITIONALLY, so inserting a new
                # input in the middle shifts the stored value of every widget
                # after it (that is how load_mode ended up showing a boolean).
                io.Combo.Input("load_mode", options=["auto", "none", "mmap", "mlock", "mmap_mlock", "direct_io"], default="auto", tooltip="[0.4.x] How model weights are loaded. 'auto' lets llama.cpp detect. 'mmap' memory-maps the file, 'mlock' pins it in RAM. No effect on 0.3.x (use the legacy toggles above).", optional=True),
                io.Combo.Input("lazy_mode", options=["auto", "off", "on"], default="auto", tooltip="[0.4.x] Lazy tensor loading (requires mmap). 'auto' = lazy only for tensors larger than 4 GiB. No effect on 0.3.x.", optional=True),
                io.Combo.Input("flash_attn_type", options=["auto", "disabled", "enabled"], default="auto", tooltip="[0.4.x] Flash Attention for the main model. 'auto' lets llama.cpp decide. No effect on 0.3.x.", optional=True),
                io.Combo.Input("vision_flash_attn", options=["auto", "enabled", "disabled"], default="auto", tooltip="[0.4.x] Flash Attention inside the vision encoder. 'auto' leaves the handler default (None). Can speed up clip_encode, which dominates large-image latency. No effect on 0.3.x.", optional=True),
                io.Combo.Input("type_k", options=["auto", "f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl", "bf16"], default="auto", tooltip="[0.4.x] KV cache data type for K (llama.cpp --cache-type-k). 'auto' keeps f16. At n_ctx=8192 the KV cache is a large slice of VRAM, so q8_0 roughly halves it and q4_0 cuts it to ~a quarter. The type's block size must divide the head dimension (32 divides 256 on Qwen3.5). No effect on 0.3.x.", optional=True),
                io.Combo.Input("type_v", options=["auto", "f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl", "bf16"], default="auto", tooltip="[0.4.x] KV cache data type for V (llama.cpp --cache-type-v). 'auto' keeps f16. IMPORTANT: a quantized V cache requires Flash Attention — llama.cpp refuses to build the context when flash_attn_type='disabled', so this plugin rejects that combination up front. No effect on 0.3.x.", optional=True),
            ],
            outputs=[
                io.Custom("LLAMA_OPTIONS").Output(display_name="OPTIONS")
            ]
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        try:
            # Filter out None/empty values
            options = {k: v for k, v in kwargs.items() if v is not None and v != ""}

            return io.NodeOutput(options)

        except Exception as e:
            raise RuntimeError(f"Failed to process options: {str(e)}")


class LlamaCPPEngine(io.ComfyNode):
    """Executes chat completion using the loaded model."""
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LlamaCPPEngine",
            display_name="Llama CPP Engine",
            category="LlamaCPP",
            inputs=[
                io.Custom("LLAMA_MODEL").Input("model", tooltip="Loaded Llama model from Model Loader"),
                io.String.Input("prompt", multiline=True, tooltip="User prompt for chat completion"),
                io.Image.Input("images", tooltip="Input image(s) for vision models (supports batches)", optional=True),
                io.Custom("LLAMA_OPTIONS").Input("options", tooltip="Model options from Options node", optional=True),
                io.String.Input("system_prompt", multiline=True, default="", tooltip="System prompt", optional=True),
                io.Combo.Input("memory_cleanup", options=["close", "backend_free", "full_cleanup", "persistent"], default="full_cleanup", tooltip="Memory cleanup method after generation", optional=True),
                io.DynamicCombo.Input("response_format", options=[
                    io.DynamicCombo.Option("text", inputs=[]),
                    io.DynamicCombo.Option("json_object", inputs=[
                        io.String.Input("json_schema", multiline=True, default="{}", tooltip="JSON Schema to enforce if json_object is selected", optional=True),
                    ])
                ], tooltip="Output format (json_object forces valid JSON)", optional=True),
                io.Int.Input("max_tokens", default=2048, min=1, max=262144, tooltip="Maximum tokens to generate (2048 covers detailed image descriptions)", optional=True),
                io.Float.Input("temperature", default=0.1, min=0.0, max=10.0, step=0.01, tooltip="Sampling temperature", optional=True),
                io.Float.Input("top_p", default=0.9, min=0.0, max=1.0, step=0.01, tooltip="Top-p sampling", optional=True),
                io.Int.Input("top_k", default=20, min=0, max=400, tooltip="Top-k sampling", optional=True),
                io.Float.Input("min_p", default=0.05, min=0.0, max=1.0, step=0.01, tooltip="Min-p sampling", optional=True),
                io.Float.Input("repeat_penalty", default=1.1, min=1.0, max=5.0, step=0.01, tooltip="Repeat penalty", optional=True),
                io.Float.Input("present_penalty", default=0.0, min=0.0, max=5.0, step=0.01, tooltip="Present penalty", optional=True),
                io.Float.Input("frequency_penalty", default=0.0, min=0.0, max=5.0, step=0.01, tooltip="Frequency penalty", optional=True),
                io.Int.Input("seed", default=1, min=-sys.maxsize, max=sys.maxsize, control_after_generate=True, tooltip="Random seed", optional=True),
                io.Combo.Input("think_display", options=["show", "hide"], default="hide", tooltip="Control <think>...</think> reasoning output. 'show' keeps it in the response; 'hide' strips it so only the final answer is returned.", optional=True),
            ],
            outputs=[
                io.String.Output(display_name="RESPONSE")
            ]
        )

    @classmethod
    def execute(cls, model: Dict[str, Any], prompt: str, images: torch.Tensor = None, options: Dict[str, Any] = None, system_prompt: str = "", memory_cleanup: str = "full_cleanup", response_format: Dict[str, Any] = {"type": "text"}, max_tokens: int = 2048, temperature: float = 0.1, top_p: float = 0.9, top_k: int = 20, min_p: float = 0.05, repeat_penalty: float = 1.1, present_penalty: float = 0.0, frequency_penalty: float = 0.0, seed: int = 1, think_display: str = "hide") -> io.NodeOutput:
        global _global_llm
        try:
            # Validate inputs
            if not isinstance(model, dict) or "model_path" not in model:
                raise ValueError("Invalid model data received from Model Loader")

            options = options or {}
            if not isinstance(options, dict):
                raise ValueError("Invalid options data received from Options node")

            if not prompt.strip():
                raise ValueError("Prompt cannot be empty")

            # Extract model info
            model_path = model["model_path"]
            chat_format = model["chat_format"]

            # Determine if vision handler is enabled
            vision_enabled = "mmproj_model_path" in model and chat_format.startswith("vision-")

            # Create messages for chat completion
            messages = []
            if system_prompt.strip():
                messages.append({"role": "system", "content": system_prompt.strip()})

            # Create user message content
            if vision_enabled and images is not None:
                user_content = [{"type": "text", "text": prompt.strip()}]
                # images is (B, H, W, C)
                B = images.shape[0]
                for i in range(B):
                    img_tensor = images[i]  # (H, W, C)
                    data_uri = image_to_data_uri(img_tensor)
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": data_uri}
                    })
                messages.append({"role": "user", "content": user_content})
            else:
                messages.append({"role": "user", "content": prompt.strip()})

            # Prepare Llama initialization parameters
            llama_kwargs = {
                "model_path": model_path,
                "chat_format": chat_format,
            }

            # Add options. Filter against the installed Llama() signature so one
            # code path serves both 0.3.x and 0.4.x — params introduced in 0.4.x
            # are dropped instead of leaking into **kwargs on an older build.
            for k, v in options.items():
                if k.startswith("vision_"):
                    continue
                # Whitelist for -1 values: only n_gpu_layers is allowed to pass -1
                if v == -1 and k != "n_gpu_layers":
                    continue
                if k not in _LLAMA_INIT_PARAMS:
                    continue
                # 0.4.x: legacy load booleans are no-ops superseded by load_mode
                if k in _LEGACY_LOAD_KEYS and _SUPPORTS_LOAD_MODE:
                    continue
                # Combo label ("auto"/"mmap"/...) -> llama.cpp enum int.
                # Anything unrecognised (e.g. a stale workflow that stored another
                # widget's value in this slot) falls back to the library default
                # instead of forwarding garbage like load_mode=False to llama.cpp.
                if k in _ENUM_OPTIONS:
                    _label = str(v).strip().lower()
                    if _label == "auto":
                        continue
                    _mapped = _ENUM_OPTIONS[k].get(_label)
                    if _mapped is None:
                        continue
                    v = _mapped
                llama_kwargs[k] = v

            # Handle vision models: pass mmproj_path to Llama() and let it
            # create the chat handler internally (required by llama-cpp-python >= 0.3.39)
            if vision_enabled:
                llama_kwargs["mmproj_path"] = model["mmproj_model_path"]

                # Collect vision_ prefixed options into chat_handler_kwargs.
                # Llama() creates GenericMTMDChatHandler (0.3.39+) internally, which
                # forwards **kwargs to MTMDChatHandler.__init__. MTMDChatHandler
                # raises TypeError for any unrecognized kwargs, so we must filter
                # against the union of both classes' explicit params.
                accepted_params = set(inspect.signature(_VisionHandlerBase).parameters.keys())
                accepted_params |= set(inspect.signature(MTMDChatHandler).parameters.keys())
                accepted_params -= {"self", "kwargs"}

                chat_handler_kwargs = {}
                for k, v in options.items():
                    if k.startswith("vision_"):
                        param_name = k.replace("vision_", "")
                        if param_name not in accepted_params:
                            continue
                        # vision_flash_attn is a tri-state Combo while the handler
                        # expects Optional[bool]. Only explicit on/off values are
                        # forwarded; "auto" or anything unrecognised keeps the
                        # handler default (None) so a stale workflow cannot
                        # silently disable Flash Attention.
                        if param_name == "flash_attn":
                            _fa = str(v).strip().lower()
                            if _fa in ("true", "1", "on", "enabled"):
                                chat_handler_kwargs[param_name] = True
                            elif _fa in ("false", "0", "off", "disabled"):
                                chat_handler_kwargs[param_name] = False
                            continue
                        # Skip -1 values (use handler defaults)
                        if v == -1:
                            continue
                        chat_handler_kwargs[param_name] = v

                # Guard the vision token bounds. GenericMTMDChatHandler raises
                # "image_max_tokens cannot be less than image_min_tokens" whenever
                # 0 < image_max_tokens < image_min_tokens, which is easy to hit
                # when a stale workflow carries an unrelated number in the
                # max_tokens widget. Drop the offending bound (falling back to the
                # handler default) instead of failing the whole generation.
                _vmin = chat_handler_kwargs.get("image_min_tokens")
                _vmax = chat_handler_kwargs.get("image_max_tokens")
                if (
                    isinstance(_vmax, int) and _vmax > 0
                    and isinstance(_vmin, int) and _vmax < _vmin
                ):
                    print(
                        f"[llama_cpp vision] WARNING: vision_image_max_tokens ({_vmax}) "
                        f"is less than vision_image_min_tokens ({_vmin}); ignoring the "
                        f"max bound. Set vision_image_max_tokens to -1 (default) or "
                        f">= {_vmin} to silence this.",
                        file=sys.stderr,
                    )
                    chat_handler_kwargs.pop("image_max_tokens", None)

                llama_kwargs["chat_handler_kwargs"] = chat_handler_kwargs
                # Mirror Options' verbose setting into the vision chat handler
                if "verbose" in options:
                    chat_handler_kwargs.setdefault("verbose", options["verbose"])
                # Let Llama() auto-detect chat_format for vision models
                llama_kwargs.pop("chat_format", None)

            # KV cache quantization and Flash Attention are coupled in llama.cpp:
            # llama-context.cpp rejects a quantized V cache while flash attention
            # is disabled ("V cache quantization requires flash_attn") and returns
            # a null context. Validate up front so the user gets an actionable
            # message instead of an opaque context-creation failure.
            _v_type = llama_kwargs.get("type_v")
            if (
                _v_type in _KV_QUANTIZED_VALUES
                and llama_kwargs.get("flash_attn_type") == _FLASH_ATTN_VALUES["disabled"]
            ):
                _v_label = next(
                    (k for k, v in _KV_TYPE_VALUES.items() if v == _v_type), str(_v_type)
                )
                raise ValueError(
                    f"type_v='{_v_label}' (quantized KV cache) requires Flash Attention, "
                    f"but flash_attn_type is 'disabled'. Set flash_attn_type to 'auto' or "
                    f"'enabled' on the Llama CPP Options node, or set type_v back to "
                    f"'auto' (f16) and retry."
                )

            # Echo the resolved 0.4.x tuning parameters when verbose is on, so it
            # is possible to confirm from the console that a Combo actually made
            # it through the enum mapping (and was not dropped as an unknown or
            # stale value).
            if options.get("verbose"):
                _tuning = {
                    _k: llama_kwargs.get(_k)
                    for _k in ("load_mode", "lazy_mode", "flash_attn_type", "type_k", "type_v")
                }
                _tuning["vision_chat_handler_kwargs"] = llama_kwargs.get(
                    "chat_handler_kwargs", {}
                )
                print(f"[llama_cpp] resolved tuning params: {_tuning}")

            # Pre-generation LLM management
            if memory_cleanup == "persistent":
                if _global_llm is None:
                    _global_llm = Llama(**llama_kwargs)
            else:  # close or backend_free - always cleanup existing and create new
                _cleanup_global_llm(memory_cleanup)
                _global_llm = Llama(**llama_kwargs)

            # Prepare response_format parameter
            response_format_param = None
            if isinstance(response_format, dict):
                fmt_type = response_format.get("response_format", "text")
                if fmt_type == "json_object":
                    response_format_param = {"type": "json_object"}
                    json_schema = response_format.get("json_schema", "{}").strip()
                    if json_schema:
                        try:
                            response_format_param["schema"] = json.loads(json_schema)
                        except json.JSONDecodeError:
                            response_format_param["schema"] = json_schema
                else:
                    response_format_param = {"type": fmt_type}
            else:
                # Fallback for simple string if it happens
                response_format_param = {"type": str(response_format)}

            # Generate response using global LLM
            response = _global_llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                present_penalty=present_penalty,
                frequency_penalty=frequency_penalty,
                min_p=min_p,
                seed=seed,
                response_format=response_format_param,
            )

            # Extract the response text
            if not response or "choices" not in response or not response["choices"]:
                raise RuntimeError("No response generated by the model")

            response_text = response["choices"][0]["message"]["content"]

            # Strip the whole <think> reasoning block (tags + content) when hiding,
            # keeping only the final answer. This is pure output post-processing:
            # the model still reasons internally, but only the answer is returned.
            # Models emit the block inconsistently — well-formed <think>..</think>,
            # missing the opening tag, with multiple </think>, or a dangling open —
            # so the robust rule is: the answer always follows the LAST </think>.
            if str(think_display).lower() == "hide":
                last_close = response_text.rfind("</think>")
                if last_close != -1:
                    response_text = response_text[last_close + len("</think>"):]
                else:
                    m = re.search(r"<think>", response_text)
                    if m:
                        response_text = response_text[:m.start()]
                response_text = re.sub(r"<think>|<//think>", "", response_text).strip()

            # Post-generation memory cleanup
            _cleanup_global_llm(memory_cleanup)

        except Exception as e:
            _err = str(e)
            # Reactive detection: the hybrid recurrent memory backend crashes
            # cryptically when n_ctx has no headroom left for the first decode
            # slot after a large image prefill ("failed to find a memory slot
            # for batch of size 1"). Translate it into an actionable hint.
            _hint = ""
            if "memory slot for batch of size" in _err or "Failed completely even with batch size" in _err:
                _nctx = options.get("n_ctx", 2048)
                _hint = (
                    "  Hint: the model ran out of KV/decoding slots on the first decode "
                    "token — this usually means n_ctx (currently %d) is too small for the "
                    "image you loaded. Increase n_ctx in the Options node (8192 is known-good "
                    "for large images; use a higher value for even larger images) and retry."
                    % _nctx
                )
            response_text = f"Error generating response: {_err}{_hint}"
            print(f"LlamaCPP Engine Error: {_err}{_hint}")

        return io.NodeOutput(response_text)


class LlamaCPPMemoryCleanup(io.ComfyNode):
    """Manually triggers memory cleanup."""
    @classmethod
    def define_schema(cls) -> io.Schema:
        template = io.MatchType.Template("passthrough")
        return io.Schema(
            node_id="LlamaCPPMemoryCleanup",
            display_name="Llama CPP Memory Cleanup",
            category="LlamaCPP",
            inputs=[
                io.Combo.Input("memory_cleanup", options=["close", "backend_free", "full_cleanup", "persistent"], default="close", tooltip="Memory cleanup method"),
                io.MatchType.Input("passthrough", template=template, tooltip="Any input to pass through", optional=True),
            ],
            outputs=[
                io.MatchType.Output(template=template, display_name="PASSTHROUGH")
            ]
        )

    @classmethod
    def execute(cls, memory_cleanup: str, passthrough=None) -> io.NodeOutput:
        try:
            _cleanup_global_llm(memory_cleanup)
        except Exception as e:
            print(f"LlamaCPP Memory Cleanup Error: {str(e)}")

        return io.NodeOutput(passthrough)
