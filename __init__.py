from .nodes import LlamaCPPModelLoader, LlamaCPPOptions, LlamaCPPEngine, LlamaCPPMemoryCleanup
from .dataset_nodes import LlamaCPPImageFolderLoader, LlamaCPPCaptionSaver
from comfy_api.latest import ComfyExtension, io

class SGLlamaExtension(ComfyExtension):
    """Extension for Llama-cpp nodes."""
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LlamaCPPModelLoader,
            LlamaCPPOptions,
            LlamaCPPEngine,
            LlamaCPPMemoryCleanup,
            LlamaCPPImageFolderLoader,
            LlamaCPPCaptionSaver,
        ]

async def comfy_entrypoint() -> SGLlamaExtension:
    return SGLlamaExtension()
