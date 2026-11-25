"""Dataset helper nodes for batch captioning.

The plugin's second use case is reverse-engineering prompts for a whole folder of
images (LoRA dataset tagging). Every existing combination of third-party nodes
breaks at least one step of that pipeline:

* folder loaders that hand over a *tensor batch* instead of a list -> the Engine
  puts all N images into one request and runs out of context
* caption savers whose filenames do not match the source image (prefixed
  counters) or whose Windows path handling mangles "F:\\..." into "F_\\..."
* savers that happily write the Engine's "Error generating response: ..." string
  into a .txt, silently poisoning the dataset

These two nodes own that pipeline end to end, are deliberately small, and keep
the Engine untouched.

Design notes
------------
* The loader returns ``[1, H, W, C]`` items so the Engine's ``images.shape[0]``
  batch loop sees exactly one image per call.
* ``not_idempotent`` on the loader forces a rescan on every queue, otherwise
  ComfyUI's value cache would keep serving a stale file list after images were
  added or removed from the folder.
* The saver is ``is_input_list=True`` so it receives the full caption list in one
  call and writes all files, instead of running once per image.
"""

import os
import re

import numpy as np
import torch
from PIL import Image, ImageOps

import folder_paths
from comfy_api.latest import io

CATEGORY = "LlamaCPP/Dataset"

_RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS

# "Error generating response: ..." is what LlamaCPPEngine returns instead of
# raising, so a failed image would otherwise be written as a caption.
_ERROR_PREFIX = "error generating response"

_NATURAL_SPLIT = re.compile(r"(\d+)")


def _natural_key(value: str):
    """Sort key that keeps 2.png before 10.png."""
    return [int(part) if part.isdigit() else part.lower() for part in _NATURAL_SPLIT.split(value)]


def _resolve_path(raw: str) -> str:
    """Absolute paths are used verbatim; relative ones resolve under ComfyUI/input."""
    value = str(raw or "").strip().strip('"').strip("'")
    if not value:
        raise ValueError("Empty path")
    value = os.path.expandvars(os.path.expanduser(value))
    if not os.path.isabs(value):
        value = os.path.join(folder_paths.get_input_directory(), value)
    return os.path.normpath(value)


def _parse_extensions(extensions: str) -> set:
    exts = set()
    for chunk in str(extensions or "").split(","):
        chunk = chunk.strip().lower().lstrip(".")
        if chunk:
            exts.add("." + chunk)
    return exts or {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _scan_images(root: str, recursive: bool, extensions: str) -> list:
    exts = _parse_extensions(extensions)
    found = []
    if recursive:
        for current, dirs, names in os.walk(root):
            # Hidden dirs (.cache, .git) and __pycache__ never hold datasets
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
            for name in names:
                if os.path.splitext(name)[1].lower() in exts:
                    found.append(os.path.join(current, name))
    else:
        for name in os.listdir(root):
            full = os.path.join(root, name)
            if os.path.isfile(full) and os.path.splitext(name)[1].lower() in exts:
                found.append(full)
    return found


def _load_image_tensor(path: str, max_side: int = 0) -> torch.Tensor:
    """Load one image as ``[1, H, W, C]`` float32 in 0..1."""
    with Image.open(path) as handle:
        image = ImageOps.exif_transpose(handle)
        if image.mode in ("RGBA", "LA", "P"):
            # Transparent PNG would otherwise lose its background to black,
            # which is a very different picture for a captioning model.
            image = image.convert("RGBA")
            canvas = Image.new("RGB", image.size, (255, 255, 255))
            canvas.paste(image, mask=image.split()[-1])
            image = canvas
        elif image.mode != "RGB":
            image = image.convert("RGB")

        if max_side and max_side > 0 and max(image.size) > max_side:
            scale = max_side / float(max(image.size))
            size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            image = image.resize(size, _RESAMPLE)

        array = np.asarray(image, dtype=np.float32) / 255.0

    return torch.from_numpy(array).unsqueeze(0)


def _as_list(value) -> list:
    """Normalise a possibly-absent / possibly-scalar value into a list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _scalar(value, default=None):
    """Widget inputs arrive wrapped in a 1-element list when is_input_list is set."""
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return default if value is None else value


class LlamaCPPImageFolderLoader(io.ComfyNode):
    """Loads every image of a folder as a LIST so downstream nodes run per image."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LlamaCPPImageFolderLoader",
            display_name="Llama CPP Image Folder Loader",
            category=CATEGORY,
            description=(
                "Load all images of a folder as a LIST. A downstream non-list node "
                "(Llama CPP Engine) then runs once per image and returns one caption each."
            ),
            not_idempotent=True,
            inputs=[
                io.String.Input(
                    "folder_path",
                    default="",
                    tooltip="Folder to scan. Absolute path is used as-is; a relative path resolves under ComfyUI/input.",
                ),
                io.String.Input(
                    "extensions",
                    default="png,jpg,jpeg,webp,bmp",
                    tooltip="Comma separated list of image extensions to pick up.",
                ),
                io.Boolean.Input(
                    "recursive",
                    default=True,
                    tooltip="Also scan sub-folders.",
                ),
                io.Combo.Input(
                    "sort_order",
                    options=["natural", "name", "modified", "none"],
                    default="natural",
                    tooltip="natural keeps 2.png before 10.png; modified sorts by mtime (useful after generating a batch).",
                ),
                io.Int.Input(
                    "max_images",
                    default=0,
                    min=0,
                    max=100000,
                    tooltip="0 = all images. Use a small number to validate a workflow before a full run.",
                ),
                io.Int.Input(
                    "start_index",
                    default=0,
                    min=0,
                    max=100000,
                    tooltip="Skip the first N images after sorting, to resume an interrupted run.",
                ),
                io.Int.Input(
                    "max_side",
                    default=0,
                    min=0,
                    max=8192,
                    tooltip=(
                        "Downscale so the longest side is at most this many pixels (0 = keep original). "
                        "Directly reduces vision tokens and base64 size; 1024 is a good captioning default."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(id="images", display_name="images", is_output_list=True),
                io.String.Output(id="paths", display_name="paths", is_output_list=True),
                io.Int.Output(id="count", display_name="count"),
            ],
        )

    @classmethod
    def execute(cls, folder_path: str, extensions: str = "png,jpg,jpeg,webp,bmp",
                recursive: bool = True, sort_order: str = "natural",
                max_images: int = 0, start_index: int = 0, max_side: int = 0) -> io.NodeOutput:
        root = _resolve_path(folder_path)
        if not os.path.isdir(root):
            raise ValueError(f"Image folder not found: {root}")

        files = _scan_images(root, bool(recursive), extensions)
        if not files:
            raise ValueError(
                f"No images with extension '{extensions}' found in {root}"
                + ("" if recursive else " (recursive is off, so sub-folders were skipped)")
            )

        if sort_order == "name":
            files.sort(key=lambda p: os.path.basename(p).lower())
        elif sort_order == "natural":
            files.sort(key=lambda p: _natural_key(os.path.relpath(p, root)))
        elif sort_order == "modified":
            files.sort(key=lambda p: os.path.getmtime(p))

        start = max(0, int(start_index or 0))
        if start:
            files = files[start:]
        limit = int(max_images or 0)
        if limit > 0:
            files = files[:limit]

        images = []
        paths = []
        for path in files:
            try:
                images.append(_load_image_tensor(path, int(max_side or 0)))
            except Exception as exc:  # unreadable / truncated file
                print(f"[llama_cpp dataset] WARNING: skipped unreadable image {path}: {exc}")
                continue
            paths.append(path)

        if not images:
            raise ValueError(f"Found {len(files)} candidate image(s) in {root} but none could be decoded.")

        print(
            f"[llama_cpp dataset] loaded {len(images)} image(s) from {root}"
            + (f" | max_side={max_side}" if max_side else "")
            + (f" | sorted by {sort_order}" if sort_order != "none" else "")
        )
        return io.NodeOutput(images, paths, len(images))


class LlamaCPPCaptionSaver(io.ComfyNode):
    """Writes one .txt per caption, by default next to the source image (same stem)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LlamaCPPCaptionSaver",
            display_name="Llama CPP Caption Saver",
            category=CATEGORY,
            description=(
                "Write one .txt per caption. By default the file lands next to the source image "
                "with the same stem (kohya-style pairing) and never re-encodes the image."
            ),
            is_input_list=True,
            is_output_node=True,
            not_idempotent=True,
            inputs=[
                io.String.Input(
                    "text",
                    force_input=True,
                    tooltip="Caption(s) from Llama CPP Engine.",
                ),
                io.String.Input(
                    "paths",
                    force_input=True,
                    optional=True,
                    tooltip=(
                        "Image path(s) from Llama CPP Image Folder Loader. "
                        "Leave unconnected to write numbered files into output_dir instead."
                    ),
                ),
                io.String.Input(
                    "output_dir",
                    default="",
                    tooltip=(
                        "Empty = write next to each source image. Relative paths resolve under ComfyUI/input."
                    ),
                ),
                io.Boolean.Input(
                    "skip_on_error",
                    default=True,
                    label_on="skip",
                    label_off="keep",
                    tooltip=(
                        "Drop captions that are the Engine's error string "
                        "('Error generating response: ...') instead of writing them into the dataset."
                    ),
                ),
                io.Boolean.Input(
                    "skip_empty",
                    default=True,
                    label_on="skip",
                    label_off="keep",
                    tooltip="Drop blank captions instead of writing an empty .txt.",
                ),
            ],
            outputs=[
                io.Int.Output(id="saved", display_name="saved"),
                io.Int.Output(id="skipped", display_name="skipped"),
            ],
        )

    @classmethod
    def execute(cls, text, paths=None, output_dir: str = "",
                skip_on_error: bool = True, skip_empty: bool = True) -> io.NodeOutput:
        captions = _as_list(text)
        if len(captions) == 1 and captions[0] is None:
            captions = []
        # Optional list inputs arrive as (None,) when unconnected under
        # INPUT_IS_LIST, so drop placeholders instead of treating them as paths.
        sources = [s for s in _as_list(paths) if s is not None and str(s).strip()]
        drop_errors = bool(_scalar(skip_on_error, True))
        drop_empty = bool(_scalar(skip_empty, True))

        raw_dir = str(_scalar(output_dir, "") or "").strip()
        target_dir = _resolve_path(raw_dir) if raw_dir else None

        if not captions:
            raise ValueError("No caption received; connect the Engine's RESPONSE output to 'text'.")

        if sources and len(sources) != len(captions):
            raise ValueError(
                f"Got {len(captions)} caption(s) but {len(sources)} path(s). "
                "Connect both lists from the same image folder loader."
            )

        if not sources and target_dir is None:
            raise ValueError(
                "'paths' is not connected, so 'output_dir' must be set to an absolute folder."
            )

        saved = 0
        skipped = 0
        for index, raw_caption in enumerate(captions):
            body = "" if raw_caption is None else str(raw_caption).strip()

            if drop_empty and not body:
                skipped += 1
                print(f"[llama_cpp dataset] skipped empty caption at index {index}")
                continue
            if drop_errors and body.lower().startswith(_ERROR_PREFIX):
                skipped += 1
                print(
                    f"[llama_cpp dataset] skipped error caption at index {index}: "
                    f"{body[:100]}"
                )
                continue

            if sources:
                source = str(sources[index]).strip()
                if not os.path.isabs(source):
                    source = os.path.join(folder_paths.get_input_directory(), source)
                stem = os.path.splitext(os.path.basename(source))[0]
                directory = target_dir or os.path.dirname(os.path.abspath(source))
            else:
                stem = f"caption_{index:05d}"
                directory = target_dir

            os.makedirs(directory, exist_ok=True)
            destination = os.path.join(directory, stem + ".txt")
            with open(destination, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(body + "\n")

            saved += 1
            print(f"[llama_cpp dataset] caption saved: {destination} ({len(body)} chars)")

        print(f"[llama_cpp dataset] captions: {saved} saved, {skipped} skipped")
        return io.NodeOutput(saved, skipped)
