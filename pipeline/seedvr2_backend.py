"""SeedVR2 frame-folder stage, Windows/Python 3.11; no local node imports.

Contract: upscale(source, dest, count, fps, settings, cancel, report,
                  *, batch_mode=False) -> None.
Source is contiguous 00000000.png onward; dest must not exist. FPS/count are
unchanged, and the runner owns final audio muxing. Settings require FHD/4K;
seedvr_batch_size (1/5/9), seedvr_noise (0..1), seedvr_max_frames (default 240)
are optional duck-typed fields. Single mode additionally caps duration at 8s:
GetVideoComponents and SaveImage materialize the entire IMAGE batch in RAM.
Batch mode uses the contained ComfyUI Python and streaming CLI (45 new frames).

Verified against numz pack 2.5.24, commit
4490bd1f482e026674543386bb2a4d176da245b9: src/interfaces/{dit_model_loader,
vae_model_loader,video_upscaler}.py, inference_cli.py, utils/{downloads,
model_registry}.py. Core schemas: local comfy_extras/nodes_video.py and nodes.py.
The CLI has no --vae_model; prepend=0 avoids its first-chunk duplication bug.
Existing weights are SHA256 checked because upstream otherwise re-downloads
corrupt weights. No installs, downloads, model-cache hacks, or vendor edits.

Requires a dedicated loopback ComfyUI using vendor/ComfyUI's default input,
output and model directories, and no competing submitters during either mode.
Queue checking/freeing cannot be atomic in the legacy API. Cleanup failure is
fatal (not ordinary cancellation); owned server files are retained if prompt
termination is unconfirmed. Temporary intermediates are removed otherwise.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
from typing import Any
from urllib.parse import quote, urlencode
import uuid

from PIL import Image

from .common import Report, check_cancel, ffmpeg, run_process
from .comfyui_backend import (
    CLEANUP_TIMEOUT, IMAGE_LIMIT, JSON_LIMIT, POLL_INTERVAL,
    ComfyUIClient, ComfyUIError, _png, _remaining, _safe_path,
)
from .resolution import target_dimensions

ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT / "vendor/ComfyUI"
WORKFLOW = ROOT / "workflows/seedvr2_api.json"
DIT_MODEL = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
VAE_MODEL = "ema_vae_fp16.safetensors"
MODEL_HASHES = {
    DIT_MODEL: "3bf1e43ebedd570e7e7a0b1b60d6a02e105978f505c8128a241cde99a8240cff",
    VAE_MODEL: "20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1",
}
CLASSES = {"1": "LoadVideo", "2": "GetVideoComponents",
           "3": "SeedVR2LoadDiTModel", "4": "SeedVR2LoadVAEModel",
           "5": "SeedVR2VideoUpscaler", "7": "SaveImage"}
LINKS = {"2": {"video": ["1", 0]},
         "5": {"image": ["2", 0], "dit": ["3", 0], "vae": ["4", 0]},
         "7": {"images": ["5", 0]}}


def _contained(path: Path, parent: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(parent.resolve()):
        raise ComfyUIError(f"SeedVR2 path escapes its local directory: {path}")
    return resolved


def check_models(cancel: threading.Event) -> Path:
    """Read-only integrity check; never let upstream repair missing/bad weights."""
    model_dir = _contained(COMFY_ROOT / "models/SEEDVR2", COMFY_ROOT)
    for name, expected in MODEL_HASHES.items():
        check_cancel(cancel)
        path = _contained(model_dir / name, model_dir)
        if not path.is_file() or not path.stat().st_size:
            raise ComfyUIError(f"SeedVR2 requires an existing local model: {path}; no automatic downloads.")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                check_cancel(cancel)
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ComfyUIError(f"SeedVR2 model integrity failed: {path}; refusing upstream re-download.")
    return model_dir


def _options(count: int, fps: Fraction, settings: Any, batch_mode: bool) -> tuple[int, float]:
    if type(count) is not int or count < 1:
        raise ValueError("SeedVR2 requires a positive integer frame count.")
    if not isinstance(fps, Fraction) or not 0 < fps <= 1000:
        raise ValueError("SeedVR2 requires positive exact Fraction FPS <= 1000.")
    if settings.output_resolution not in ("FHD", "4K"):
        raise ValueError("SeedVR2 output_resolution must be FHD or 4K.")
    batch = getattr(settings, "seedvr_batch_size", 5)
    noise = getattr(settings, "seedvr_noise", 0.03)
    if type(batch) is not int or batch not in (1, 5, 9):
        raise ValueError("SeedVR2 12 GB batch size must be 1, 5 or 9 (4n+1).")
    if type(noise) not in (int, float) or not math.isfinite(noise) or not 0 <= noise <= 1:
        raise ValueError("SeedVR2 noise must be finite and between 0 and 1.")
    if not batch_mode:
        bound = getattr(settings, "seedvr_max_frames", 240)
        if type(bound) is not int or bound < 1:
            raise ValueError("SeedVR2 single-mode frame bound must be a positive integer.")
        if Fraction(count) / fps > 8 or count > min(bound, 240):
            raise ValueError("SeedVR2 single mode is limited to 8 seconds and its frame bound; use batch_mode=True for streaming.")
    return batch, float(noise)


def _read_png(path: Path) -> bytes:
    with path.open("rb") as stream:
        data = stream.read(IMAGE_LIMIT + 1)
    if len(data) > IMAGE_LIMIT:
        raise ComfyUIError("SeedVR2 PNG exceeds 128 MiB limit.")
    return data


def _source_size(source: Path, count: int, cancel: threading.Event) -> tuple[int, int]:
    if not source.is_dir():
        raise ValueError("SeedVR2 source must be a numbered PNG directory.")
    if sum(1 for p in source.iterdir() if p.suffix.lower() == ".png") != count:
        raise ValueError("SeedVR2 source PNG count differs from count.")
    size = None
    for index in range(count):
        check_cancel(cancel)
        path = source / f"{index:08d}.png"
        if not path.is_file():
            raise ValueError(f"Missing numbered source frame: {path.name}")
        current = _png(_read_png(path), size)
        size = current
    assert size is not None
    return size


def load_workflow(target: tuple[int, int], batch: int, noise: float,
                  resolution: str, video: str, folder: str) -> dict[str, Any]:
    try:
        with WORKFLOW.open("rb") as stream:
            raw = stream.read(JSON_LIMIT + 1)
        if len(raw) > JSON_LIMIT:
            raise ValueError("workflow exceeds size limit")
        graph = json.loads(raw)
        if not isinstance(graph, dict) or set(graph) != set(CLASSES):
            raise ValueError("expected the fixed SeedVR2 video graph")
        for node_id, class_type in CLASSES.items():
            node = graph[node_id]
            if not isinstance(node, dict) or node.get("class_type") != class_type or not isinstance(node.get("inputs"), dict):
                raise ValueError(f"node {node_id} must be {class_type}")
            for name, value in node["inputs"].items():
                if isinstance(value, (dict, list)) and value != LINKS.get(node_id, {}).get(name):
                    raise ValueError(f"unauthorized connection {node_id}.{name}")
            for name, link in LINKS.get(node_id, {}).items():
                if node["inputs"].get(name) != link:
                    raise ValueError(f"missing connection {node_id}.{name}")
        tile, overlap, blocks = (512, 64, 28) if resolution == "FHD" else (384, 48, 32)
        graph["1"]["inputs"]["file"] = video
        graph["3"]["inputs"].update(model=DIT_MODEL, device="cuda:0", blocks_to_swap=blocks,
                                     swap_io_components=True, offload_device="cpu",
                                     cache_model=False, attention_mode="sdpa")
        graph["4"]["inputs"].update(model=VAE_MODEL, device="cuda:0", encode_tiled=True,
                                     decode_tiled=True, encode_tile_size=tile, decode_tile_size=tile,
                                     encode_tile_overlap=overlap, decode_tile_overlap=overlap,
                                     offload_device="cpu", cache_model=False, tile_debug="false")
        graph["5"]["inputs"].update(seed=42, resolution=min(target), max_resolution=max(target),
                                     batch_size=batch, uniform_batch_size=True, temporal_overlap=3,
                                     prepend_frames=4, color_correction="lab", input_noise_scale=0.0,
                                     latent_noise_scale=noise, offload_device="cpu", enable_debug=False)
        graph["7"]["inputs"]["filename_prefix"] = f"{folder}/frames"
        return graph
    except (OSError, ValueError, TypeError) as error:
        raise ComfyUIError(f"SeedVR2 workflow invalid: {error}") from error


def validate_workflow(graph: dict[str, Any], info: dict[str, Any]) -> None:
    """Use legacy /object_info serialization of both core and V3 node schemas."""
    for node_id, node in graph.items():
        class_type, inputs = node["class_type"], node["inputs"]
        schema = info.get(class_type)
        if not isinstance(schema, dict) or not isinstance(schema.get("input"), dict):
            raise ComfyUIError(f"ComfyUI lacks {class_type}; install/restart SeedVR2 separately.")
        required = schema["input"].get("required", {})
        optional = schema["input"].get("optional", {})
        if not isinstance(required, dict) or not isinstance(optional, dict):
            raise ComfyUIError(f"Malformed schema for {class_type}.")
        if set(required) - set(inputs) or set(inputs) - (set(required) | set(optional)):
            raise ComfyUIError(f"{class_type} input mismatch; update workflow/upstream, no fallback.")
        if node_id == "7" and schema.get("output_node") is not True:
            raise ComfyUIError("SeedVR2 SaveImage must be an output node.")
        for name, value in inputs.items():
            spec = (required | optional)[name]
            if not isinstance(spec, list) or not spec:
                raise ComfyUIError(f"Malformed schema for {class_type}.{name}.")
            kind = spec[0]
            limits = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            valid = True
            if isinstance(value, list):
                outputs = info.get(graph[value[0]]["class_type"], {}).get("output", [])
                valid = isinstance(outputs, list) and value[1] < len(outputs) and outputs[value[1]] == kind
            elif isinstance(kind, list):
                # Core enumerates root videos only; validate_inputs checks local subfolders.
                valid = isinstance(value, str) if (node_id, name) == ("1", "file") else value in kind
            elif kind in ("INT", "FLOAT"):
                valid = (type(value) in ((int,) if kind == "INT" else (int, float)) and
                         math.isfinite(value) and limits.get("min", -math.inf) <= value <= limits.get("max", math.inf))
            elif kind == "BOOLEAN":
                valid = type(value) is bool
            elif kind == "STRING":
                valid = isinstance(value, str)
            else:
                valid = False
            if not valid:
                raise ComfyUIError(f"Invalid SeedVR2 schema value/link: {class_type}.{name}.")


def wait_outputs(client: ComfyUIClient, prompt_id: str, count: int,
                 cancel: threading.Event, deadline: float) -> list[dict[str, Any]]:
    """The existing wait_output is singleton-only; use its protocol for batches."""
    while True:
        check_cancel(cancel)
        history = client.api("GET", f"/history/{quote(prompt_id, safe='')}", deadline)
        entry = history.get(prompt_id)
        if entry is not None:
            if not isinstance(entry, dict) or not isinstance(entry.get("status"), dict):
                raise ComfyUIError("Malformed SeedVR2 execution history.")
            status = entry["status"]
            messages = status.get("messages", [])
            if not isinstance(messages, list):
                raise ComfyUIError("Malformed SeedVR2 execution messages.")
            if status.get("status_str") == "error" or any(
                    isinstance(m, list) and len(m) == 2 and m[0] in ("execution_error", "execution_interrupted")
                    for m in messages):
                raise ComfyUIError(f"SeedVR2 execution failed: {str(messages)[:2000]}")
            if status.get("completed") is True and status.get("status_str") == "success":
                outputs = entry.get("outputs", {})
                saved = outputs.get("7", {}) if isinstance(outputs, dict) else {}
                images = saved.get("images") if isinstance(saved, dict) else None
                if not isinstance(images, list) or len(images) != count or any(not isinstance(i, dict) for i in images):
                    raise ComfyUIError(f"SeedVR2 SaveImage must return exactly {count} frames; no trimming.")
                return images
        if cancel.wait(min(POLL_INTERVAL, _remaining(deadline))):
            check_cancel(cancel)


def _write_output(data: bytes, path: Path, target: tuple[int, int],
                  previous: tuple[int, int] | None) -> tuple[int, int]:
    size = _png(data, previous)
    # Model rounds dimensions for its spatial grid. Only allow aspect rounding,
    # not thumbnails, face crops, wildly different resolutions, or distortion.
    if any(abs(a - b) > 32 for a, b in zip(size, target)):
        raise ComfyUIError(f"SeedVR2 output dimensions {size} do not match target {target} (rounding allowance 32px).")
    with Image.open(io.BytesIO(data)) as image:
        image = image.convert("RGB")
        if size != target:
            image = image.resize(target, Image.Resampling.LANCZOS)
        image.save(path, format="PNG")
    return size


def cli_command(video: Path, output: Path, model_dir: Path, target: tuple[int, int],
                batch: int, noise: float, resolution: str) -> list[str]:
    python = _contained(COMFY_ROOT / ".venv/Scripts/python.exe", COMFY_ROOT)
    script = _contained(COMFY_ROOT / "custom_nodes/seedvr2_videoupscaler/inference_cli.py", COMFY_ROOT)
    missing = [str(path) for path in (python, script) if not path.is_file()]
    if missing:
        raise ComfyUIError(
            "SeedVR2 batch prerequisites missing:\n" + "\n".join(missing)
            + "\nRun scripts/Setup-ComfyUI.ps1 -SeedVR2 from this project, then restart "
            "the application. Model downloads alone do not install the CLI."
        )
    tile, overlap, blocks = (512, 64, 28) if resolution == "FHD" else (384, 48, 32)
    # Redirected stdout defaults to CP1252 on Windows; upstream prints emoji
    # during imports. Explicit UTF-8 mode works even without environment vars.
    return [str(python), "-X", "utf8", "-u", str(script), str(video.resolve()), "--output", str(output.resolve()),
            "--model_dir", str(model_dir.resolve()), "--dit_model", DIT_MODEL,
            "--resolution", str(min(target)), "--max_resolution", str(max(target)),
            "--batch_size", str(batch), "--uniform_batch_size", "--chunk_size", "45",
            "--temporal_overlap", "3", "--prepend_frames", "0", "--blocks_to_swap", str(blocks),
            "--swap_io_components", "--dit_offload_device", "cpu", "--vae_offload_device", "cpu",
            "--tensor_offload_device", "cpu", "--vae_encode_tiled", "--vae_decode_tiled",
            "--vae_encode_tile_size", str(tile), "--vae_decode_tile_size", str(tile),
            "--vae_encode_tile_overlap", str(overlap), "--vae_decode_tile_overlap", str(overlap),
            "--attention_mode", "sdpa", "--color_correction", "lab",
            "--latent_noise_scale", str(noise), "--input_noise_scale", "0",
            "--cuda_device", "0", "--output_format", "png"]


def upscale(source: Path, dest: Path, count: int, fps: Fraction, settings: Any,
            cancel: threading.Event, report: Report, *, batch_mode: bool = False) -> None:
    """Produce exactly count even-sized, eight-digit PNGs; return only after handoff."""
    check_cancel(cancel)
    batch, noise = _options(count, fps, settings, batch_mode)
    source, dest = Path(source).resolve(), Path(dest).resolve()
    if dest.exists() or dest == source or dest.is_relative_to(source):
        raise ValueError("SeedVR2 requires a fresh destination outside the source folder.")
    size = _source_size(source, count, cancel)
    target = target_dimensions(*size, settings.output_resolution)
    report(0.0, "SeedVR2: checking local model integrity")
    model_dir = check_models(cancel)
    client = ComfyUIClient()
    deadline = time.monotonic() + min(client.timeout, CLEANUP_TIMEOUT)
    running, pending = client.queue(deadline)
    if running or pending:
        raise ComfyUIError("SeedVR2 requires an idle dedicated ComfyUI instance.")
    token = uuid.uuid4().hex
    folder = f"seedvr2/{token}"
    input_dir = _contained(COMFY_ROOT / "input" / folder, COMFY_ROOT)
    output_dir = _contained(COMFY_ROOT / "output" / folder, COMFY_ROOT)
    work_root = ROOT / "temp"
    work_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="seedvr2_", dir=work_root))
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    client_id, owned_id = str(uuid.uuid4()), None
    attempted, safe_cleanup = False, True
    try:
        frames = work / "frames"
        frames.mkdir()
        if batch_mode:
            video = work / "input.mkv"
            command = cli_command(video, work / "cli", model_dir, target, batch, noise, settings.output_resolution)
        else:
            input_dir.mkdir(parents=True, exist_ok=False)
            video = input_dir / "input.mkv"
            graph = load_workflow(target, batch, noise, settings.output_resolution,
                                  f"{folder}/input.mkv [input]", folder)
            validate_workflow(graph, client.api("GET", "/object_info", deadline))
        report(0.02, "SeedVR2: preparing lossless video (original FPS, no audio)")
        run_process([ffmpeg(), "-hide_banner", "-nostdin", "-y", "-framerate", str(fps),
                     "-start_number", "0", "-i", str(source / "%08d.png"), "-map", "0:v:0",
                     "-frames:v", str(count), "-an", "-c:v", "ffv1", "-level", "3",
                     "-pix_fmt", "bgr0", str(video)], cancel, log_dir / f"seedvr2_{token}_prepare.log")
        if not video.is_file() or video.stat().st_size == 0:
            raise ComfyUIError("SeedVR2 temporary FFV1 video is missing/empty.")
        check_cancel(cancel)
        deadline = time.monotonic() + client.timeout
        running, pending = client.queue(deadline)
        if running or pending:
            raise ComfyUIError("ComfyUI became busy; refusing competing SeedVR2 GPU work.")
        previous = None
        if batch_mode:
            client.release(time.monotonic() + CLEANUP_TIMEOUT)
            report(0.05, "SeedVR2 CLI: streaming 45-frame chunks")
            def tick() -> None:
                produced = sum(1 for _ in (work / "cli/input").glob("*.png"))
                report(min(0.9, 0.05 + 0.85 * produced / count), f"SeedVR2 CLI: {produced}/{count} frames")
            run_process(command, cancel, log_dir / f"seedvr2_{token}_cli.log", tick)
            sequence = work / "cli" / video.stem
            paths = sorted(sequence.glob("*.png"))
            if len(paths) != count or [p.name for p in paths] != [f"{video.stem}_{i:06d}.png" for i in range(count)]:
                raise ComfyUIError(f"SeedVR2 CLI must produce exactly {count} contiguous frames; no trimming.")
            for index, path in enumerate(paths):
                check_cancel(cancel)
                previous = _write_output(_read_png(path), frames / f"{index:08d}.png", target, previous)
                report(0.9 + 0.09 * (index + 1) / count, f"SeedVR2: validating {index + 1}/{count}")
        else:
            owned_id = str(uuid.uuid4())
            attempted = True
            report(0.05, "SeedVR2 ComfyUI: processing bounded video batch")
            result = client.api("POST", "/prompt", deadline,
                                {"prompt": graph, "client_id": client_id, "prompt_id": owned_id})
            returned = result.get("prompt_id")
            if not isinstance(returned, str) or not returned or len(returned) > 128 or result.get("error") or result.get("node_errors"):
                raise ComfyUIError(f"SeedVR2 prompt rejected: {str(result)[:2000]}")
            owned_id = returned
            images = wait_outputs(client, owned_id, count, cancel, deadline)
            seen = set()
            for index, image in enumerate(images):
                check_cancel(cancel)
                name = _safe_path(image.get("filename"), filename=True)
                subfolder = _safe_path(image.get("subfolder"))
                if image.get("type") != "output" or subfolder != folder or not name.endswith(".png") or name in seen:
                    raise ComfyUIError("SeedVR2 returned duplicate/unsafe/non-owned output frames.")
                seen.add(name)
                data = client.request("GET", "/view?" + urlencode({"filename": name, "subfolder": subfolder, "type": "output"}),
                                      deadline, limit=IMAGE_LIMIT)
                previous = _write_output(data, frames / f"{index:08d}.png", target, previous)
                report(0.9 + 0.09 * (index + 1) / count, f"SeedVR2: retrieving {index + 1}/{count}")
        # Never publish partial frames or report success before verified GPU release.
        if attempted:
            cleanup = time.monotonic() + CLEANUP_TIMEOUT
            client.stop_owned(owned_id, client_id, cleanup)
            client.release(cleanup)
            attempted = False
        check_cancel(cancel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Remove owned server artifacts before publishing a successful result.
        for path in (input_dir, output_dir):
            if path.exists():
                shutil.rmtree(path)
        shutil.move(str(frames), str(dest))
    finally:
        original_error = sys.exception()
        try:
            if attempted:
                cleanup = time.monotonic() + CLEANUP_TIMEOUT
                client.stop_owned(owned_id, client_id, cleanup)
                client.release(cleanup)
        except Exception as error:
            safe_cleanup = False
            raise ComfyUIError(f"SeedVR2 cleanup/GPU handoff unconfirmed; do not start RIFE. Initial failure: {original_error}; cleanup: {error}") from error
        finally:
            try:
                shutil.rmtree(work)
            finally:
                if safe_cleanup:
                    # Only unique owned directories; never clear global outputs/history.
                    try:
                        if input_dir.exists():
                            shutil.rmtree(input_dir)
                    finally:
                        if output_dir.exists():
                            shutil.rmtree(output_dir)
    report(1.0, f"SeedVR2 complete: {count} frames at {fps} FPS, {target[0]}x{target[1]}")