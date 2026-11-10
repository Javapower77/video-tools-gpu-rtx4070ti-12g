"""Single-frame FaceDetailer client; no ComfyUI/node code is imported locally.

Deployment contract (Windows, Python 3.11): run a *dedicated app instance* of
ComfyUI on loopback, with Impact Pack and Impact Subpack already installed.
Put Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors in its models/checkpoints
and face_yolov8m.pt in models/ultralytics/bbox. Nothing is installed/downloaded
here. COMFYUI_URL defaults to http://127.0.0.1:8188; COMFYUI_TIMEOUT defaults
to 900 seconds per frame (including upload, execution and output retrieval).

restore reads the comfy_* fields exposed by Settings and the UI. Inputs and outputs remain on
the server in unique video_enhancer/<job UUID> folders; its API has no scoped
file-delete route. Only dest frames are written locally. Same seed per frame
reduces one source of variation, but does NOT guarantee temporal consistency.

Research: upstream raw server.py and main.py (comfyanonymous/ComfyUI/master),
modules/impact/impact_pack.py (ltdrdata/ComfyUI-Impact-Pack/Main), and
modules/subpack_nodes.py (ltdrdata/ComfyUI-Impact-Subpack/main), 2026-10-02.
FaceDetailer output 0 is the full composited image; outputs 1/2 are face crops.
Detector output 0 is BBOX_DETECTOR. /free only sets worker flags: HTTP 200 is
NOT a GPU-release acknowledgement. We additionally wait for idle queues and
stable bounded allocator reservations on every CUDA device via /system_stats. A failed or
unverifiable handoff raises, so callers MUST NOT start RIFE on failure.

The legacy API cannot make queue-check + /free atomic. A dedicated instance
with no other submitters is therefore mandatory, not merely a performance tip.
Use a current ComfyUI supporting targeted /interrupt {prompt_id}; never send a
global interrupt. Cleanup has a separate bounded deadline and ignores the
already-set cancellation event while waiting for the owned job to leave.
"""
from __future__ import annotations

import copy
import http.client
import io
import ipaddress
import json
import logging
import math
import os
from pathlib import Path
import threading
import sys
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlencode, urlsplit
import uuid

from PIL import Image

from .common import Report, check_cancel

if TYPE_CHECKING:
    from .config import Settings

WORKFLOW = Path(__file__).resolve().parents[1] / "workflows/facedetailer_api.json"
JSON_LIMIT = 8 * 1024 * 1024
IMAGE_LIMIT = 128 * 1024 * 1024
REQUEST_TIMEOUT = 10.0
POLL_INTERVAL = 0.25
CLEANUP_TIMEOUT = 60.0
ASYNC_RESIDUAL_LIMIT = 128 * 1024 * 1024
LOGGER = logging.getLogger(__name__)
CLASSES = {
    "1": "LoadImage", "2": "CheckpointLoaderSimple", "3": "CLIPTextEncode",
    "4": "CLIPTextEncode", "5": "UltralyticsDetectorProvider",
    "6": "FaceDetailer", "7": "SaveImage",
}
LINKS = {
    "3": {"clip": ["2", 1]}, "4": {"clip": ["2", 1]},
    "6": {"image": ["1", 0], "model": ["2", 0], "clip": ["2", 1],
          "vae": ["2", 2], "positive": ["3", 0], "negative": ["4", 0],
          "bbox_detector": ["5", 0]},
    "7": {"images": ["6", 0]},
}


class ComfyUIError(RuntimeError):
    """Protocol, execution, configuration, or GPU handoff failure."""


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ComfyUIError("ComfyUI deadline exceeded; GPU handoff is not confirmed.")
    return remaining


class ComfyUIClient:
    """Bounded direct stdlib HTTP, no proxies, DNS rebinding or redirects."""

    def __init__(self, url: str | None = None, timeout: float | None = None):
        parsed = urlsplit(url if url is not None else os.getenv(
            "COMFYUI_URL", "http://127.0.0.1:8188"))
        host = parsed.hostname
        if host == "localhost":
            host = "127.0.0.1"  # Never resolve a configurable DNS name.
        try:
            local = host is not None and ipaddress.ip_address(host).is_loopback
            port = parsed.port or 8188
        except ValueError as error:
            raise ValueError("COMFYUI_URL must use a literal loopback address.") from error
        if (parsed.scheme not in ("http", "https") or not local or
                parsed.username is not None or parsed.password is not None or
                parsed.path not in ("", "/") or parsed.query or parsed.fragment):
            raise ValueError("COMFYUI_URL must be a localhost-only HTTP(S) origin, no path or credentials.")
        self.host, self.port, self.secure = host, port, parsed.scheme == "https"
        self.timeout = float(timeout if timeout is not None else os.getenv("COMFYUI_TIMEOUT", "900"))
        if not math.isfinite(self.timeout) or not 0 < self.timeout <= 86400:
            raise ValueError("COMFYUI_TIMEOUT must be finite, > 0 and <= 86400 seconds.")

    def request(self, method: str, route: str, deadline: float, *,
                body: bytes | None = None, content_type: str = "application/json",
                limit: int = JSON_LIMIT) -> bytes:
        if not route.startswith("/") or route.startswith("//") or "\r" in route or "\n" in route:
            raise ValueError("Invalid ComfyUI API route.")
        if body is not None and len(body) > IMAGE_LIMIT + 4096:
            raise ComfyUIError("ComfyUI request body exceeds size limit.")
        connection_type = http.client.HTTPSConnection if self.secure else http.client.HTTPConnection
        connection = connection_type(self.host, self.port,
                                     timeout=min(REQUEST_TIMEOUT, _remaining(deadline)))
        try:
            connection.request(method, route, body=body, headers={
                "Content-Type": content_type, "Accept-Encoding": "identity"})
            connection.timeout = min(REQUEST_TIMEOUT, _remaining(deadline))
            if connection.sock is not None:
                connection.sock.settimeout(connection.timeout)
            response = connection.getresponse()
            length = response.getheader("Content-Length")
            if length is not None and (int(length) < 0 or int(length) > limit):
                raise ComfyUIError(f"ComfyUI {route}: response exceeds {limit} bytes.")
            data = bytearray()
            while True:
                remaining = min(REQUEST_TIMEOUT, _remaining(deadline))
                if connection.sock is not None:
                    connection.sock.settimeout(remaining)
                chunk = response.read1(min(65536, limit + 1 - len(data)))
                if not chunk:
                    break
                data.extend(chunk)
                if len(data) > limit:
                    raise ComfyUIError(f"ComfyUI {route}: response exceeds {limit} bytes.")
            _remaining(deadline)
            if not 200 <= response.status < 300:
                diagnostic = bytes(data[:2000]).decode("utf-8", errors="replace")
                raise ComfyUIError(f"ComfyUI {method} {route}: HTTP {response.status}: {diagnostic}")
            return bytes(data)
        except (OSError, ValueError, http.client.HTTPException) as error:
            raise ComfyUIError(f"ComfyUI {method} {route} failed: {error}") from error
        finally:
            connection.close()

    def api(self, method: str, route: str, deadline: float,
            payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
        data = self.request(method, route, deadline, body=body)
        if not data:  # /queue, /interrupt and /free return an empty HTTP 200.
            if method == "GET":
                raise ComfyUIError(f"ComfyUI {route}: empty JSON response.")
            return {}
        try:
            result = json.loads(data)
        except (ValueError, UnicodeError) as error:
            raise ComfyUIError(f"ComfyUI {route}: invalid JSON.") from error
        if not isinstance(result, dict):
            raise ComfyUIError(f"ComfyUI {route}: expected JSON object.")
        return result

    def queue(self, deadline: float) -> tuple[list, list]:
        result = self.api("GET", "/queue", deadline)
        groups = []
        for key in ("queue_running", "queue_pending"):
            items = result.get(key)
            if not isinstance(items, list) or any(
                    not isinstance(item, list) or len(item) < 4 or
                    not isinstance(item[1], str) or not isinstance(item[3], dict) for item in items):
                raise ComfyUIError(f"ComfyUI /queue: malformed {key}; ownership cannot be verified.")
            groups.append(items)
        return groups[0], groups[1]

    def stop_owned(self, prompt_id: str | None, client_id: str, deadline: float) -> None:
        """Recover ambiguous submissions by unique client_id; never clear queue."""
        waiter = threading.Event()
        interrupted: set[str] = set()
        while True:
            running, pending = self.queue(deadline)
            owned = [item for item in running + pending if
                     item[1] == prompt_id or item[3].get("client_id") == client_id]
            if not owned:
                return
            ids = list({item[1] for item in owned})
            self.api("POST", "/queue", deadline, {"delete": ids})
            # Recheck after deletion: a pending job may have started meanwhile.
            running, _ = self.queue(deadline)
            for item in running:
                if item[1] in ids and item[1] not in interrupted:
                    self.api("POST", "/interrupt", deadline, {"prompt_id": item[1]})
                    interrupted.add(item[1])
            waiter.wait(min(POLL_INTERVAL, _remaining(deadline)))

    def release(self, deadline: float) -> None:
        running, pending = self.queue(deadline)
        if running or pending:
            raise ComfyUIError("ComfyUI queue is not idle; refusing /free. Use a dedicated app instance; do not start RIFE.")
        self.api("POST", "/free", deadline, {"unload_models": True, "free_memory": True})
        waiter = threading.Event()
        previous = None
        last_evidence = None
        while True:
            running, pending = self.queue(deadline)
            if running or pending:
                raise ComfyUIError("Another ComfyUI prompt appeared during GPU handoff; do not start RIFE.")
            devices = self.api("GET", "/system_stats", deadline).get("devices")
            if not isinstance(devices, list) or not devices:
                raise ComfyUIError("ComfyUI /system_stats lacks device memory evidence; do not start RIFE.")
            released = True
            evidence = []
            for device in devices:
                if not isinstance(device, dict) or "type" not in device:
                    raise ComfyUIError("ComfyUI /system_stats has malformed device evidence.")
                if device["type"] == "cuda":
                    reserved = device.get("torch_vram_total")
                    if type(reserved) not in (int, float) or not math.isfinite(reserved) or reserved < 0:
                        raise ComfyUIError("ComfyUI /system_stats lacks valid torch_vram_total.")
                    # cudaMallocAsync retains a small allocator pool after /free.
                    # Still fail closed on model-sized allocations or unknown allocators.
                    asynchronous = "cudamallocasync" in str(device.get("name", "")).lower()
                    limit = ASYNC_RESIDUAL_LIMIT if asynchronous else 0
                    released = released and reserved <= limit
                    evidence.append((device.get("index"), reserved, limit))
                elif device["type"] != "cpu":
                    raise ComfyUIError("Unsupported ComfyUI device; GPU release cannot be verified.")
            last_evidence = evidence
            # /free is asynchronous: require two consecutive identical idle,
            # below-cap observations rather than treating HTTP 200 as release.
            if released and previous == evidence:
                LOGGER.info("ComfyUI GPU handoff: idle queue, stable reserved-byte evidence=%s", evidence)
                return
            previous = evidence if released else None
            try:
                waiter.wait(min(POLL_INTERVAL, _remaining(deadline)))
            except ComfyUIError as error:
                raise ComfyUIError(f"ComfyUI GPU release deadline exceeded; reserved evidence={last_evidence}") from error


def load_workflow(settings: Settings) -> dict[str, Any]:
    """Keep editable scalar inputs, but enforce exactly seven nodes and links."""
    try:
        with WORKFLOW.open("rb") as stream:
            raw = stream.read(JSON_LIMIT + 1)
        if len(raw) > JSON_LIMIT:
            raise ValueError("workflow exceeds size limit")
        graph = json.loads(raw)
        if not isinstance(graph, dict) or set(graph) != set(CLASSES):
            raise ValueError("expected exactly fixed node IDs 1 through 7")
        for node_id, class_type in CLASSES.items():
            node = graph[node_id]
            if node.get("class_type") != class_type or not isinstance(node.get("inputs"), dict):
                raise ValueError(f"node {node_id} must be {class_type}")
            for name, value in node["inputs"].items():
                if isinstance(value, (list, dict)) and value != LINKS.get(node_id, {}).get(name):
                    raise ValueError(f"unauthorized connection {node_id}.{name}")
            for name, link in LINKS.get(node_id, {}).items():
                if node["inputs"].get(name) != link:
                    raise ValueError(f"required connection {node_id}.{name} must be {link}")
        graph["2"]["inputs"]["ckpt_name"] = settings.comfy_checkpoint
        graph["5"]["inputs"]["model_name"] = settings.comfy_detector
        graph["3"]["inputs"]["text"] = settings.comfy_positive
        graph["4"]["inputs"]["text"] = settings.comfy_negative
        for name in ("denoise", "steps", "cfg", "seed", "guide_size"):
            graph["6"]["inputs"][name] = getattr(settings, f"comfy_{name}")
        # Wildcards change seeds/prompts per detected face. This contract forbids them.
        if graph["6"]["inputs"].get("wildcard") != "":
            raise ValueError("wildcard must remain empty for fixed-seed frame processing")
        return graph
    except (OSError, ValueError, TypeError, AttributeError) as error:
        raise ComfyUIError(f"FaceDetailer workflow/settings invalid: {error}") from error


def validate_workflow(graph: dict[str, Any], info: dict[str, Any]) -> None:
    """Fail explicitly on missing nodes, required inputs, enums or link types."""
    for node_id, node in graph.items():
        class_type, inputs = node["class_type"], node["inputs"]
        schema = info.get(class_type)
        if not isinstance(schema, dict) or not isinstance(schema.get("input"), dict):
            raise ComfyUIError(f"ComfyUI lacks {class_type}. Check Impact Pack / Impact Subpack installation and restart.")
        required = schema["input"].get("required", {})
        optional = schema["input"].get("optional", {})
        if not isinstance(required, dict) or not isinstance(optional, dict):
            raise ComfyUIError(f"Malformed /object_info schema for {class_type}.")
        missing = set(required) - set(inputs)
        unknown = set(inputs) - (set(required) | set(optional))
        if missing or unknown:
            raise ComfyUIError(f"{class_type} input mismatch: missing {sorted(missing)}, unsupported {sorted(unknown)}. Update workflow/upstream nodes; no fallback.")
        if bool(schema.get("output_node")) != (node_id == "7"):
            raise ComfyUIError("SaveImage node 7 must be the only output node.")
        for name, value in inputs.items():
            spec = (required | optional)[name]
            if not isinstance(spec, list) or not spec:
                raise ComfyUIError(f"Malformed /object_info input {class_type}.{name}.")
            kind = spec[0]
            options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            label = f"{class_type}.{name}"
            if isinstance(value, list):
                outputs = info.get(graph[value[0]]["class_type"], {}).get("output", [])
                if value[1] >= len(outputs) or outputs[value[1]] != kind:
                    raise ComfyUIError(f"{label}: upstream output connection type changed.")
            elif isinstance(kind, list):
                # LoadImage enumerates files before upload (including only root files
                # in some versions). The upload receipt validates this controlled input.
                if node_id == "1" and name == "image":
                    continue
                if value not in kind:
                    hint = "models/checkpoints" if name == "ckpt_name" else "models/ultralytics/bbox" if name == "model_name" else "node choices"
                    raise ComfyUIError(f"{label}: {value!r} unavailable; check {hint}. Available: {kind[:20]!r}; no fallback.")
            elif kind in ("INT", "FLOAT"):
                if (type(value) not in ((int,) if kind == "INT" else (int, float)) or
                        not math.isfinite(value) or value < options.get("min", -math.inf) or
                        value > options.get("max", math.inf)):
                    raise ComfyUIError(f"{label}: invalid {kind} value {value!r} or outside upstream limits.")
            elif kind == "BOOLEAN" and type(value) is not bool:
                raise ComfyUIError(f"{label}: expected boolean.")
            elif kind == "STRING" and not isinstance(value, str):
                raise ComfyUIError(f"{label}: expected string.")
            elif kind not in ("INT", "FLOAT", "BOOLEAN", "STRING"):
                raise ComfyUIError(f"{label}: expected connected {kind} input.")


def _png(data: bytes, size: tuple[int, int] | None = None) -> tuple[int, int]:
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG" or getattr(image, "n_frames", 1) != 1:
                raise ValueError("expected one static PNG frame")
            if size is not None and image.size != size:
                raise ValueError(f"output dimensions {image.size} differ from original {size}")
            dimensions = image.size
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            image.load()  # verify checks structure; load also detects corrupt pixel data.
        return dimensions
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise ComfyUIError(f"Invalid ComfyUI PNG frame: {error}") from error


def _safe_path(value: Any, *, filename: bool = False) -> str:
    # ComfyUI SaveImage uses os.path on its server, returning backslashes on
    # Windows. Normalize first, then apply the same traversal/ownership checks.
    if isinstance(value, str) and not filename:
        value = value.replace("\\", "/")
    if (not isinstance(value, str) or (filename and not value) or
            any(char in value for char in ("\\", ":", "\x00", "\r", "\n", "[", "]")) or
            value.startswith("/") or any(part in (".", "..") for part in value.split("/")) or
            (filename and "/" in value)):
        raise ComfyUIError("ComfyUI returned an unsafe image path.")
    return value


def upload_frame(client: ComfyUIClient, data: bytes, filename: str,
                 folder: str, deadline: float) -> str:
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in (("type", "input"), ("subfolder", folder), ("overwrite", "false")):
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.extend([f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{filename}"\r\nContent-Type: image/png\r\n\r\n'.encode(),
                  data, f'\r\n--{boundary}--\r\n'.encode()])
    raw = client.request("POST", "/upload/image", deadline, body=b"".join(parts),
                         content_type=f"multipart/form-data; boundary={boundary}")
    try:
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ValueError("expected upload object")
        name = _safe_path(result.get("name"), filename=True)
        subfolder = _safe_path(result.get("subfolder"))
        if result.get("type") != "input" or subfolder != folder or not name.endswith(".png"):
            raise ValueError("upload receipt does not match owned input folder/PNG type")
        return f"{subfolder}/{name} [input]"
    except (ValueError, UnicodeError) as error:
        raise ComfyUIError(f"Invalid ComfyUI upload receipt: {error}") from error


def wait_output(client: ComfyUIClient, prompt_id: str, cancel: threading.Event,
                deadline: float) -> dict[str, Any]:
    while True:
        check_cancel(cancel)
        history = client.api("GET", f"/history/{quote(prompt_id, safe='')}", deadline)
        entry = history.get(prompt_id)
        if entry is not None:
            if not isinstance(entry, dict) or not isinstance(entry.get("status"), dict):
                raise ComfyUIError("ComfyUI returned malformed execution history.")
            status = entry["status"]
            messages = status.get("messages", [])
            if not isinstance(messages, list):
                raise ComfyUIError("ComfyUI returned malformed execution messages.")
            for message in messages:
                if isinstance(message, list) and len(message) == 2 and message[0] in (
                        "execution_error", "execution_interrupted"):
                    raise ComfyUIError(f"ComfyUI {message[0]}: {str(message[1])[:2000]}")
            if status.get("status_str") == "error":
                raise ComfyUIError(f"ComfyUI execution failed: {str(messages)[:2000]}")
            if status.get("completed") is True and status.get("status_str") == "success":
                outputs = entry.get("outputs", {})
                images = outputs.get("7", {}).get("images") if isinstance(outputs, dict) and isinstance(outputs.get("7"), dict) else None
                if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], dict):
                    raise ComfyUIError("SaveImage node 7 must return exactly one PNG, not face crops/previews.")
                return images[0]
        if cancel.wait(min(POLL_INTERVAL, _remaining(deadline))):
            check_cancel(cancel)


def restore(source: Path, dest: Path, count: int, settings: Settings,
            cancel: threading.Event, report: Report) -> None:
    """Return only after owned prompts stop and GPU handoff is verified.

    On cleanup timeout/unreachable server, raises ComfyUIError, NOT success or
    ordinary cancellation: the caller must halt the pipeline, not run RIFE.
    """
    check_cancel(cancel)
    if type(count) is not int or count < 1:
        raise ValueError("ComfyUI restore requires a positive integer frame count.")
    client = ComfyUIClient()
    graph = load_workflow(settings)
    preflight = time.monotonic() + min(client.timeout, CLEANUP_TIMEOUT)
    validate_workflow(graph, client.api("GET", "/object_info", preflight))
    running, pending = client.queue(preflight)
    if running or pending:
        raise ComfyUIError("ComfyUI must be idle at stage start. Use a dedicated app instance.")
    check_cancel(cancel)
    dest.mkdir()  # Match the existing face-restorer's fresh-destination contract.
    client_id = str(uuid.uuid4())
    folder = f"video_enhancer/{uuid.uuid4().hex}"
    owned_id: str | None = None
    submission_attempted = False
    try:
        for index in range(count):
            check_cancel(cancel)
            deadline = time.monotonic() + client.timeout
            filename = f"{index:08d}.png"
            with (source / filename).open("rb") as stream:
                data = stream.read(IMAGE_LIMIT + 1)
            if len(data) > IMAGE_LIMIT:
                raise ComfyUIError("Input PNG exceeds 128 MiB limit.")
            size = _png(data)
            frame_graph = copy.deepcopy(graph)
            frame_graph["1"]["inputs"]["image"] = upload_frame(client, data, filename, folder, deadline)
            frame_graph["7"]["inputs"]["filename_prefix"] = f"{folder}/{index:08d}"
            check_cancel(cancel)
            owned_id = str(uuid.uuid4())
            submission_attempted = True
            result = client.api("POST", "/prompt", deadline, {
                "prompt": frame_graph, "client_id": client_id, "prompt_id": owned_id})
            # Older ComfyUI versions mint their own ID; use the returned ID.
            returned_id = result.get("prompt_id")
            if not isinstance(returned_id, str) or not returned_id or len(returned_id) > 128:
                raise ComfyUIError(f"ComfyUI rejected prompt: {str(result)[:2000]}")
            owned_id = returned_id
            if result.get("error") or result.get("node_errors"):
                raise ComfyUIError(f"ComfyUI prompt validation failed: {str(result)[:2000]}")
            image = wait_output(client, owned_id, cancel, deadline)
            filename_out = _safe_path(image.get("filename"), filename=True)
            subfolder = _safe_path(image.get("subfolder"))
            if image.get("type") != "output" or subfolder != folder or not filename_out.endswith(".png"):
                raise ComfyUIError("SaveImage output is not a PNG in this job's owned output folder.")
            output = client.request("GET", "/view?" + urlencode({
                "filename": filename_out, "subfolder": subfolder, "type": "output"}),
                deadline, limit=IMAGE_LIMIT)
            _png(output, size)
            check_cancel(cancel)
            (dest / filename).write_bytes(output)
            # History may be visible slightly before the running queue is updated.
            client.stop_owned(owned_id, client_id, time.monotonic() + CLEANUP_TIMEOUT)
            owned_id = None
            report((index + 1) / count, f"ComfyUI FaceDetailer: {index + 1}/{count}")
    finally:
        original_error = sys.exception()
        if submission_attempted:
            cleanup_deadline = time.monotonic() + CLEANUP_TIMEOUT
            try:
                client.stop_owned(owned_id, client_id, cleanup_deadline)
                client.release(cleanup_deadline)
            except Exception as error:
                detail = f" Initial failure: {original_error}" if original_error is not None else ""
                LOGGER.exception("ComfyUI cleanup failed; original failure=%r", original_error)
                raise ComfyUIError(f"ComfyUI cleanup/GPU release unconfirmed; do not start RIFE: {error}.{detail}") from error
    check_cancel(cancel)