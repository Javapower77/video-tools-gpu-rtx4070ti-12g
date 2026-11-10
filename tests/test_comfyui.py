"""Offline protocol tests: no ComfyUI, model loading, installs or inference."""
import copy
import http.client
import io
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from pipeline import comfyui_backend as backend
from pipeline.common import CancelledError


def png(size=(32, 24), format="PNG"):
    buffer = io.BytesIO()
    Image.new("RGB", size, "navy").save(buffer, format=format)
    return buffer.getvalue()


def settings():
    return SimpleNamespace(
        comfy_checkpoint="Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
        comfy_detector="bbox/face_yolov8m.pt", comfy_positive="natural face",
        comfy_negative="blurry", comfy_denoise=0.3, comfy_steps=20,
        comfy_cfg=5.0, comfy_seed=42, comfy_guide_size=512)


def object_info(graph):
    """Wire types are explicit; scalar schemas emulate upstream /object_info."""
    outputs = {"1": ["IMAGE", "MASK"], "2": ["MODEL", "CLIP", "VAE"],
               "3": ["CONDITIONING"], "4": ["CONDITIONING"],
               "5": ["BBOX_DETECTOR", "SEGM_DETECTOR"],
               "6": ["IMAGE", "IMAGE", "IMAGE", "MASK", "DETAILER_PIPE", "IMAGE"],
               "7": []}
    result = {}
    for node_id, node in graph.items():
        required = {}
        for name, value in node["inputs"].items():
            if isinstance(value, list):
                required[name] = [outputs[value[0]][value[1]]]
            elif name in ("ckpt_name", "model_name", "image", "sampler_name", "scheduler",
                          "sam_detection_hint", "sam_mask_hint_use_negative"):
                required[name] = [[value]]
            else:
                kind = {bool: "BOOLEAN", int: "INT", float: "FLOAT", str: "STRING"}[type(value)]
                required[name] = [kind]
        result[node["class_type"]] = {
            "input": {"required": required}, "output": outputs[node_id],
            "output_node": node_id == "7"}
    result["FaceDetailer"]["input"]["required"]["denoise"] = ["FLOAT", {"min": 0.0001, "max": 1.0}]
    result["FaceDetailer"]["input"]["required"]["steps"] = ["INT", {"min": 1, "max": 10000}]
    return result


class FakeServer:
    def __init__(self):
        self.timeout = 3.0
        self.graph = backend.load_workflow(settings())
        self.calls = []
        self.prompts = []
        self.receipts = []
        self.output = png()
        self.failure = None
        self.cancel_on_history = None
        self.free_failure = False

    def queue(self, deadline):
        self.calls.append(("GET", "/queue", None))
        return [], []

    def api(self, method, route, deadline, payload=None):
        self.calls.append((method, route, payload))
        if route == "/object_info":
            return object_info(self.graph)
        if route == "/prompt":
            self.prompts.append(copy.deepcopy(payload))
            return {"prompt_id": payload["prompt_id"], "node_errors": {}}
        if route.startswith("/history/"):
            if self.cancel_on_history is not None:
                self.cancel_on_history.set()
                return {}
            prompt = self.prompts[-1]
            folder = prompt["prompt"]["7"]["inputs"]["filename_prefix"].rsplit("/", 1)[0]
            status = {"completed": True, "status_str": "success", "messages": []}
            if self.failure:
                status = {"completed": False, "status_str": "error",
                          "messages": [["execution_error", {"exception_message": self.failure}]]}
            return {prompt["prompt_id"]: {"status": status, "outputs": {
                "6": {"images": [{"filename": "crop.png"}]},
                "7": {"images": [{"filename": "frame_00001_.png", "subfolder": folder, "type": "output"}]}}}}
        if route == "/free":
            if self.free_failure:
                raise backend.ComfyUIError("HTTP 500")
            return {}
        if route == "/system_stats":
            return {"devices": [{"type": "cuda", "torch_vram_total": 0}]}
        return {}

    def request(self, method, route, deadline, **kwargs):
        self.calls.append((method, route, kwargs))
        if route == "/upload/image":
            body = kwargs["body"]
            folder = body.split(b'name="subfolder"\r\n\r\n')[1].split(b"\r\n")[0].decode()
            self.receipts.append(folder)
            return json.dumps({"name": "frame.png", "subfolder": folder, "type": "input"}).encode()
        return self.output

    stop_owned = backend.ComfyUIClient.stop_owned
    release = backend.ComfyUIClient.release


class WorkflowTests(unittest.TestCase):
    def test_fixed_topology_and_defaults(self):
        raw = json.loads(backend.WORKFLOW.read_text())
        graph = backend.load_workflow(settings())
        self.assertEqual(set(raw), set("1234567"))
        self.assertEqual(raw["5"]["inputs"]["model_name"], "bbox/face_yolov8m.pt")
        self.assertEqual(raw["2"]["inputs"]["ckpt_name"], settings().comfy_checkpoint)
        self.assertEqual(graph["7"]["inputs"]["images"], ["6", 0])
        self.assertEqual(graph["6"]["inputs"]["seed"], 42)
        self.assertNotIn("sam_model_opt", graph["6"]["inputs"])
        backend.validate_workflow(graph, object_info(graph))

    def test_input_schema_mismatches(self):
        graph = backend.load_workflow(settings())
        for mutation, message in (
            (lambda info: info.pop("FaceDetailer"), "lacks FaceDetailer"),
            (lambda info: info["FaceDetailer"]["input"]["required"].update({"new_required": ["INT"]}), "missing"),
            (lambda info: info["FaceDetailer"]["input"]["required"].pop("cycle"), "unsupported"),
            (lambda info: info["CheckpointLoaderSimple"]["input"]["required"].update({"ckpt_name": [["other"]]}), "models/checkpoints"),
            (lambda info: info["UltralyticsDetectorProvider"]["input"]["required"].update({"model_name": [["other"]]}), "models/ultralytics/bbox"),
            (lambda info: info["UltralyticsDetectorProvider"].update({"output": ["IMAGE"]}), "connection type"),
            (lambda info: info["FaceDetailer"].update({"output_node": True}), "only output"),
        ):
            with self.subTest(message=message):
                info = object_info(graph)
                mutation(info)
                with self.assertRaisesRegex(backend.ComfyUIError, message):
                    backend.validate_workflow(graph, info)

    def test_invalid_controlled_values(self):
        graph = backend.load_workflow(settings())
        info = object_info(graph)
        for name, value in (("steps", True), ("steps", 0), ("denoise", float("nan")),
                            ("noise_mask", "true"), ("wildcard", 3)):
            bad = copy.deepcopy(graph)
            bad["6"]["inputs"][name] = value
            with self.subTest(name=name, value=value), self.assertRaises(backend.ComfyUIError):
                backend.validate_workflow(bad, info)

    def test_workflow_cannot_add_nodes_or_change_links(self):
        graph = json.loads(backend.WORKFLOW.read_text())
        for mutation in (
            lambda g: g.update({"8": g["7"]}),
            lambda g: g["7"]["inputs"].update({"images": ["6", 1]}),
            lambda g: g["6"]["inputs"].update({"wildcard": "__random__"}),
            lambda g: g["6"]["inputs"].update({"sam_model_opt": ["2", 0]}),
        ):
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "graph.json"
                bad = copy.deepcopy(graph)
                mutation(bad)
                path.write_text(json.dumps(bad))
                with patch.object(backend, "WORKFLOW", path), self.assertRaises(backend.ComfyUIError):
                    backend.load_workflow(settings())


class TransportTests(unittest.TestCase):
    def test_only_loopback_origins(self):
        for url in ("http://example.com:8188", "http://192.168.1.1", "http://127.0.0.1/path",
                    "http://user@localhost", "http://localhost?x=1", "ftp://localhost",
                    "http://localhost:99999", "http://127.0.0.1/#x"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                backend.ComfyUIClient(url)
        for url in ("http://localhost:8188", "http://127.0.0.2:8188", "http://[::1]:8188"):
            client = backend.ComfyUIClient(url)
            self.assertIn(client.host, ("127.0.0.1", "127.0.0.2", "::1"))
        for timeout in (0, -1, float("nan"), float("inf"), 86401):
            with self.assertRaises(ValueError):
                backend.ComfyUIClient(timeout=timeout)

    def transport(self, status=200, data=b"{}", length=None):
        response = Mock()
        response.status = status
        response.getheader.return_value = length
        response.read1.side_effect = [data, b""]
        connection = Mock()
        connection.getresponse.return_value = response
        return connection

    def test_http_status_redirect_limit_and_network_errors(self):
        for status, data, length, limit in (
            (302, b"redirect", None, 100), (400, b'{"node_errors":"bad graph"}', None, 100),
            (500, b"error", None, 100), (200, b"012345", None, 5),
            (200, b"{}", "9999", 5), (200, b"{}", "invalid", 5),
        ):
            connection = self.transport(status, data, length)
            with self.subTest(status=status, length=length), patch.object(
                    backend.http.client, "HTTPConnection", return_value=connection), self.assertRaises(backend.ComfyUIError):
                backend.ComfyUIClient().request("GET", "/queue", time.monotonic() + 1, limit=limit)
            connection.close.assert_called_once()
            self.assertEqual(connection.request.call_count, 1)  # No redirect follow.
        for error in (TimeoutError("timeout"), ConnectionRefusedError(), http.client.BadStatusLine("bad")):
            connection = self.transport()
            connection.request.side_effect = error
            with patch.object(backend.http.client, "HTTPConnection", return_value=connection), self.assertRaises(backend.ComfyUIError):
                backend.ComfyUIClient().api("GET", "/queue", time.monotonic() + 1)
            connection.close.assert_called_once()

    def test_json_and_deadlines(self):
        client = backend.ComfyUIClient()
        for data in (b"", b"invalid", b"[]", b"null"):
            with patch.object(client, "request", return_value=data), self.assertRaises(backend.ComfyUIError):
                client.api("GET", "/queue", time.monotonic() + 1)
        with patch.object(client, "request", return_value=b""):
            self.assertEqual(client.api("POST", "/free", time.monotonic() + 1, {}), {})
        with self.assertRaisesRegex(backend.ComfyUIError, "deadline"):
            client.request("GET", "/queue", time.monotonic() - 1)

    def test_real_local_http_exchange(self):
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        # One bounded request, not a daemon or inference service.
        server.timeout = 2
        worker = threading.Thread(target=server.handle_request)
        worker.start()
        try:
            client = backend.ComfyUIClient(f"http://127.0.0.1:{server.server_port}")
            self.assertEqual(client.api("GET", "/queue", time.monotonic() + 2), {})
        finally:
            worker.join(timeout=3)
            server.server_close()
        self.assertFalse(worker.is_alive())


class OwnershipTests(unittest.TestCase):
    def item(self, prompt_id, client="mine"):
        return [0, prompt_id, {}, {"client_id": client}, ["7"]]

    def test_pending_delete_only_and_running_targeted_interrupt(self):
        for running in (False, True):
            client = backend.ComfyUIClient()
            mine, other = self.item("owned"), self.item("other", "other-client")
            first = ([mine, other], []) if running else ([other], [mine])
            with patch.object(client, "queue", side_effect=[first, first, ([other], [])]), patch.object(client, "api", return_value={}) as api:
                client.stop_owned("owned", "mine", time.monotonic() + 2)
            calls = [(c.args[1], c.args[3]) for c in api.call_args_list]
            self.assertIn(("/queue", {"delete": ["owned"]}), calls)
            self.assertEqual([body for route, body in calls if route == "/interrupt"],
                             [{"prompt_id": "owned"}] if running else [])
            self.assertTrue(all("clear" not in body for _, body in calls))

    def test_pending_to_running_race_and_ambiguous_submit(self):
        client = backend.ComfyUIClient()
        mine = self.item("server-minted")
        with patch.object(client, "queue", side_effect=[([], [mine]), ([mine], []), ([], [])]), patch.object(client, "api", return_value={}) as api:
            client.stop_owned(None, "mine", time.monotonic() + 2)
        self.assertEqual(api.call_args_list[-1].args[3], {"prompt_id": "server-minted"})

    def test_queue_malformed_and_cleanup_timeout(self):
        client = backend.ComfyUIClient()
        for result in ({}, {"queue_running": [], "queue_pending": ["bad"]}):
            with patch.object(client, "api", return_value=result), self.assertRaises(backend.ComfyUIError):
                client.queue(time.monotonic() + 1)
        with patch.object(client, "queue", return_value=([self.item("owned")], [])), patch.object(client, "api", return_value={}), self.assertRaisesRegex(backend.ComfyUIError, "deadline"):
            client.stop_owned("owned", "mine", time.monotonic() + 0.01)

    def test_release_requires_idle_success_and_memory_evidence(self):
        client = backend.ComfyUIClient()
        with patch.object(client, "queue", return_value=([self.item("other", "other")], [])), patch.object(client, "api") as api, self.assertRaisesRegex(backend.ComfyUIError, "not idle"):
            client.release(time.monotonic() + 1)
        api.assert_not_called()
        for evidence in ({}, {"devices": []}, {"devices": [{"type": "cuda"}]},
                         {"devices": [{"type": "cuda", "torch_vram_total": -1}]}):
            with patch.object(client, "queue", return_value=([], [])), patch.object(client, "api", side_effect=[{}, evidence]), self.assertRaises(backend.ComfyUIError):
                client.release(time.monotonic() + 1)
        with patch.object(client, "queue", return_value=([], [])), patch.object(client, "api", side_effect=[{},
                {"devices": [{"type": "cuda", "torch_vram_total": 100}]},
                {"devices": [{"type": "cuda", "torch_vram_total": 0}]},
                {"devices": [{"type": "cuda", "torch_vram_total": 0}]}]) as api:
            client.release(time.monotonic() + 2)
        self.assertEqual(api.call_args_list[0].args[3], {"unload_models": True, "free_memory": True})
        self.assertEqual(api.call_count, 4)

    def test_async_allocator_residual_cap_and_stability(self):
        client = backend.ComfyUIClient()
        stats = {"devices": [{"type": "cuda", "index": 0,
                             "name": "cuda:0 RTX 4070 Ti : cudaMallocAsync",
                             "torch_vram_total": 64 * 1024 * 1024}]}
        with patch.object(client, "queue", return_value=([], [])), patch.object(
                client, "api", side_effect=[{}, stats, stats]) as api:
            client.release(time.monotonic() + 2)
        self.assertEqual(api.call_count, 3)
        for name, reserved in (("cudaMallocAsync", 256 * 1024 * 1024),
                               ("native", 64 * 1024 * 1024)):
            bad = {"devices": [{"type": "cuda", "name": name, "torch_vram_total": reserved}]}
            with patch.object(client, "queue", return_value=([], [])), patch.object(
                    client, "api", side_effect=lambda method, route, deadline, payload=None:
                    {} if route == "/free" else bad), self.assertRaises(backend.ComfyUIError):
                client.release(time.monotonic() + 0.01)

    def test_release_wait_timeout_and_concurrent_submit(self):
        client = backend.ComfyUIClient()
        with patch.object(client, "queue", return_value=([], [])), patch.object(client, "api", side_effect=lambda method, route, deadline, payload=None:
                {} if route == "/free" else {"devices": [{"type": "cuda", "torch_vram_total": 100}]}), self.assertRaisesRegex(backend.ComfyUIError, "deadline"):
            client.release(time.monotonic() + 0.01)
        with patch.object(client, "queue", side_effect=[([], []), ([self.item("other", "other")], [])]), patch.object(client, "api", return_value={}), self.assertRaisesRegex(backend.ComfyUIError, "appeared"):
            client.release(time.monotonic() + 1)

    def test_cancellation_cleanup_waits_for_owned_departure(self):
        client = backend.ComfyUIClient()
        mine = self.item("owned")
        snapshots = [([mine], []), ([mine], []), ([mine], []), ([mine], []), ([], [])]
        with patch.object(client, "queue", side_effect=snapshots) as queue, patch.object(client, "api", return_value={}) as api:
            client.stop_owned("owned", "mine", time.monotonic() + 2)
        self.assertEqual(queue.call_count, 5)
        interrupts = [call for call in api.call_args_list if call.args[1] == "/interrupt"]
        self.assertEqual(len(interrupts), 1)
        self.assertEqual(interrupts[0].args[3], {"prompt_id": "owned"})


class RestoreTests(unittest.TestCase):
    def run_restore(self, server, cancel=None, count=1):
        cancel = cancel or threading.Event()
        with tempfile.TemporaryDirectory() as temp:
            source, dest = Path(temp) / "source", Path(temp) / "dest"
            source.mkdir()
            for index in range(count):
                (source / f"{index:08d}.png").write_bytes(png())
            with patch.object(backend, "ComfyUIClient", return_value=server):
                backend.restore(source, dest, count, settings(), cancel, Mock())
            return [file.read_bytes() for file in sorted(dest.glob("*.png"))]

    def test_two_frames_fixed_seed_unique_prompts_multipart_and_release(self):
        server = FakeServer()
        self.assertEqual(self.run_restore(server, count=2), [png(), png()])
        prompts = server.prompts
        self.assertEqual([p["prompt"]["6"]["inputs"]["seed"] for p in prompts], [42, 42])
        self.assertNotEqual(prompts[0]["prompt_id"], prompts[1]["prompt_id"])
        self.assertEqual(prompts[0]["client_id"], prompts[1]["client_id"])
        self.assertEqual(server.receipts[0], server.receipts[1])
        uploads = [body for _, route, body in server.calls if route == "/upload/image"]
        self.assertIn(b"Content-Type: image/png", uploads[0]["body"])
        self.assertIn(b'name="overwrite"\r\n\r\nfalse', uploads[0]["body"])
        self.assertTrue(all(p["prompt"]["7"]["inputs"]["images"] == ["6", 0] for p in prompts))
        self.assertTrue(any(route == "/free" for _, route, _ in server.calls))
        self.assertEqual(server.calls[-1][1], "/system_stats")
        other = FakeServer()
        self.run_restore(other)
        self.assertNotEqual(server.receipts[0], other.receipts[0])
        self.assertNotEqual(server.prompts[0]["client_id"], other.prompts[0]["client_id"])

    def test_execution_error_and_wrong_png_reject_and_release(self):
        for mode in ("error", "dimensions", "jpeg", "corrupt"):
            server = FakeServer()
            if mode == "error":
                server.failure = "CUDA out of memory"
            else:
                server.output = {"dimensions": png((33, 24)), "jpeg": png(format="JPEG"), "corrupt": b"not png"}[mode]
            with self.subTest(mode=mode), self.assertRaises(backend.ComfyUIError):
                self.run_restore(server)
            self.assertTrue(any(route == "/free" for _, route, _ in server.calls))

    def test_cancel_preflight_and_during_execution(self):
        cancel = threading.Event()
        cancel.set()
        server = FakeServer()
        with self.assertRaises(CancelledError):
            self.run_restore(server, cancel)
        self.assertEqual(server.calls, [])
        cancel.clear()
        server.cancel_on_history = cancel
        with self.assertRaises(CancelledError):
            self.run_restore(server, cancel)
        self.assertTrue(any(route == "/free" for _, route, _ in server.calls))
        self.assertFalse(any(route.startswith("/view") for _, route, _ in server.calls))

    def test_free_failure_blocks_handoff(self):
        server = FakeServer()
        server.free_failure = True
        with self.assertRaisesRegex(backend.ComfyUIError, "do not start RIFE"):
            self.run_restore(server)

    def test_initial_error_remains_visible_when_cleanup_fails(self):
        server = FakeServer()
        server.failure = "original inference failure"
        server.free_failure = True
        with self.assertRaisesRegex(backend.ComfyUIError, "Initial failure:.*original inference failure"):
            self.run_restore(server)

    def test_windows_subfolders_normalized_without_weakening_safety(self):
        self.assertEqual(backend._safe_path("video_enhancer\\owned"), "video_enhancer/owned")
        for value in ("..\\owned", "\\absolute", "\\\\server\\share", "C:\\owned",
                      "safe\\..\\owned", "safe/../owned"):
            with self.subTest(value=value), self.assertRaises(backend.ComfyUIError):
                backend._safe_path(value)
        server = FakeServer()
        original_api = server.api

        def api(method, route, deadline, payload=None):
            result = original_api(method, route, deadline, payload)
            if route.startswith("/history/"):
                for entry in result.values():
                    entry["outputs"]["7"]["images"][0]["subfolder"] = server.receipts[0].replace("/", "\\")
            return result

        server.api = api
        self.assertEqual(self.run_restore(server), [png()])

    def test_history_deadline_and_saveimage_only(self):
        client = backend.ComfyUIClient()
        with patch.object(client, "api", return_value={}), self.assertRaisesRegex(backend.ComfyUIError, "deadline"):
            backend.wait_output(client, "owned", threading.Event(), time.monotonic() + 0.01)
        entry = {"owned": {"status": {"completed": True, "status_str": "success"},
                           "outputs": {"6": {"images": [{"filename": "crop.png"}]}}}}
        with patch.object(client, "api", return_value=entry), self.assertRaisesRegex(backend.ComfyUIError, "node 7"):
            backend.wait_output(client, "owned", threading.Event(), time.monotonic() + 1)

    def test_unsafe_paths_and_upload_receipt(self):
        for value in ("../x.png", "C:\\x.png", "/x.png", "x [output]", "x\n.png", None):
            with self.subTest(value=value), self.assertRaises(backend.ComfyUIError):
                backend._safe_path(value, filename=True)
        client = backend.ComfyUIClient()
        for raw in (b"invalid", b"[]", b'{"name":"x.png","subfolder":"other","type":"input"}'):
            with patch.object(client, "request", return_value=raw), self.assertRaises(backend.ComfyUIError):
                backend.upload_frame(client, png(), "frame.png", "owned", time.monotonic() + 1)


if __name__ == "__main__":
    unittest.main()