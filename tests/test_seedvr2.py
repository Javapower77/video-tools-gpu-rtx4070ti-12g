"""Offline stdlib mocks: no server, GPU, FFmpeg execution, installs or downloads."""
from contextlib import ExitStack
from fractions import Fraction
import hashlib
import inspect
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from pipeline import seedvr2_backend as backend
from pipeline.common import CancelledError
from pipeline.comfyui_backend import ComfyUIError


def png(size=(16, 16), color="red"):
    stream = io.BytesIO()
    Image.new("RGB", size, color).save(stream, format="PNG")
    return stream.getvalue()


def settings(**kwargs):
    return SimpleNamespace(**({"output_resolution": "FHD", "seedvr_batch_size": 5,
                               "seedvr_noise": 0.03} | kwargs))


def schema(graph):
    """Legacy serialized shapes of the verified core/V3 schemas, offline."""
    outputs = {"1": ["VIDEO"], "2": ["IMAGE", "AUDIO", "FLOAT", "COMBO", "COMBO"],
               "3": ["SEEDVR2_DIT"], "4": ["SEEDVR2_VAE"], "5": ["IMAGE"], "7": []}
    result = {}
    for node_id, node in graph.items():
        inputs = {}
        for name, value in node["inputs"].items():
            if isinstance(value, list):
                kind = outputs[value[0]][value[1]]
            elif name == "file":
                kind = []  # LoadVideo's combo only enumerates root files.
            elif name in ("device", "offload_device", "model", "attention_mode", "tile_debug", "color_correction"):
                kind = [value]
            else:
                kind = {bool: "BOOLEAN", int: "INT", float: "FLOAT", str: "STRING"}[type(value)]
            inputs[name] = [kind]
        result[node["class_type"]] = {"input": {"required": inputs, "optional": {}},
                                      "output": outputs[node_id], "output_node": node_id == "7"}
    return result


class SeedVR2UnitTests(unittest.TestCase):
    def test_contract(self):
        signature = inspect.signature(backend.upscale)
        self.assertEqual(list(signature.parameters),
                         ["source", "dest", "count", "fps", "settings", "cancel", "report", "batch_mode"])
        self.assertEqual(signature.parameters["batch_mode"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIs(signature.parameters["batch_mode"].default, False)

    def test_exact_fps_and_single_bounds(self):
        backend._options(240, Fraction(30), settings(), False)
        backend._options(191, Fraction(24000, 1001), settings(), False)
        for count, fps, opts in [(241, Fraction(30), settings()),
                                 (192, Fraction(24000, 1001), settings()),
                                 (9, Fraction(30), settings(seedvr_max_frames=8)),
                                 (480, Fraction(60), settings(seedvr_max_frames=480)),
                                 (240, 30.0, settings()), (0, Fraction(30), settings()),
                                 (1, Fraction(0), settings()), (True, Fraction(30), settings()),
                                 (1, Fraction(30), settings(seedvr_max_frames=0))]:
            with self.subTest(count=count, fps=fps, opts=opts), self.assertRaises(ValueError):
                backend._options(count, fps, opts, False)
        backend._options(10000, Fraction(30), settings(), True)

    def test_invalid_settings(self):
        for kwargs in ({"output_resolution": "Scale"}, {"seedvr_batch_size": 13},
                       {"seedvr_batch_size": True}, {"seedvr_noise": float("nan")},
                       {"seedvr_noise": -1}, {"seedvr_noise": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                backend._options(1, Fraction(30), settings(**kwargs), False)

    def test_graph_and_presets(self):
        for resolution, target, tile, overlap, blocks in (
                ("FHD", (1920, 1080), 512, 64, 28), ("4K", (2160, 3840), 384, 48, 32)):
            graph = backend.load_workflow(target, 5, 0.03, resolution, "owned/input.mkv [input]", "owned")
            backend.validate_workflow(graph, schema(graph))
            self.assertEqual(graph["1"]["inputs"], {"file": "owned/input.mkv [input]"})
            self.assertEqual(graph["2"]["inputs"], {"video": ["1", 0]})
            self.assertEqual(graph["5"]["inputs"]["image"], ["2", 0])
            self.assertEqual(graph["7"]["inputs"]["images"], ["5", 0])
            self.assertNotIn("SaveVideo", [n["class_type"] for n in graph.values()])
            self.assertEqual(graph["3"]["inputs"]["blocks_to_swap"], blocks)
            self.assertFalse(graph["3"]["inputs"]["cache_model"])
            self.assertFalse(graph["4"]["inputs"]["cache_model"])
            self.assertEqual(graph["4"]["inputs"]["encode_tile_size"], tile)
            self.assertEqual(graph["4"]["inputs"]["decode_tile_overlap"], overlap)
            self.assertEqual(graph["5"]["inputs"]["resolution"], min(target))
            self.assertEqual(graph["5"]["inputs"]["max_resolution"], max(target))
            self.assertEqual(graph["5"]["inputs"]["prepend_frames"], 4)

    def test_schema_fails_closed(self):
        graph = backend.load_workflow((1920, 1080), 5, 0.03, "FHD", "owned/video.mkv", "owned")
        info = schema(graph)
        variants = [{}, json.loads(json.dumps(info)), json.loads(json.dumps(info)),
                    json.loads(json.dumps(info)), json.loads(json.dumps(info))]
        variants[1]["SeedVR2VideoUpscaler"]["input"]["required"]["new_required"] = ["INT"]
        variants[2]["GetVideoComponents"]["output"][0] = "AUDIO"
        variants[3]["SeedVR2LoadDiTModel"]["input"]["required"]["model"] = [["different.safetensors"]]
        variants[4]["SaveImage"]["output_node"] = False
        for info in variants:
            with self.subTest(info=info), self.assertRaises(ComfyUIError):
                backend.validate_workflow(graph, info)

    def test_workflow_connections_cannot_be_edited(self):
        graph = json.loads(backend.WORKFLOW.read_text(encoding="utf-8"))
        graph["5"]["inputs"]["image"] = ["1", 0]
        with patch.object(Path, "open", return_value=io.BytesIO(json.dumps(graph).encode())):
            with self.assertRaisesRegex(ComfyUIError, "connection"):
                backend.load_workflow((1920, 1080), 5, 0.03, "FHD", "input.mkv", "owned")

    def test_wait_batch_history_and_count(self):
        client = MagicMock()
        images = [{"filename": f"frame_{i}.png"} for i in range(3)]
        client.api.side_effect = [{}, {"pid": {"status": {"completed": True, "status_str": "success"},
                                             "outputs": {"7": {"images": images}}}}]
        cancel = MagicMock()
        cancel.is_set.return_value = False
        cancel.wait.return_value = False
        self.assertEqual(backend.wait_outputs(client, "pid", 3, cancel, backend.time.monotonic() + 10), images)
        for output in (images[:2], images + [images[0]], [None] * 3):
            client.api.side_effect = None
            client.api.return_value = {"pid": {"status": {"completed": True, "status_str": "success"},
                                              "outputs": {"7": {"images": output}}}}
            with self.assertRaises(ComfyUIError):
                backend.wait_outputs(client, "pid", 3, cancel, backend.time.monotonic() + 10)
        client.api.return_value = {"pid": {"status": {"status_str": "error", "messages": []}}}
        with self.assertRaises(ComfyUIError):
            backend.wait_outputs(client, "pid", 3, cancel, backend.time.monotonic() + 10)

    def test_wait_cancellation(self):
        cancel = threading.Event()
        cancel.set()
        client = MagicMock()
        with self.assertRaises(CancelledError):
            backend.wait_outputs(client, "pid", 3, cancel, backend.time.monotonic() + 10)
        client.api.assert_not_called()

    def test_model_integrity_read_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models = root / "models/SEEDVR2"
            models.mkdir(parents=True)
            cancel = threading.Event()
            with patch.object(backend, "COMFY_ROOT", root):
                with self.assertRaisesRegex(ComfyUIError, "existing local model"):
                    backend.check_models(cancel)
                for name in backend.MODEL_HASHES:
                    (models / name).write_bytes(b"not a real weight")
                with self.assertRaisesRegex(ComfyUIError, "integrity"):
                    backend.check_models(cancel)
                self.assertEqual((models / backend.DIT_MODEL).read_bytes(), b"not a real weight")
                hashes = {name: hashlib.sha256(b"not a real weight").hexdigest() for name in backend.MODEL_HASHES}
                with patch.object(backend, "MODEL_HASHES", hashes):
                    self.assertEqual(backend.check_models(cancel), models)
                cancel.set()
                with self.assertRaises(CancelledError):
                    backend.check_models(cancel)

    def test_output_png_rounding_and_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "output.png"
            self.assertEqual(backend._write_output(png((18, 16)), path, (16, 16), None), (18, 16))
            with Image.open(path) as image:
                self.assertEqual(image.size, (16, 16))
            for data, previous in [(b"bad", None), (png((128, 128)), None), (png((18, 16)), (16, 16))]:
                with self.assertRaises(ComfyUIError):
                    backend._write_output(data, path, (16, 16), previous)


class SeedVR2StageTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.comfy = self.root / "vendor/ComfyUI"
        self.source = self.root / "source"
        self.source.mkdir()
        for index in range(3):
            (self.source / f"{index:08d}.png").write_bytes(png())
        self.dest = self.root / "dest"
        for relative in (".venv/Scripts/python.exe", "custom_nodes/seedvr2_videoupscaler/inference_cli.py"):
            path = self.comfy / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        self.stack.enter_context(patch.object(backend, "ROOT", self.root))
        self.stack.enter_context(patch.object(backend, "COMFY_ROOT", self.comfy))
        self.models = self.comfy / "models/SEEDVR2"
        self.stack.enter_context(patch.object(backend, "check_models", return_value=self.models))
        self.stack.enter_context(patch.object(backend, "target_dimensions", return_value=(16, 16)))
        self.client = MagicMock(timeout=10)
        self.client.queue.return_value = ([], [])
        self.client.request.return_value = png((18, 16))
        self.stack.enter_context(patch.object(backend, "ComfyUIClient", return_value=self.client))
        self.graph = backend.load_workflow((16, 16), 5, 0.03, "FHD", "input.mkv", "owned")
        self.client.api.side_effect = self.api
        self.process = self.stack.enter_context(patch.object(backend, "run_process", side_effect=self.run_process))
        self.cancel = threading.Event()
        self.report = MagicMock()
        self.receipts = None
        self.cli_count = 3
        self.commands = []

    def api(self, method, route, deadline, payload=None):
        if route == "/object_info":
            return schema(self.graph)
        if route == "/prompt":
            self.submission = payload
            return {"prompt_id": "pid"}
        raise AssertionError(f"Unexpected API route: {route}")

    def run_process(self, args, cancel, log, tick=None):
        self.commands.append(args)
        if "--output_format" in args:
            video = Path(args[args.index("--output") - 1])
            folder = Path(args[args.index("--output") + 1]) / video.stem
            folder.mkdir(parents=True)
            for index in range(self.cli_count):
                (folder / f"{video.stem}_{index:06d}.png").write_bytes(png((18, 16)))
            if tick:
                tick()
        else:
            Path(args[-1]).write_bytes(b"mock FFV1")

    def wait_outputs(self, client, prompt_id, count, cancel, deadline):
        folder = self.submission["prompt"]["7"]["inputs"]["filename_prefix"].rsplit("/", 1)[0]
        out = self.comfy / "output" / folder
        out.mkdir(parents=True)
        (out / "owned.png").write_bytes(png())
        return self.receipts if self.receipts is not None else [
            {"filename": f"frames_{i:05d}_.png", "subfolder": folder.replace("/", "\\"), "type": "output"}
            for i in range(count)]

    def execute(self, batch_mode=False):
        with patch.object(backend, "wait_outputs", side_effect=self.wait_outputs):
            return backend.upscale(self.source, self.dest, 3, Fraction(30000, 1001),
                                   settings(), self.cancel, self.report, batch_mode=batch_mode)

    def assert_outputs(self):
        self.assertEqual([p.name for p in sorted(self.dest.iterdir())], [f"{i:08d}.png" for i in range(3)])
        for path in self.dest.iterdir():
            with Image.open(path) as image:
                self.assertEqual(image.size, (16, 16))
        self.assertEqual(list((self.root / "temp").iterdir()), [])
        self.assertEqual(len(list(self.source.glob("*.png"))), 3)

    def test_single_contract_lossless_video_and_cleanup(self):
        self.assertIsNone(self.execute())
        self.assert_outputs()
        prepare = self.commands[0]
        self.assertEqual(prepare[prepare.index("-framerate") + 1], "30000/1001")
        self.assertEqual(prepare[prepare.index("-c:v") + 1], "ffv1")
        self.assertEqual(prepare[prepare.index("-frames:v") + 1], "3")
        self.assertIn("-an", prepare)
        self.assertNotIn("/upload/image", [c.args[1] for c in self.client.api.call_args_list])
        self.assertEqual(self.client.request.call_count, 3)
        self.client.stop_owned.assert_called_once()
        self.client.release.assert_called_once()
        self.assertEqual(list((self.comfy / "input/seedvr2").iterdir()), [])
        self.assertEqual(list((self.comfy / "output/seedvr2").iterdir()), [])

    def test_cli_flags_output_and_idle_handoff(self):
        self.assertIsNone(self.execute(True))
        self.assert_outputs()
        self.client.release.assert_called_once()
        self.client.stop_owned.assert_not_called()
        args = self.commands[1]
        for flag, value in {"--chunk_size": "45", "--prepend_frames": "0", "--temporal_overlap": "3",
                            "--output_format": "png", "--attention_mode": "sdpa", "--blocks_to_swap": "28",
                            "--vae_encode_tile_size": "512", "--vae_decode_tile_overlap": "64",
                            "--dit_model": backend.DIT_MODEL, "--model_dir": str(self.models.resolve())}.items():
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertEqual(Path(args[0]), self.comfy / ".venv/Scripts/python.exe")
        self.assertEqual(args[1:4], ["-X", "utf8", "-u"])
        self.assertNotIn("--vae_model", args)
        self.assertNotIn("--10bit", args)
        self.assertNotIn("--cache_dit", args)
        self.assertNotIn("--cache_vae", args)

    def test_4k_cli_preset(self):
        args = backend.cli_command(self.source / "input.mkv", self.root / "out", self.models,
                                   (3840, 2160), 5, 0.03, "4K")
        for flag, value in {"--blocks_to_swap": "32", "--vae_encode_tile_size": "384",
                            "--vae_decode_tile_overlap": "48", "--resolution": "2160",
                            "--max_resolution": "3840"}.items():
            self.assertEqual(args[args.index(flag) + 1], value)

    def test_cli_count_failure_never_trim_or_publish(self):
        for count in (2, 4):
            self.cli_count = count
            with self.assertRaisesRegex(ComfyUIError, "exactly 3"):
                self.execute(True)
            self.assertFalse(self.dest.exists())
            self.assertEqual(list((self.root / "temp").iterdir()), [])

    def test_cli_cancellation_shared_process_contract(self):
        original = self.run_process
        def cancelled(args, cancel, log, tick=None):
            if "--output_format" in args:
                self.assertIs(cancel, self.cancel)
                raise CancelledError("cancelled CLI")
            original(args, cancel, log, tick)
        self.process.side_effect = cancelled
        with self.assertRaises(CancelledError):
            self.execute(True)
        self.assertFalse(self.dest.exists())
        self.assertEqual(list((self.root / "temp").iterdir()), [])

    def test_single_cancel_stops_only_owned_and_releases(self):
        with patch.object(backend, "wait_outputs", side_effect=CancelledError("cancelled")):
            with self.assertRaises(CancelledError):
                backend.upscale(self.source, self.dest, 3, Fraction(30), settings(), self.cancel, self.report)
        self.client.stop_owned.assert_called_once()
        self.assertEqual(self.client.stop_owned.call_args.args[0], "pid")
        self.client.release.assert_called_once()
        self.assertFalse(self.dest.exists())

    def test_ambiguous_submission_recovered_by_client_id(self):
        def rejected(method, route, deadline, payload=None):
            if route == "/object_info":
                return schema(self.graph)
            raise ComfyUIError("connection lost after submission")
        self.client.api.side_effect = rejected
        with self.assertRaisesRegex(ComfyUIError, "connection lost"):
            self.execute()
        self.client.stop_owned.assert_called_once()
        self.assertTrue(self.client.stop_owned.call_args.args[1])
        self.client.release.assert_called_once()
        self.assertFalse(self.dest.exists())

    def test_unconfirmed_handoff_fatal_and_owned_files_retained(self):
        self.client.stop_owned.side_effect = ComfyUIError("offline")
        with self.assertRaisesRegex(ComfyUIError, "handoff unconfirmed"):
            self.execute()
        self.assertFalse(self.dest.exists())
        self.assertEqual(len(list((self.comfy / "input/seedvr2").iterdir())), 1)
        self.assertEqual(len(list((self.comfy / "output/seedvr2").iterdir())), 1)

    def test_non_owned_or_duplicate_output_rejected(self):
        self.receipts = [{"filename": "frame.png", "subfolder": "../other", "type": "output"}] * 3
        with self.assertRaises(ComfyUIError):
            self.execute()
        self.assertFalse(self.dest.exists())
        self.client.stop_owned.assert_called_once()

    def test_busy_queue_and_precancel_do_not_start_process(self):
        self.client.queue.return_value = (["other job"], [])
        with self.assertRaisesRegex(ComfyUIError, "idle"):
            self.execute(True)
        self.process.assert_not_called()
        self.client.release.assert_not_called()
        self.cancel.set()
        with self.assertRaises(CancelledError):
            self.execute()

    def test_queue_becomes_busy_before_execution(self):
        self.client.queue.side_effect = [([], []), (["other job"], [])]
        with self.assertRaisesRegex(ComfyUIError, "became busy"):
            self.execute(True)
        self.assertEqual(self.process.call_count, 1)  # FFmpeg only, never CLI.
        self.client.release.assert_not_called()

    def test_missing_source_existing_dest_and_bad_ffmpeg_output(self):
        self.dest.mkdir()
        with self.assertRaises(ValueError):
            self.execute()
        self.dest.rmdir()
        (self.source / "00000001.png").unlink()
        with self.assertRaises(ValueError):
            self.execute()
        (self.source / "00000001.png").write_bytes(png())
        self.process.side_effect = lambda *args: None
        with self.assertRaisesRegex(ComfyUIError, "missing/empty"):
            self.execute()
        self.assertFalse(self.dest.exists())


if __name__ == "__main__":
    unittest.main()