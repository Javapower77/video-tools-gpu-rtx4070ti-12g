import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

from pipeline.config import Settings
from pipeline.resolution import target_dimensions
from pipeline.runner import run_video
from pipeline.extractor import parse_probe
from tests.test_pipeline import probe_data


class NewPipelineTests(unittest.TestCase):
    def test_seedvr_runner_dispatch_preserves_batch_mode(self):
        for batch in (False, True):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = root / "video.mp4"
                source.touch()
                def extract(source, dest, *_):
                    dest.mkdir()
                    return 2
                def seed(source, dest, *_args, **_kwargs):
                    dest.mkdir()
                def encode(frames, source, dest, *_):
                    dest.write_bytes(b"video")
                torch = MagicMock()
                torch.cuda.is_available.return_value = False
                with patch.dict(os.environ, TEMP_DIR=str(root / "temp"), OUTPUT_DIR=str(root / "out")), \
                     patch.dict("sys.modules", torch=torch), patch("pipeline.runner.preflight"), \
                     patch("pipeline.runner.probe", return_value=parse_probe(probe_data(width=2, height=2))), \
                     patch("pipeline.runner.extract", side_effect=extract), \
                     patch("pipeline.seedvr2_backend.upscale", side_effect=seed) as dispatch, \
                     patch("pipeline.runner.upscale") as legacy, \
                     patch("pipeline.runner.encode", side_effect=encode):
                    run_video(source, Settings(upscale_model="SeedVR2", output_resolution="FHD",
                                              face_model="Off", rife_version="Off"),
                              threading.Event(), lambda *_: None, batch_mode=batch)
                    self.assertEqual(dispatch.call_args.kwargs["batch_mode"], batch)
                    legacy.assert_not_called()

    def test_aspect_ratio_resolution(self):
        for width, height, resolution, expected in (
            (1920, 1080, "4K", (3840, 2160)), (640, 480, "FHD", (1440, 1080)),
            (1080, 1920, "4K", (1214, 2160)), (3840, 2160, "FHD", (1920, 1080)),
        ):
            self.assertEqual(target_dimensions(width, height, resolution), expected)

    def test_comfy_settings_validation(self):
        Settings(face_model="ComfyUI", output_resolution="4K").validate()
        for change in ({"comfy_denoise": float("nan")}, {"comfy_seed": -1},
                       {"comfy_steps": 2.5}, {"comfy_cfg": float("inf")},
                       {"output_resolution": "8K"}):
            with self.assertRaises(ValueError):
                Settings(face_model="ComfyUI", **change).validate()

    def test_face_rife_upscale_order_and_release_failure(self):
        for failure in (False, True):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = root / "video.mp4"
                source.touch()
                order = []

                def frames(source, dest, *args):
                    dest.mkdir()
                    return 2

                def faces(source, dest, *args):
                    order.append("face")
                    if failure:
                        raise RuntimeError("GPU handoff failed")
                    dest.mkdir()

                def rife(source, dest, *args):
                    order.append("rife")
                    dest.mkdir()
                    return 4

                def upscale(source, dest, count, *args):
                    order.append("upscale")
                    self.assertEqual(count, 4)
                    dest.mkdir()

                def encode(frames, source, dest, *args):
                    order.append("encode")
                    dest.write_bytes(b"video")

                torch = MagicMock()
                torch.cuda.is_available.return_value = False
                with patch.dict(os.environ, TEMP_DIR=str(root / "temp"), OUTPUT_DIR=str(root / "out")), \
                     patch.dict("sys.modules", torch=torch), patch("pipeline.runner.preflight"), \
                     patch("pipeline.runner.probe", return_value=parse_probe(probe_data(width=2, height=2))), \
                     patch("pipeline.runner.extract", side_effect=frames), \
                     patch("pipeline.comfyui_backend.restore", side_effect=faces), \
                     patch("pipeline.runner.interpolate", side_effect=rife), \
                     patch("pipeline.runner.upscale", side_effect=upscale), \
                     patch("pipeline.runner.encode", side_effect=encode):
                    settings = Settings(face_model="ComfyUI", upscale_model="4x-UltraSharp", output_resolution="FHD")
                    if failure:
                        with self.assertRaisesRegex(RuntimeError, "handoff"):
                            run_video(source, settings, threading.Event(), lambda *_: None)
                        self.assertEqual(order, ["face"])
                    else:
                        run_video(source, settings, threading.Event(), lambda *_: None)
                        self.assertEqual(order, ["face", "rife", "upscale", "encode"])