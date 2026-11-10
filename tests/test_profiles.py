import importlib.util
from itertools import product
from pathlib import Path
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace

from pipeline.config import Settings
from pipeline.profiles import PROFILES


class ProfileTests(unittest.TestCase):
    def test_normal_matches_current_defaults(self):
        self.assertEqual(PROFILES["Normal"].settings.output_resolution, "FHD")
        self.assertEqual(PROFILES["Normal"].settings.face_model, "ComfyUI")

    def test_boost_uses_supported_models(self):
        boost = PROFILES["Boost"].settings
        boost.validate()
        self.assertEqual(boost.upscale_model, "4x-UltraSharp")
        self.assertEqual(boost.scale, 4)
        self.assertEqual(boost.rife_version, "4.26")
        self.assertEqual(boost.face_model, "ComfyUI")
        self.assertEqual(boost.output_resolution, "4K")
        self.assertEqual(boost.face_weight, 0.5)
        self.assertTrue(boost.fp16)
        self.assertEqual(boost.tile, 256)
        self.assertEqual(boost.rife_scale, 1)
        self.assertIn("12 GB VRAM warning", PROFILES["Boost"].description)
        self.assertIn("Heavy", PROFILES["Boost"].description)

    def test_presets_cannot_be_mutated(self):
        with self.assertRaises(TypeError):
            PROFILES["Normal"] = PROFILES["Boost"]
        with self.assertRaises(FrozenInstanceError):
            PROFILES["Boost"].settings.scale = 2
        customized = replace(PROFILES["Boost"].settings, scale=2)
        customized.validate()
        self.assertEqual(PROFILES["Boost"].settings.scale, 4)


@unittest.skipUnless(importlib.util.find_spec("gradio") and importlib.util.find_spec("dotenv"),
                     "Requires installed UI dependencies")
class ProfileUITests(unittest.TestCase):
    def test_all_operation_combinations_for_single_and_batch(self):
        import gradio as gr
        from app import admit_job, apply_profile
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "video.mp4"
            source.touch()
            for mode in ("Single video", "Batch"):
                for enabled in product((False, True), repeat=3):
                    with self.subTest(mode=mode, enabled=enabled):
                        args = (None, mode, str(source), [str(source)],
                                *apply_profile("Boost")[:9], False, *enabled)
                        if not any(enabled):
                            with self.assertRaisesRegex(gr.Error, "Select at least one operation"):
                                admit_job(*args)
                            continue
                        job = admit_job(*args)[0]
                        settings = Settings(**job.options)
                        settings.validate()
                        self.assertEqual(settings.scale, 4 if enabled[0] else 1)
                        self.assertEqual(settings.face_model, "ComfyUI" if enabled[1] else "Off")
                        self.assertEqual(settings.rife_version, "4.26" if enabled[2] else "Off")
                        self.assertEqual(settings.target_fps, 60)
                        self.assertEqual(settings.face_weight, 0.5)

    def test_legacy_off_controls_cannot_submit_no_operations(self):
        import gradio as gr
        from app import admit_job, apply_profile
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "video.mp4"
            source.touch()
            values = list(apply_profile("Normal")[:9])
            values[1], values[2], values[4] = "1", "Off", "Off"
            with self.assertRaisesRegex(gr.Error, "Select at least one operation"):
                admit_job(None, "Single video", str(source), None, *values, False)

    def test_operation_checkbox_wiring_and_profile_preservation(self):
        from functools import partial
        from app import admit_job, apply_profile, build_app, operation_controls
        demo = build_app()
        try:
            callbacks = [fn for fn in demo.fns.values()
                         if isinstance(fn.fn, partial) and fn.fn.func is operation_controls]
            self.assertEqual(len(callbacks), 3)
            self.assertEqual([len(fn.outputs) for fn in callbacks], [6, 11, 3])
            for fn in callbacks:
                self.assertTrue(fn.inputs[0].value)
                self.assertFalse(fn.queue)
                updates = fn.fn(False)
                self.assertEqual(len(updates), len(fn.outputs))
                self.assertTrue(all(update["interactive"] is False for update in updates))
                self.assertTrue(all("value" not in update for update in updates))
                self.assertTrue(all(update["interactive"] for update in fn.fn(True)))
            profile = next(fn for fn in demo.fns.values() if fn.fn is apply_profile)
            toggles = [fn.inputs[0] for fn in callbacks]
            self.assertTrue(all(toggle not in profile.outputs for toggle in toggles))
            submission = next(fn for fn in demo.fns.values() if fn.fn is admit_job)
            self.assertEqual(submission.inputs[14:17], toggles)
        finally:
            demo.close()

    def test_callback_values_and_switch_back(self):
        from app import apply_profile
        for name in ("Normal", "Boost", "Normal"):
            values = apply_profile(name)
            settings = Settings(
                upscale_model=values[0], scale=int(values[1]), face_model=values[2],
                face_weight=values[3], rife_version=values[4], target_fps=values[5],
                fp16=values[6], tile=values[7], rife_scale=float(values[8]),
                output_resolution=values[10],
            )
            self.assertEqual(settings, PROFILES[name].settings)
            self.assertEqual(values[9], PROFILES[name].description)
        import gradio as gr
        with self.assertRaises(gr.Error):
            apply_profile("Unknown")

    def test_ui_wires_profile_to_enhancement_controls(self):
        from app import apply_profile, build_app
        demo = build_app()
        try:
            callbacks = [fn for fn in demo.fns.values() if fn.fn is apply_profile]
            self.assertEqual(len(callbacks), 1)
            callback = callbacks[0]
            self.assertEqual(callback.inputs[0].value, "Normal")
            self.assertEqual(len(callback.outputs), 11)
            self.assertEqual(callback.outputs[0].label, "Upscale model (last stage)")
            self.assertEqual(callback.outputs[8].label, "RIFE processing scale")
            self.assertFalse(callback.queue)
        finally:
            demo.close()

    def test_jobs_snapshot_editable_controls_for_single_and_batch(self):
        from app import admit_job, apply_profile
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "video.mp4"
            source.touch()
            for mode in ("Single video", "Batch"):
                values = list(apply_profile("Boost")[:9])
                values[1] = "2"  # User adjusts Boost for a 1080p input.
                job = admit_job(None, mode, str(source), [str(source)], *values, True)[0]
                expected = replace(PROFILES["Boost"].settings, scale=2, keep_temp=True, output_resolution="Scale")
                self.assertEqual(Settings(**job.options), expected)
                apply_profile("Normal")
                self.assertEqual(Settings(**job.options), expected)


if __name__ == "__main__":
    unittest.main()