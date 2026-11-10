from fractions import Fraction
from itertools import product
from pathlib import Path
import os
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from pipeline.common import CancelledError, run_process
from pipeline.config import Settings, directory
from pipeline.extractor import parse_probe, extract
from pipeline.interpolator import frame_position, output_count
from pipeline.encoder import encode
from pipeline.runner import run_video


def probe_data(**changes):
    stream = dict(codec_type="video", index=2, avg_frame_rate="30000/1001",
                  duration="2", width=1920, height=1080, pix_fmt="yuv420p")
    stream.update(changes)
    return {"streams": [stream]}


class ConfigTests(unittest.TestCase):
    def test_defaults(self):
        Settings().validate()

    def test_invalid_settings(self):
        for change in (dict(scale=3), dict(face_model="CodeFormer"), dict(target_fps=float("nan")),
                       dict(face_weight=2), dict(rife_scale=0.1), dict(tile=-1)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                Settings(**change).validate()

    def test_custom_directory(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, OUTPUT_DIR=temp):
            self.assertEqual(directory("OUTPUT_DIR", "outputs"), Path(temp))


class ProbeTests(unittest.TestCase):
    def test_fractional_fps(self):
        info = parse_probe(probe_data())
        self.assertEqual(info.fps, Fraction(30000, 1001))
        self.assertEqual(info.stream_index, 2)

    def test_rate_fallback(self):
        self.assertEqual(parse_probe(probe_data(avg_frame_rate="0/0", r_frame_rate="24/1")).fps,
                         Fraction(24))

    def test_reject_hdr(self):
        for changes in (dict(pix_fmt="yuv420p10le"), dict(color_transfer="smpte2084")):
            with self.assertRaises(ValueError):
                parse_probe(probe_data(**changes))

    def test_reject_invalid_input(self):
        for data in ({"streams": []}, probe_data(duration="0"), probe_data(avg_frame_rate="bad"),
                     probe_data(width=0)):
            with self.assertRaises(ValueError):
                parse_probe(data)

    def test_skip_cover_art(self):
        data = probe_data()
        data["streams"].insert(0, dict(codec_type="video", disposition={"attached_pic": 1}))
        self.assertEqual(parse_probe(data).stream_index, 2)

    def test_extract_stream_map(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "frames"
            def fake(args, *_):
                self.assertEqual(args[args.index("-map") + 1], "0:2")
                (folder / "00000000.png").touch()
            with patch("pipeline.extractor.run_process", side_effect=fake):
                self.assertEqual(extract(Path("input.mp4"), folder, parse_probe(probe_data()),
                                         threading.Event(), lambda *_: None), 1)


class TimingTests(unittest.TestCase):
    def test_30_to_60_duration(self):
        self.assertEqual(output_count(300, Fraction(30), Fraction(60)), 600)
        self.assertEqual(frame_position(1, 300, Fraction(30), Fraction(60)), (0, 0.5))
        self.assertEqual(frame_position(599, 300, Fraction(30), Fraction(60)), (299, 0))

    def test_fractional_and_downsampling(self):
        self.assertEqual(output_count(300, Fraction(30000, 1001), Fraction(60)), 601)
        self.assertEqual(frame_position(2, 300, Fraction(60), Fraction(30)), (4, 0))
        self.assertEqual(output_count(1, Fraction(60), Fraction(1)), 1)


class ProcessTests(unittest.TestCase):
    def test_process_success_and_error(self):
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / "process.log"
            run_process([sys.executable, "-c", "print('ok')"], threading.Event(), log)
            self.assertIn("ok", log.read_text())
            with self.assertRaisesRegex(RuntimeError, "failed"):
                run_process([sys.executable, "-c", "raise RuntimeError('oops')"], threading.Event(), log)

    def test_cancel_before_spawn(self):
        event = threading.Event()
        event.set()
        with patch("pipeline.common.subprocess.Popen") as spawn:
            with self.assertRaises(CancelledError):
                run_process(["unused"], event, Path("unused.log"))
            spawn.assert_not_called()

    def test_cancel_live_process(self):
        with tempfile.TemporaryDirectory() as temp:
            event = threading.Event()
            with self.assertRaises(CancelledError):
                run_process([sys.executable, "-c", "while True: pass"], event,
                            Path(temp) / "cancel.log", tick=event.set)

    def test_encoder_ignores_partial_progress(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            output = work / "result.mkv"
            def fake(args, cancel, log, tick):
                (work / "encode.progress").write_text("frame=\nframe=12\n")
                tick()
                self.assertEqual(args[args.index("-c:a") + 1], "copy")
                self.assertIn("1:a?", args)
                output.write_bytes(b"result")
            with patch("pipeline.encoder.run_process", side_effect=fake):
                encode(work, work / "input.mp4", output, Fraction(60), 60,
                       threading.Event(), lambda *_: None, work)


class RunnerTests(unittest.TestCase):
    def test_independent_stage_combinations(self):
        for up, face, rife in product((False, True), repeat=3):
            with self.subTest(up=up, face=face, rife=rife), tempfile.TemporaryDirectory() as temp:
                base = Path(temp)
                source = base / "input.mp4"
                source.touch()

                def fake_extract(source, dest, *_):
                    dest.mkdir()
                    return 2

                def fake_stage(frames, dest, *_):
                    dest.mkdir()
                    return 4

                def fake_encode(frames, original, output, *_):
                    output.write_bytes(b"video")

                fake_torch = unittest.mock.MagicMock()
                fake_torch.cuda.is_available.return_value = False
                with patch.dict(os.environ, TEMP_DIR=str(base / "temp"), OUTPUT_DIR=str(base / "out")), \
                     patch.dict(sys.modules, torch=fake_torch), \
                     patch("pipeline.runner.preflight"), \
                     patch("pipeline.runner.probe", return_value=parse_probe(probe_data(width=2, height=2))), \
                     patch("pipeline.runner.extract", side_effect=fake_extract), \
                     patch("pipeline.runner.upscale", side_effect=fake_stage) as upscale, \
                     patch("pipeline.runner.restore", side_effect=fake_stage) as restore, \
                     patch("pipeline.runner.interpolate", side_effect=fake_stage) as interpolate, \
                     patch("pipeline.runner.encode", side_effect=fake_encode) as encoder:
                    settings = Settings(scale=2 if up else 1, face_model="GFPGAN" if face else "Off",
                                        rife_version="4.26" if rife else "Off")
                    output = run_video(source, settings, threading.Event(), lambda *_: None)
                    self.assertTrue(output.is_file())
                    self.assertEqual(upscale.call_count, int(up))
                    self.assertEqual(restore.call_count, int(face))
                    self.assertEqual(interpolate.call_count, int(rife))
                    args = encoder.call_args.args
                    self.assertEqual(args[3], Fraction(60) if rife else Fraction(30000, 1001))
                    self.assertEqual(args[4], 4 if rife else 2)

    def test_cleanup_and_atomic_output(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as temp:
                base = Path(temp)
                source = base / "input.mp4"
                source.touch()
                def fake_extract(source, dest, *_):
                    dest.mkdir()
                    (dest / "00000000.png").touch()
                    return 1
                def fake_encode(frames, original, output, *args):
                    output.write_bytes(b"video")
                    if fail:
                        raise RuntimeError("encoding failure")
                fake_torch = unittest.mock.MagicMock()
                fake_torch.cuda.is_available.return_value = False
                with patch.dict(os.environ, TEMP_DIR=str(base / "temp"), OUTPUT_DIR=str(base / "out")), \
                     patch.dict(sys.modules, torch=fake_torch), \
                     patch("pipeline.runner.preflight"), \
                     patch("pipeline.runner.probe", return_value=parse_probe(probe_data(width=2, height=2))), \
                     patch("pipeline.runner.extract", side_effect=fake_extract), \
                     patch("pipeline.runner.encode", side_effect=fake_encode):
                    settings = Settings(scale=1, face_model="Off", rife_version="Off")
                    if fail:
                        with self.assertRaises(RuntimeError):
                            run_video(source, settings, threading.Event(), lambda *_: None)
                    else:
                        output = run_video(source, settings, threading.Event(), lambda *_: None)
                        self.assertEqual(output.read_bytes(), b"video")
                    self.assertEqual(list((base / "temp").iterdir()), [])
                    self.assertEqual(list((base / "out").glob("*.partial.mkv")), [])


if __name__ == "__main__":
    unittest.main()