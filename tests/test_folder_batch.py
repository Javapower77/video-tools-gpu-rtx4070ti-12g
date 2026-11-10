import importlib.util
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

from pipeline.config import Settings
from pipeline.file_dates import windows_file_times
from pipeline.folder_batch import discover_videos
from pipeline.runner import run_video
from tests.test_pipeline import probe_data
from pipeline.extractor import parse_probe


class FolderDiscoveryTests(unittest.TestCase):
    def test_recursive_overlapping_folders_and_exclusions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            child = root / "child"
            child.mkdir()
            for path in (root / "A.MP4", child / "B.mov", child / "notes.txt",
                         child / "B_a1b2c3d4.mkv", child / ".job.partial.mkv"):
                path.touch()
            found = discover_videos(f'"{child}"\n{root}\n{child}')
            self.assertEqual(set(found), {root / "A.MP4", child / "B.mov"})
            self.assertEqual(len(found), 2)

    def test_empty_missing_relative_and_file_roots(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            file = root / "video.mp4"
            file.touch()
            for text in ("", "relative", str(root / "missing"), str(file)):
                with self.subTest(text=text), self.assertRaises((ValueError, OSError)):
                    discover_videos(text)
            file.unlink()
            with self.assertRaisesRegex(ValueError, "No supported"):
                discover_videos(str(root))


class FolderOutputTests(unittest.TestCase):
    def run_mocked(self, base, times_fail=False):
        source = base / "original.name.mp4"
        source.touch()
        times = (132000000000000000, 132000000000000001, 132000000000000002)

        def fake_dates(path, captured=None):
            if times_fail and captured is not None:
                raise OSError("timestamp failure")
            return times

        def fake_extract(source, dest, *_):
            dest.mkdir()
            return 1

        def fake_encode(frames, original, output, *_):
            output.write_bytes(b"video")

        torch = MagicMock()
        torch.cuda.is_available.return_value = False
        with patch.dict(os.environ, TEMP_DIR=str(base / "temp"), OUTPUT_DIR=str(base / "out")), \
             patch.dict("sys.modules", torch=torch), \
             patch("pipeline.runner.preflight"), \
             patch("pipeline.runner.windows_file_times", side_effect=fake_dates) as dates, \
             patch("pipeline.runner.probe", return_value=parse_probe(probe_data(width=2, height=2))), \
             patch("pipeline.runner.extract", side_effect=fake_extract), \
             patch("pipeline.runner.encode", side_effect=fake_encode):
            output = run_video(source, Settings(scale=1, face_model="Off", rife_version="Off"),
                               threading.Event(), lambda *_: None, save_beside_source=True)
        self.assertEqual(output.parent, source.parent)
        self.assertRegex(output.name, r"^original\.name_[0-9a-f]{8}\.mkv$")
        self.assertNotEqual(output, source)
        self.assertEqual(dates.call_args_list[0].args, (source,))
        self.assertEqual(dates.call_args_list[1].args[1], times)
        self.assertEqual(len(dates.call_args_list), 3)
        self.assertFalse((base / "out").exists())
        self.assertEqual(list((base / "temp").iterdir()), [])

    def test_output_name_location_and_date_capture(self):
        with tempfile.TemporaryDirectory() as temp:
            self.run_mocked(Path(temp))

    def test_timestamp_failure_removes_partial_and_does_not_publish(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            with self.assertRaisesRegex(OSError, "timestamp failure"):
                self.run_mocked(base, times_fail=True)
            self.assertEqual(list(base.glob("*.mkv")), [])
            self.assertTrue((base / "original.name.mp4").exists())
            self.assertEqual(list((base / "temp").iterdir()), [])

    @unittest.skipUnless(os.name == "nt", "Windows FILETIME API")
    def test_real_windows_creation_and_modification_dates_survive_rename(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.mp4"
            partial = Path(temp) / ".output.partial.mkv"
            source.touch()
            partial.write_bytes(b"test")
            times = (132000000000000000, 132000000000000100, 132000000000000200)
            windows_file_times(source, times)
            captured = windows_file_times(source)
            windows_file_times(partial, captured)
            output = partial.rename(Path(temp) / "source_1234abcd.mkv")
            restored = windows_file_times(output)
            self.assertEqual(restored[0], captured[0])
            self.assertEqual(restored[2], captured[2])


@unittest.skipUnless(importlib.util.find_spec("gradio") and importlib.util.find_spec("dotenv"),
                     "Requires installed UI dependencies")
class FolderUITests(unittest.TestCase):
    def test_folder_admission_and_single_mode_isolation(self):
        from app import admit_job, apply_profile
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "video.mp4"
            source.touch()
            values = apply_profile("Normal")[:9]
            job = admit_job(None, "Batch", None, None, *values, False,
                            True, True, True, "Local folders", str(root))[0]
            self.assertTrue(job.folder_batch)
            self.assertEqual(job.sources, (source,))
            single = admit_job(None, "Single video", str(source), None, *values, False,
                               True, True, True, "Local folders", "missing")[0]
            self.assertFalse(single.folder_batch)

    def test_folder_worker_and_results_not_published_by_gradio(self):
        from app import Job, process_job, stream_job
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "video.mp4"
            source.touch()
            output = root / "video_1234abcd.mkv"
            output.write_bytes(b"test")
            from dataclasses import asdict
            job = Job((source,), asdict(Settings()), folder_batch=True)
            with patch("pipeline.runner.run_video", return_value=output) as runner:
                process_job(job)
            self.assertEqual(job.errors, 0)
            self.assertEqual(job.outputs, [output])
            self.assertTrue(runner.call_args.kwargs["save_beside_source"])
            with patch("app.process_job", side_effect=lambda job: job.done.set()):
                snapshots = list(stream_job(job))
            self.assertTrue(all(snapshot[5] == [] and snapshot[6] is None for snapshot in snapshots))

    def test_cancel_folder_picker_retains_selection(self):
        from app import choose_folders, source_controls
        with patch("app.select_local_folders", return_value=None):
            self.assertNotIn("value", choose_folders())
        with patch("app.select_local_folders", return_value="D:\\Videos"):
            self.assertEqual(choose_folders(), "D:\\Videos")
        self.assertEqual([value["visible"] for value in source_controls("Batch", "Local folders")],
                         [False, True, False, True])


if __name__ == "__main__":
    unittest.main()