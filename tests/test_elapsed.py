import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("gradio") and importlib.util.find_spec("dotenv"),
                     "Requires installed UI dependencies")
class ElapsedTests(unittest.TestCase):
    def test_active_and_finished_timer(self):
        from app import Job
        job = Job((Path("video.mp4"),), {})
        with patch("app.time.monotonic", return_value=130):
            self.assertEqual(job.elapsed_detail(), "Elapsed: 0 s")
            job.started_at = 100
            self.assertEqual(job.elapsed_detail(), "Elapsed: 30 s")
            job.finished_at = 120
            self.assertEqual(job.elapsed_detail(), "Elapsed: 20 s")

    def test_batch_video_timer(self):
        from app import Job
        job = Job((Path("one.mp4"), Path("two.mp4")), {})
        job.started_at, job.video_started_at, job.video_index = 100, 120, 2
        with patch("app.time.monotonic", return_value=140):
            self.assertEqual(job.elapsed_detail(), "Elapsed: 40 s · Video 2/2: 20 s")
            job.video_finished_at = 135
            self.assertEqual(job.elapsed_detail(), "Elapsed: 40 s · Video 2/2: 15 s")