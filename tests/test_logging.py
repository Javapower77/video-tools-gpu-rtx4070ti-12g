import logging
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from pipeline.common import run_process
from pipeline.logging_setup import configure_logging


class LoggingTests(unittest.TestCase):
    def setUp(self):
        self.root = logging.getLogger()
        self.previous_handlers = self.root.handlers[:]
        self.previous_level = self.root.level
        self.root.handlers = []

    def tearDown(self):
        logging.captureWarnings(False)
        for handler in self.root.handlers:
            handler.close()
        self.root.handlers = self.previous_handlers
        self.root.setLevel(self.previous_level)

    def test_rotation_and_idempotent_configuration(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
            "LOG_DIR": temp, "LOG_MAX_BYTES": "1024", "LOG_BACKUP_COUNT": "2", "LOG_LEVEL": "INFO"
        }):
            target = configure_logging()
            configure_logging()
            files = [handler for handler in self.root.handlers if isinstance(handler, logging.FileHandler)]
            self.assertEqual(len(files), 1)
            for i in range(30):
                logging.getLogger("test").info("Unicode ✓ %s %s", i, "x" * 200)
            self.assertTrue(target.is_file())
            self.assertTrue(Path(str(target) + ".1").is_file())
            self.assertLessEqual(len(list(Path(temp).glob("app.log*"))), 3)
            self.assertIn("Unicode ✓", target.read_text(encoding="utf-8"))
            for handler in files:
                handler.close()
                self.root.removeHandler(handler)

    def test_child_diagnostics_survive_workspace_cleanup(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
            "LOG_DIR": temp, "LOG_MAX_BYTES": "10485760", "LOG_BACKUP_COUNT": "2", "LOG_LEVEL": "INFO"
        }):
            target = configure_logging()
            with tempfile.TemporaryDirectory() as work:
                with self.assertRaises(RuntimeError):
                    run_process([sys.executable, "-c", "print('child diagnostic'); raise RuntimeError('failure')"],
                                threading.Event(), Path(work) / "rife.log")
            contents = target.read_text(encoding="utf-8")
            self.assertIn("child diagnostic", contents)
            self.assertIn("RuntimeError: failure", contents)
            for handler in self.root.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    self.root.removeHandler(handler)


if __name__ == "__main__":
    unittest.main()