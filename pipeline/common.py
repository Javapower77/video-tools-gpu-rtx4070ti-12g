from pathlib import Path
import os
import subprocess
import threading
from typing import Callable
from .logging_setup import record_process_log

Report = Callable[[float, str], None]


class CancelledError(RuntimeError):
    pass


def check_cancel(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise CancelledError("Processing cancelled.")


def run_process(args: list[str], cancel: threading.Event, log: Path,
                tick: Callable[[], None] | None = None) -> None:
    """No shell; file-backed output avoids pipe deadlocks and unbounded RAM."""
    check_cancel(cancel)
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(args, stdout=output, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            while process.poll() is None:
                if cancel.wait(0.2):
                    raise CancelledError("Processing cancelled.")
                if tick:
                    tick()
            check_cancel(cancel)
            if process.returncode:
                with log.open("rb") as diagnostic:
                    diagnostic.seek(max(0, log.stat().st_size - 6000))
                    tail = diagnostic.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"{Path(args[0]).name} failed ({process.returncode}):\n{tail}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            output.flush()
            record_process_log(log, process.returncode)


def ffmpeg() -> str:
    return os.getenv("FFMPEG", "ffmpeg")


def read_frame(path: Path):
    import cv2
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot decode frame: {path}")
    return image


def write_frame(path: Path, image) -> None:
    import cv2
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Cannot write frame (check free disk space): {path}")


def release_gpu() -> None:
    import gc
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()