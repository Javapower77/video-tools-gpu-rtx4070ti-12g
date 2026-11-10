from fractions import Fraction
from pathlib import Path
import sys
import threading
from .common import Report, run_process
from .config import ROOT, Settings, directory


def output_count(count: int, source_fps: Fraction, target_fps: Fraction) -> int:
    return max(1, round(count * target_fps / source_fps))


def frame_position(index: int, count: int, source_fps: Fraction, target_fps: Fraction) -> tuple[int, float]:
    position = index * source_fps / target_fps
    left = min(int(position), count - 1)
    return left, float(position - left) if left < count - 1 else 0.0


def interpolate(source: Path, dest: Path, count: int, fps: Fraction, settings: Settings,
                cancel: threading.Event, report: Report) -> int:
    target = Fraction(str(settings.target_fps))
    total = output_count(count, fps, target)
    dest.mkdir()
    def tick():
        done = sum(1 for _ in dest.glob("*.png"))
        report(min(done / total, 0.99), f"Interpolating: {done}/{total}")
    args = [sys.executable, str(ROOT / "pipeline" / "rife_worker.py"),
            "--source", str(source), "--dest", str(dest), "--count", str(count),
            "--fps", str(fps), "--target", str(target), "--scale", str(settings.rife_scale),
            "--repo", str(directory("RIFE_REPO", "vendor/Practical-RIFE")),
            "--weights", str(directory("WEIGHTS_DIR", "weights") / "rife" / settings.rife_version)]
    if settings.fp16:
        args.append("--fp16")
    run_process(args, cancel, dest.parent / "rife.log", tick)
    if sum(1 for _ in dest.glob("*.png")) != total:
        raise RuntimeError("RIFE did not produce the expected number of frames.")
    return total