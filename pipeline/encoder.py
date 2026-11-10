from fractions import Fraction
from pathlib import Path
import threading
from .common import Report, ffmpeg, run_process


def encode(frames: Path, original: Path, output: Path, fps: Fraction, count: int,
           cancel: threading.Event, report: Report, work: Path) -> None:
    progress = work / "encode.progress"
    def tick():
        if progress.exists():
            lines = progress.read_text(encoding="utf-8", errors="replace").splitlines()
            values = [int(line.split("=", 1)[1]) for line in lines
                      if line.startswith("frame=") and line.split("=", 1)[1].strip().isdigit()]
            if values:
                report(min(values[-1] / count, 0.99), f"Encoding: {values[-1]}/{count}")
    # Matroska preserves source audio codecs without an unreliable AAC fallback.
    run_process([ffmpeg(), "-hide_banner", "-y", "-framerate", str(fps), "-start_number", "0",
                 "-i", str(frames / "%08d.png"), "-i", str(original),
                 "-map", "0:v:0", "-map", "1:a?", "-map_metadata", "1", "-map_chapters", "-1",
                 "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                 "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p",
                 "-c:a", "copy", "-t", str(float(Fraction(count) / fps)),
                 "-progress", str(progress), "-nostats", str(output)],
                cancel, work / "encode.log", tick)
    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError("Encoding produced an empty output.")