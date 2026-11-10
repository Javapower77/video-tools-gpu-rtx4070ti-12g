from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import json
import math
import os
import threading
from .common import Report, ffmpeg, run_process


@dataclass(frozen=True)
class VideoInfo:
    fps: Fraction
    duration: float
    width: int
    height: int
    stream_index: int = 0


def parse_probe(data: dict) -> VideoInfo:
    streams = data.get("streams", [])
    stream = next((s for s in streams if s.get("codec_type") == "video"
                   and not s.get("disposition", {}).get("attached_pic", 0)), None)
    if stream is None:
        raise ValueError("Input contains no video stream.")
    try:
        average = stream.get("avg_frame_rate") or "0"
        fps = Fraction(average) if average != "0/0" else Fraction(0)
        if fps <= 0:
            fps = Fraction(stream.get("r_frame_rate") or "0")
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("Input has invalid frame rate.") from error
    pixel_format = stream.get("pix_fmt", "")
    if stream.get("color_transfer") in ("smpte2084", "arib-std-b67") or any(
        marker in pixel_format for marker in ("10", "12", "14", "16", "f32")
    ):
        raise ValueError("HDR/high-bit-depth input is not supported. Convert to SDR 8-bit first.")
    duration = float(stream.get("duration") or data.get("format", {}).get("duration", 0))
    width, height = int(stream["width"]), int(stream["height"])
    if fps <= 0 or fps > 1000 or not math.isfinite(duration) or duration <= 0 or min(width, height) <= 0:
        raise ValueError("Input has invalid FPS, duration, or dimensions.")
    return VideoInfo(fps, duration, width, height, int(stream.get("index", 0)))


def probe(source: Path, cancel: threading.Event, work: Path) -> VideoInfo:
    log = work / "probe.json"
    run_process([os.getenv("FFPROBE", "ffprobe"), "-v", "error", "-show_streams",
                 "-show_format", "-of", "json", str(source)], cancel, log)
    return parse_probe(json.loads(log.read_text(encoding="utf-8")))


def extract(source: Path, dest: Path, info: VideoInfo, cancel: threading.Event, report: Report) -> int:
    dest.mkdir()
    estimate = max(1, round(info.duration * float(info.fps)))
    def tick():
        count = sum(1 for _ in dest.glob("*.png"))
        report(min(count / estimate, 0.99), f"Extracting frames: {count}")
    run_process([ffmpeg(), "-hide_banner", "-y", "-i", str(source), "-map", f"0:{info.stream_index}",
                 "-an", "-vf", f"fps={info.fps}", "-start_number", "0",
                 str(dest / "%08d.png")], cancel, dest.parent / "extract.log", tick)
    count = sum(1 for _ in dest.glob("*.png"))
    if count == 0:
        raise RuntimeError("FFmpeg extracted no frames.")
    return count