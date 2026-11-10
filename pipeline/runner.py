from fractions import Fraction
from pathlib import Path
import shutil
import threading
import uuid
from .common import Report, check_cancel, release_gpu
from .config import Settings, directory
from .extractor import extract, probe
from .upscaler import upscale
from .face_restorer import restore
from .interpolator import interpolate
from .encoder import encode
from .file_dates import windows_file_times
from .resolution import target_dimensions


def preflight(settings: Settings) -> None:
    import os
    for key, default in (("FFMPEG", "ffmpeg"), ("FFPROBE", "ffprobe")):
        if not shutil.which(os.getenv(key, default)):
            raise RuntimeError(f"{default} not found. Install FFmpeg and add its bin directory to PATH.")
    weights = directory("WEIGHTS_DIR", "weights")
    required = []
    if (settings.scale != 1 or settings.output_resolution != "Scale") and settings.upscale_model != "SeedVR2":
        required.append(weights / f"{settings.upscale_model}.pth")
    if settings.face_model == "GFPGAN":
        required.append(weights / "GFPGANv1.4.pth")
        # GFPGAN internally uses this relative cache; populate it before creation.
        cache = Path("gfpgan/weights")
        cache.mkdir(parents=True, exist_ok=True)
        for name in ("detection_Resnet50_Final.pth", "parsing_parsenet.pth"):
            source = weights / "facexlib" / name
            required.append(source)
            if source.exists() and not (cache / name).exists():
                shutil.copy2(source, cache / name)
    if settings.rife_version != "Off":
        folder = weights / "rife" / settings.rife_version
        required.extend((folder / "flownet.pkl", folder / "RIFE_HDv3.py"))
        if not directory("RIFE_REPO", "vendor/Practical-RIFE").is_dir():
            raise RuntimeError("Practical-RIFE checkout missing. Run scripts/Setup.ps1.")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing models. Run scripts/Download-Models.ps1; install RIFE archive manually:\n" + "\n".join(missing))


def run_video(source: Path, settings: Settings, cancel: threading.Event, report: Report,
              *, save_beside_source: bool = False, batch_mode: bool = False) -> Path:
    settings.validate()
    source = source.resolve(strict=True)
    if not source.is_file():
        raise ValueError("Input must be a file.")
    check_cancel(cancel)
    source_times = windows_file_times(source) if save_beside_source else None
    preflight(settings)
    import torch
    gpu = torch.cuda.is_available()
    report(0, f"Device: {torch.cuda.get_device_name(0) if gpu else 'CPU (slow)'}; GFPGAN uses FP32.")
    token = uuid.uuid4().hex
    work = directory("TEMP_DIR", "temp") / token
    outputs = source.parent if save_beside_source else directory("OUTPUT_DIR", "outputs")
    outputs.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True)
    final = outputs / (f"{source.stem}_{token[:8]}.mkv" if save_beside_source
                       else f"{source.stem[:80]}_{token[:12]}.mkv")
    partial = outputs / f".{token}.partial.mkv"
    def stage(start: float, span: float):
        return lambda progress, message: report(start + span * progress, message)
    try:
        info = probe(source, cancel, work)
        input_count = max(1, round(info.duration * float(info.fps)))
        target_count = input_count if settings.rife_version == "Off" else max(
            1, round(input_count * settings.target_fps / float(info.fps)))
        # Conservative sum of stages, including higher target FPS and PNG overhead.
        output_pixels = info.width * info.height * settings.scale ** 2
        if settings.output_resolution != "Scale":
            out_width, out_height = target_dimensions(info.width, info.height, settings.output_resolution)
            output_pixels = out_width * out_height
        estimate = 3 * 1.1 * (info.width * info.height * (input_count * 2 + target_count)
                              + output_pixels * target_count)
        free = shutil.disk_usage(work).free
        if estimate > free:
            raise OSError(f"Estimated frame storage {estimate / 2**30:.1f} GiB exceeds {free / 2**30:.1f} GiB free. Use a shorter clip or lower scale.")
        report(0.01, "Normalizing video to constant frame rate; output is SDR 8-bit.")
        frames = work / "extracted"
        count = extract(source, frames, info, cancel, stage(0.01, 0.14))
        if settings.face_model != "Off":
            next_frames = work / "restored"
            try:
                if settings.face_model == "ComfyUI":
                    from .comfyui_backend import restore as detail_faces
                    detail_faces(frames, next_frames, count, settings, cancel, stage(0.15, 0.35))
                else:
                    restore(frames, next_frames, count, settings, cancel, stage(0.15, 0.35))
            finally:
                release_gpu()
            if not settings.keep_temp:
                shutil.rmtree(frames)
            frames = next_frames
        fps = info.fps
        if settings.rife_version != "Off" and Fraction(str(settings.target_fps)) != fps:
            next_frames = work / "interpolated"
            count = interpolate(frames, next_frames, count, fps, settings, cancel, stage(0.5, 0.2))
            fps = Fraction(str(settings.target_fps))
            if not settings.keep_temp:
                shutil.rmtree(frames)
            frames = next_frames
        elif settings.rife_version == "Off":
            report(0.7, f"RIFE disabled: keeping source FPS ({float(fps):.3f}).")
        if settings.scale != 1 or settings.output_resolution != "Scale":
            next_frames = work / "upscaled"
            try:
                if settings.upscale_model == "SeedVR2":
                    from .seedvr2_backend import upscale as seed_upscale
                    seed_upscale(frames, next_frames, count, fps, settings, cancel,
                                 stage(0.7, 0.2), batch_mode=batch_mode)
                else:
                    upscale(frames, next_frames, count, settings, cancel, stage(0.7, 0.2))
            finally:
                release_gpu()
            if not settings.keep_temp:
                shutil.rmtree(frames)
            frames = next_frames
        encode(frames, source, partial, fps, count, cancel, stage(0.9, 0.09), work)
        check_cancel(cancel)
        if source_times is not None:
            windows_file_times(partial, source_times)
            # Check exact creation/write precision before publishing the result.
            restored = windows_file_times(partial)
            if restored[0] != source_times[0] or restored[2] != source_times[2]:
                raise OSError("Destination filesystem did not preserve source creation/modified dates exactly.")
            # Windows rename fails if the randomly named destination already exists.
            partial.rename(final)
        else:
            partial.replace(final)
        report(1, f"Completed: {final.name}")
        return final
    finally:
        partial.unlink(missing_ok=True)
        if not settings.keep_temp:
            shutil.rmtree(work, ignore_errors=True)
        else:
            report(1 if final.exists() else 0, f"Temporary files retained: {work}")