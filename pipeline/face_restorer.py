from pathlib import Path
import threading
from .common import Report, check_cancel, read_frame, write_frame
from .compat import basicsr_compat
from .config import Settings, directory


def restore(source: Path, dest: Path, count: int, settings: Settings,
            cancel: threading.Event, report: Report) -> None:
    basicsr_compat()
    from gfpgan import GFPGANer
    # GFPGAN's helper internally detects faces and runs its network only on crops.
    # Keep this stage in FP32: its normalization/style layers are not FP16-safe.
    enhancer = GFPGANer(model_path=str(directory("WEIGHTS_DIR", "weights") / "GFPGANv1.4.pth"),
                         upscale=1, arch="clean", channel_multiplier=2, bg_upsampler=None)
    original_forward = enhancer.gfpgan.forward
    def strict_forward(*args, **kwargs):
        try:
            return original_forward(*args, **kwargs)
        except RuntimeError as error:
            # GFPGAN swallows RuntimeError and silently pastes un-restored crops.
            raise ValueError(f"GFPGAN inference failed: {error}") from error
    enhancer.gfpgan.forward = strict_forward
    dest.mkdir()
    for index in range(count):
        check_cancel(cancel)
        frame = read_frame(source / f"{index:08d}.png")
        faces, _, output = enhancer.enhance(frame, has_aligned=False, only_center_face=False,
                                            paste_back=True, weight=settings.face_weight)
        write_frame(dest / f"{index:08d}.png", output if faces and output is not None else frame)
        report((index + 1) / count, f"Face restoration: {index + 1}/{count} ({len(faces)} faces)")