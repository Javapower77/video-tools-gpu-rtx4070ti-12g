from pathlib import Path
import tempfile
import threading
from .common import Report, check_cancel, read_frame, write_frame
from .compat import basicsr_compat
from .config import Settings, directory
from .resolution import target_dimensions


def upscale(source: Path, dest: Path, count: int, settings: Settings,
            cancel: threading.Event, report: Report) -> None:
    import torch
    basicsr_compat()
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact
    class FailFastUpscaler(RealESRGANer):
        """Upstream tile_process suppresses RuntimeError; never reuse stale tiles."""
        def tile_process(self):
            import math
            batch, channel, height, width = self.img.shape
            self.output = self.img.new_zeros((batch, channel, height * self.scale, width * self.scale))
            for row in range(math.ceil(height / self.tile_size)):
                for col in range(math.ceil(width / self.tile_size)):
                    check_cancel(cancel)
                    x0, y0 = col * self.tile_size, row * self.tile_size
                    x1, y1 = min(x0 + self.tile_size, width), min(y0 + self.tile_size, height)
                    px0, py0 = max(x0 - self.tile_pad, 0), max(y0 - self.tile_pad, 0)
                    px1, py1 = min(x1 + self.tile_pad, width), min(y1 + self.tile_pad, height)
                    tile = self.model(self.img[:, :, py0:py1, px0:px1])
                    s = self.scale
                    self.output[:, :, y0*s:y1*s, x0*s:x1*s] = tile[
                        :, :, (y0-py0)*s:(y1-py0)*s, (x0-px0)*s:(x1-px0)*s]
    name = settings.upscale_model
    native_scale = 2 if name == "RealESRGAN_x2plus" else 4
    if name.startswith("realesr-"):
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64,
                               num_conv=16 if name == "realesr-animevideov3" else 32,
                               upscale=4, act_type="prelu")
    else:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=6 if name.endswith("6B") else 23,
                        num_grow_ch=32, scale=native_scale)
    model_path = directory("WEIGHTS_DIR", "weights") / f"{name}.pth"
    # RealESRGANer expects BasicSR params wrappers; UltraSharp may be a raw
    # legacy ESRGAN state dict. Normalize privately without changing weights.
    with tempfile.TemporaryDirectory(prefix="ultrasharp-checkpoint-") as temporary:
        if name == "4x-UltraSharp":
            from .esrgan_checkpoint import normalize_ultrasharp
            state = normalize_ultrasharp(torch.load(model_path, map_location="cpu", weights_only=True))
            model.load_state_dict(state, strict=True)
            model_path = Path(temporary) / "normalized.pth"
            torch.save({"params": state}, model_path)
            del state
        enhancer = FailFastUpscaler(scale=native_scale, model_path=str(model_path),
                                model=model, tile=settings.tile, tile_pad=10, pre_pad=0,
                                half=settings.fp16 and torch.cuda.is_available(),
                                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    dest.mkdir()
    with torch.inference_mode():
        for index in range(count):
            check_cancel(cancel)
            frame = read_frame(source / f"{index:08d}.png")
            try:
                if settings.output_resolution == "Scale":
                    output, _ = enhancer.enhance(frame, outscale=settings.scale)
                else:
                    import cv2
                    height, width = frame.shape[:2]
                    target = target_dimensions(width, height, settings.output_resolution)
                    factor = target[0] / width
                    output, _ = enhancer.enhance(frame, outscale=factor)
                    if output is not None and (output.shape[1], output.shape[0]) != target:
                        output = cv2.resize(output, target, interpolation=cv2.INTER_LANCZOS4)
            except torch.cuda.OutOfMemoryError as error:
                raise RuntimeError("GPU memory exhausted. Reduce tile size to 128 or scale to 2x.") from error
            if output is None:
                raise RuntimeError("Upscaler returned no frame; reduce tile size.")
            write_frame(dest / f"{index:08d}.png", output)
            report((index + 1) / count, f"Upscaling: {index + 1}/{count}")