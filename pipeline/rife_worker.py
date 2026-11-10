"""Isolated process: trusted Practical-RIFE archive owns train_log imports."""
import argparse
from fractions import Fraction
import importlib
from pathlib import Path
import sys
import types


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("source", "dest", "fps", "target", "repo", "weights"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipeline.interpolator import output_count, frame_position
    from pipeline.common import read_frame, write_frame
    import cv2
    import numpy as np
    import torch
    from torch.nn import functional as F
    sys.path.insert(0, args.repo)
    from pipeline.rife_precision import install_dtype_safe_warp
    install_dtype_safe_warp()
    package = types.ModuleType("train_log")
    package.__path__ = [args.weights]
    sys.modules["train_log"] = package
    model_class = importlib.import_module("train_log.RIFE_HDv3").Model
    model = model_class()
    model.load_model(args.weights, -1)
    model.eval()
    model.device()
    if getattr(model, "version", 0) < 3.9:
        raise RuntimeError("RIFE model must support arbitrary timestep inference (version >= 3.9).")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    half = args.fp16 and device.type == "cuda"
    if half:
        model.flownet.half()
    fps, target = Fraction(args.fps), Fraction(args.target)
    cached_left = -1
    left = right = first = second = None
    with torch.inference_mode():
        for index in range(output_count(args.count, fps, target)):
            position, fraction = frame_position(index, args.count, fps, target)
            if position != cached_left:
                left = read_frame(Path(args.source) / f"{position:08d}.png")
                right = read_frame(Path(args.source) / f"{min(position + 1, args.count - 1):08d}.png")
                h, w = left.shape[:2]
                block = max(128, int(128 / args.scale))
                padding = (0, (-w) % block, 0, (-h) % block)
                def tensor(image):
                    value = torch.from_numpy(image[:, :, ::-1].copy()).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255
                    value = F.pad(value, padding)
                    return value.half() if half else value
                first, second = tensor(left), tensor(right)
                # Conservative hard-cut guard; no synthesized morph across a cut.
                delta = np.mean(np.abs(cv2.resize(left, (32, 32)).astype(np.float32)
                                       - cv2.resize(right, (32, 32)).astype(np.float32))) / 255
                cached_left = position
            if fraction == 0 or delta > 0.35:
                output = left
            else:
                prediction = model.inference(first, second, fraction, args.scale)
                output = prediction[0].clamp(0, 1).mul(255).round().byte().cpu().numpy().transpose(1, 2, 0)[:h, :w, ::-1].copy()
            write_frame(Path(args.dest) / f"{index:08d}.png", output)


if __name__ == "__main__":
    main()