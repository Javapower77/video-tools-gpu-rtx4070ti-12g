from dataclasses import dataclass
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("RealESRGAN_x2plus", "RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B",
          "realesr-animevideov3", "realesr-general-x4v3", "4x-UltraSharp", "SeedVR2")


def directory(key: str, default: str) -> Path:
    path = Path(os.getenv(key, default))
    return path if path.is_absolute() else ROOT / path


@dataclass(frozen=True)
class Settings:
    upscale_model: str = "RealESRGAN_x2plus"
    scale: int = 2
    face_model: str = "GFPGAN"
    face_weight: float = 0.5
    rife_version: str = "4.26"
    target_fps: float = 60
    fp16: bool = True
    tile: int = 256
    rife_scale: float = 1.0
    keep_temp: bool = False
    output_resolution: str = "Scale"
    seedvr_batch_size: int = 5
    seedvr_noise: float = 0.03
    comfy_checkpoint: str = "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"
    comfy_detector: str = "bbox/face_yolov8m.pt"
    comfy_positive: str = "photorealistic face, natural skin texture, subtle pores, preserve facial features, consistent lighting"
    comfy_negative: str = "plastic skin, waxy skin, oversharpened, deformed face, artifacts, excessive skin texture"
    comfy_denoise: float = 0.25
    comfy_steps: int = 20
    comfy_cfg: float = 5.0
    comfy_seed: int = 42
    comfy_guide_size: int = 768

    def validate(self) -> None:
        if self.upscale_model not in MODELS or self.scale not in (1, 2, 4):
            raise ValueError("Invalid upscale model or scale.")
        if self.face_model not in ("GFPGAN", "ComfyUI", "Off"):
            raise ValueError("Supported face models: ComfyUI, GFPGAN, Off; CodeFormer is not installed.")
        if self.rife_version not in ("4.26", "4.25", "Off"):
            raise ValueError("Invalid RIFE version.")
        if not 1 <= self.target_fps <= 240 or not 0 <= self.face_weight <= 1:
            raise ValueError("FPS must be 1–240 and face weight 0–1.")
        if self.tile < 0 or self.tile > 2048 or self.rife_scale not in (0.25, 0.5, 1.0):
            raise ValueError("Invalid tile size or RIFE processing scale.")
        if self.output_resolution not in ("Scale", "FHD", "4K"):
            raise ValueError("Output resolution must be Scale, FHD or 4K.")
        if self.upscale_model == "SeedVR2":
            if self.output_resolution == "Scale":
                raise ValueError("SeedVR2 requires FHD or 4K output resolution.")
            if type(self.seedvr_batch_size) is not int or self.seedvr_batch_size not in (1, 5, 9):
                raise ValueError("SeedVR2 batch size must be 1, 5 or 9.")
            if not 0 <= self.seedvr_noise <= 1:
                raise ValueError("SeedVR2 latent noise must be 0–1.")
        if self.face_model == "ComfyUI":
            if not 0 < self.comfy_denoise <= 1 or not 1 <= self.comfy_cfg <= 20:
                raise ValueError("FaceDetailer denoise must be >0–1 and CFG 1–20.")
            for value, low, high in ((self.comfy_steps, 1, 100),
                                     (self.comfy_seed, 0, 2**63 - 1),
                                     (self.comfy_guide_size, 256, 1024)):
                if type(value) is not int or not low <= value <= high:
                    raise ValueError("Invalid FaceDetailer steps, seed or guide size.")
            if not self.comfy_checkpoint or not self.comfy_detector:
                raise ValueError("Select a ComfyUI checkpoint and face detector.")