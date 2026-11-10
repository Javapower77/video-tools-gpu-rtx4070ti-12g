"""Reviewed presets; selecting one copies values, not hardware guarantees."""

from dataclasses import dataclass
from types import MappingProxyType

from pipeline.config import Settings


@dataclass(frozen=True)
class Profile:
    settings: Settings
    description: str


PROFILES = MappingProxyType({
    "Normal": Profile(
        Settings(upscale_model="4x-UltraSharp", face_model="ComfyUI", output_resolution="FHD"),
        "**Normal · FHD:** ComfyUI FaceDetailer (Juggernaut XL, denoise 0.25) → "
        "RIFE 4.26 at 60 FPS → 4x-UltraSharp fitted within 1920×1080. "
        "Tile 256, RIFE scale 1. Generated skin detail can flicker or change identity.",
    ),
    "Boost": Profile(
        Settings(upscale_model="4x-UltraSharp", scale=4, face_model="ComfyUI", output_resolution="4K"),
        "**Boost · 4K:** ComfyUI FaceDetailer (Juggernaut XL, denoise 0.25) → "
        "RIFE 4.26 at 60 FPS → 4x-UltraSharp fitted within 3840×2160. "
        "**12 GB VRAM warning:** use dedicated ComfyUI, guide 768 and tile 256. "
        "Models are unloaded before RIFE. Test a short clip; output quality and "
        "temporal consistency are not guaranteed. RIFE Heavy is excluded.",
    ),
})