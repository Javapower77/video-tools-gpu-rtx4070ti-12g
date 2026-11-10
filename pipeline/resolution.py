"""Fit output within a landscape FHD/UHD box without cropping or stretching."""


def target_dimensions(width: int, height: int, resolution: str) -> tuple[int, int]:
    if width <= 0 or height <= 0 or resolution not in ("FHD", "4K"):
        raise ValueError("Invalid source dimensions or output resolution.")
    max_width, max_height = (1920, 1080) if resolution == "FHD" else (3840, 2160)
    ratio = min(max_width / width, max_height / height)
    # H.264 yuv420p needs even dimensions; at most two pixels of rounding.
    return max(2, int(width * ratio) // 2 * 2), max(2, int(height * ratio) // 2 * 2)