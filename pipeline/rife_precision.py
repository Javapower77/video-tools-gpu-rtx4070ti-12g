"""Application-owned inference warp; never modify the vendor checkout."""
from functools import lru_cache


def install_dtype_safe_warp() -> None:
    """Install before importing train_log: IFNet binds warp with a from-import."""
    import torch
    from torch.nn import functional as F
    from model import warplayer

    @lru_cache(maxsize=8)
    def base_grid(device, height: int, width: int):
        # Compute coordinates in FP32, including on FP16 inference paths.
        horizontal = torch.linspace(-1, 1, width, device=device, dtype=torch.float32)
        vertical = torch.linspace(-1, 1, height, device=device, dtype=torch.float32)
        y, x = torch.meshgrid(vertical, horizontal, indexing="ij")
        return torch.stack((x, y), dim=-1).unsqueeze(0)

    def warp(image, flow):
        height, width = image.shape[-2:]
        if height < 2 or width < 2:
            raise ValueError("RIFE warp requires both spatial dimensions >= 2.")
        displacement = flow.to(device=image.device, dtype=torch.float32).permute(0, 2, 3, 1)
        normalization = displacement.new_tensor(((width - 1) / 2, (height - 1) / 2))
        grid = base_grid(image.device, height, width) + displacement / normalization
        # grid_sample requires input and grid to have the same scalar type.
        return F.grid_sample(image, grid.to(dtype=image.dtype), mode="bilinear",
                             padding_mode="border", align_corners=True)

    warplayer.warp = warp