"""Optional installed-stack regression: skips when torch/vendor are absent."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
AVAILABLE = importlib.util.find_spec("torch") is not None and (
    ROOT / "vendor/Practical-RIFE/model/warplayer.py").is_file()


@unittest.skipUnless(AVAILABLE, "Requires installed torch and Practical-RIFE")
class WarpPrecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        sys.path.insert(0, str(ROOT / "vendor/Practical-RIFE"))
        from pipeline.rife_precision import install_dtype_safe_warp
        install_dtype_safe_warp()
        from model import warplayer
        cls.warplayer = warplayer

    def test_cpu_float_identity(self):
        torch = self.torch
        image = torch.rand(1, 3, 16, 32)
        result = self.warplayer.warp(image, torch.zeros(1, 2, 16, 32))
        self.assertEqual(result.dtype, image.dtype)
        torch.testing.assert_close(result, image, atol=3e-6, rtol=1e-5)

    def test_cuda_mixed_precision_cache(self):
        torch = self.torch
        if not torch.cuda.is_available():
            self.skipTest("Requires CUDA for FP16 grid_sample")
        # Identical shapes, alternating types: a cached FP32 base must not leak
        # into the grid_sample input even after a previous FP32 invocation.
        for dtype in (torch.float32, torch.float16, torch.float32, torch.float16):
            image = torch.ones(2, 3, 32, 32, device="cuda", dtype=dtype)
            for flow_dtype in (torch.float32, torch.float16):
                flow = torch.zeros(2, 2, 32, 32, device="cuda", dtype=flow_dtype)
                result = self.warplayer.warp(image, flow)
                self.assertEqual(result.dtype, dtype)
                self.assertEqual(result.device, image.device)
                torch.testing.assert_close(result, image)


if __name__ == "__main__":
    unittest.main()