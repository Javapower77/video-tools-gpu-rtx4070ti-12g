import sys
import types


def basicsr_compat() -> None:
    """BasicSR 1.4.2 imports a torchvision module removed in 0.17+."""
    from torchvision.transforms.functional import rgb_to_grayscale
    name = "torchvision.transforms.functional_tensor"
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.rgb_to_grayscale = rgb_to_grayscale
        sys.modules[name] = module