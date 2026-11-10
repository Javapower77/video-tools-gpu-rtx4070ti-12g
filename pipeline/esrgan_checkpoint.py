"""Translate legacy ESRGAN RRDB keys into BasicSR RRDBNet keys."""
import re


def normalize_ultrasharp(checkpoint: dict) -> dict:
    state = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
    if not isinstance(state, dict):
        raise ValueError("UltraSharp checkpoint must contain a state dictionary.")
    state = {key.removeprefix("module."): value for key, value in state.items()}
    if "conv_first.weight" in state:
        return state
    converted = {}
    layers = {"0": "conv_first", "3": "conv_up1", "6": "conv_up2",
              "8": "conv_hr", "10": "conv_last"}
    for key, value in state.items():
        block = re.fullmatch(r"model\.1\.sub\.(\d+)\.RDB([123])\.conv([1-5])(?:\.0)?\.(weight|bias)", key)
        if block:
            index, rdb, conv, parameter = block.groups()
            target = f"body.{index}.rdb{rdb}.conv{conv}.{parameter}"
        elif key in ("model.1.sub.23.weight", "model.1.sub.23.bias"):
            target = "conv_body." + key.rsplit(".", 1)[1]
        else:
            layer = re.fullmatch(r"model\.(0|3|6|8|10)\.(weight|bias)", key)
            if not layer:
                raise ValueError(f"Unsupported UltraSharp checkpoint key: {key}")
            target = layers[layer[1]] + "." + layer[2]
        converted[target] = value
    return converted