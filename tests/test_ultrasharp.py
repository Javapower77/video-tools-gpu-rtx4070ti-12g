import unittest
from pipeline.config import Settings
from pipeline.esrgan_checkpoint import normalize_ultrasharp


class UltraSharpTests(unittest.TestCase):
    def test_model_selection(self):
        Settings(upscale_model="4x-UltraSharp", scale=4).validate()

    def test_legacy_mapping(self):
        state = {"model.0.weight": 1, "model.1.sub.0.RDB1.conv1.0.weight": 2,
                 "model.1.sub.22.RDB3.conv5.bias": 3, "model.1.sub.23.weight": 4,
                 "model.3.weight": 5, "model.6.bias": 6, "model.8.weight": 7,
                 "model.10.bias": 8}
        self.assertEqual(normalize_ultrasharp(state), {
            "conv_first.weight": 1, "body.0.rdb1.conv1.weight": 2,
            "body.22.rdb3.conv5.bias": 3, "conv_body.weight": 4,
            "conv_up1.weight": 5, "conv_up2.bias": 6, "conv_hr.weight": 7,
            "conv_last.bias": 8})

    def test_native_wrapped_and_module_prefix(self):
        self.assertEqual(normalize_ultrasharp({"params_ema": {"module.conv_first.weight": 1}}),
                         {"conv_first.weight": 1})

    def test_unknown_layout_fails(self):
        with self.assertRaises(ValueError):
            normalize_ultrasharp({"unexpected.weight": 1})