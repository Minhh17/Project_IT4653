import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    import torch
    from dlstudy.model import build_model

    TORCH_AVAILABLE = hasattr(torch, "randn")
except (ImportError, AttributeError):
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "A complete PyTorch installation is not available")
class ModelTests(unittest.TestCase):
    def test_all_normalizations_preserve_output_shape(self):
        for normalization in ("batch", "layer", "group"):
            config = {
                "name": "resnet18_cifar",
                "base_channels": 8,
                "normalization": normalization,
                "group_norm_groups": 4,
                "dropout": 0.1,
            }
            model = build_model(config, num_classes=10)
            output = model(torch.randn(2, 3, 32, 32))
            self.assertEqual(tuple(output.shape), (2, 10))


if __name__ == "__main__":
    unittest.main()
