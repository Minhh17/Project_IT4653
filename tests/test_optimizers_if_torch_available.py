import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    import torch
    from dlstudy.config import apply_overrides, load_yaml
    from dlstudy.optimization import build_optimizer

    TORCH_AVAILABLE = hasattr(torch, "optim")
except (ImportError, AttributeError):
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "A complete PyTorch installation is not available")
class OptimizerTests(unittest.TestCase):
    def setUp(self):
        self.base = load_yaml(PROJECT_ROOT / "configs/base.yaml")

    def test_all_six_optimizers_can_update_parameters(self):
        for name in ("sgd", "sgd_momentum", "nesterov", "rmsprop", "adam", "adamw"):
            model = torch.nn.Linear(4, 2)
            config = apply_overrides(self.base, {"optimizer.name": name})
            optimizer = build_optimizer(model, config)
            loss = model(torch.randn(3, 4)).sum()
            loss.backward()
            optimizer.step()
            self.assertEqual(len(optimizer.param_groups), 2)
            self.assertEqual(optimizer.param_groups[1]["weight_decay"], 0.0)


if __name__ == "__main__":
    unittest.main()
