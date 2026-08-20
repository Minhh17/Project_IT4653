import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    import torch
    from dlstudy.config import apply_overrides, load_yaml
    from dlstudy.optimization import LearningRateSchedule

    TORCH_AVAILABLE = hasattr(torch, "optim")
except (ImportError, AttributeError):
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "A complete PyTorch installation is not available")
class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.base = load_yaml(PROJECT_ROOT / "configs/base.yaml")
        self.parameter = torch.nn.Parameter(torch.tensor(1.0))

    def _schedule(self, overrides, steps_per_epoch=2):
        config = apply_overrides(self.base, overrides)
        optimizer = torch.optim.SGD([self.parameter], lr=config["optimizer"]["lr"])
        return LearningRateSchedule(optimizer, config, steps_per_epoch)

    def test_linear_warmup_reaches_one(self):
        schedule = self._schedule({"train.epochs": 4, "scheduler.warmup_epochs": 1})
        self.assertAlmostEqual(schedule.factor(0), 0.1)
        self.assertAlmostEqual(schedule.factor(1), 1.0)
        self.assertAlmostEqual(schedule.factor(2), 1.0)

    def test_step_decay_uses_epoch_boundary(self):
        schedule = self._schedule(
            {
                "train.epochs": 6,
                "scheduler.name": "step",
                "scheduler.warmup_epochs": 0,
                "scheduler.step_size_epochs": 2,
                "scheduler.gamma": 0.1,
            }
        )
        self.assertAlmostEqual(schedule.factor(3), 1.0)
        self.assertAlmostEqual(schedule.factor(4), 0.1)

    def test_cosine_finishes_at_minimum_lr(self):
        schedule = self._schedule(
            {
                "train.epochs": 2,
                "scheduler.name": "cosine",
                "scheduler.warmup_epochs": 0,
                "scheduler.min_lr": 0.001,
            }
        )
        self.assertAlmostEqual(schedule.apply(3), 0.001)


if __name__ == "__main__":
    unittest.main()
