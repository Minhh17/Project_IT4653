import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dlstudy.config import (  # noqa: E402
    ConfigError,
    apply_overrides,
    load_yaml,
    semantic_fingerprint,
    validate_config,
)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.base = load_yaml(PROJECT_ROOT / "configs/base.yaml")

    def test_base_config_is_valid(self):
        validate_config(self.base)

    def test_dotted_override_does_not_mutate_base(self):
        changed = apply_overrides(self.base, {"optimizer.lr": 0.001})
        self.assertEqual(changed["optimizer"]["lr"], 0.001)
        self.assertEqual(self.base["optimizer"]["lr"], 0.1)

    def test_seed_is_ignored_by_semantic_fingerprint(self):
        other_seed = apply_overrides(self.base, {"train.seed": 2026})
        self.assertEqual(semantic_fingerprint(self.base), semantic_fingerprint(other_seed))

    def test_retry_attempt_is_not_a_scientific_change(self):
        retry = apply_overrides(self.base, {"experiment.attempt": "retry1"})
        self.assertEqual(semantic_fingerprint(self.base), semantic_fingerprint(retry))

    def test_invalid_dropout_fails_early(self):
        broken = apply_overrides(self.base, {"model.dropout": 1.0})
        with self.assertRaises(ConfigError):
            validate_config(broken)

    def test_typo_in_override_is_rejected(self):
        with self.assertRaises(ConfigError):
            apply_overrides(self.base, {"optimzer.lr": 0.001})

    def test_quoted_boolean_is_rejected(self):
        broken = apply_overrides(self.base, {"data.augmentation": "false"})
        with self.assertRaises(ConfigError):
            validate_config(broken)

    def test_fractional_epoch_is_not_silently_truncated(self):
        broken = apply_overrides(self.base, {"train.epochs": 20.9})
        with self.assertRaises(ConfigError):
            validate_config(broken)

    def test_group_count_and_patience_must_be_positive(self):
        for key in ("model.group_norm_groups", "regularization.early_stopping.patience"):
            broken = apply_overrides(self.base, {key: 0})
            with self.subTest(key=key), self.assertRaises(ConfigError):
                validate_config(broken)


if __name__ == "__main__":
    unittest.main()
