"""CIFAR data loading with one fixed, reproducible train/validation split."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

from .splits import make_split_indices, split_hash
from .utils import seed_data_worker


_CIFAR_LAYOUTS = {
    "cifar10": {
        "folder": "cifar-10-batches-py",
        "files": [
            "batches.meta",
            "data_batch_1",
            "data_batch_2",
            "data_batch_3",
            "data_batch_4",
            "data_batch_5",
            "test_batch",
        ],
    },
    "cifar100": {"folder": "cifar-100-python", "files": ["meta", "train", "test"]},
}


@dataclass
class DataBundle:
    train: DataLoader
    train_eval: DataLoader
    val: DataLoader
    test: Optional[DataLoader]
    num_classes: int
    split_hash: str
    debug_active: bool


def resolve_cifar_root(data_config: Dict[str, Any], dataset_name: str) -> Path:
    """Return the parent directory expected by torchvision's CIFAR loader.

    Kaggle chooses `/kaggle/input/<dataset-slug>` and the slug can change, so
    official configs use `root: auto`. We search by the stable folder layout,
    not by a community dataset name. Finding zero or two copies is an error so
    the group never silently trains on an unintended input.
    """
    layout = _CIFAR_LAYOUTS[dataset_name]
    configured = str(data_config["root"])
    if configured == "auto":
        kaggle_input = Path("/kaggle/input")
        matches = sorted(path for path in kaggle_input.rglob(layout["folder"]) if path.is_dir())
        if len(matches) != 1:
            raise FileNotFoundError(
                "Expected exactly one /kaggle/input/**/{}, found {}. "
                "Use Add Input with the extracted Python-batch CIFAR dataset, "
                "or set data.root to its parent directory.".format(layout["folder"], len(matches))
            )
        dataset_folder = matches[0]
        root = dataset_folder.parent
    else:
        root = Path(configured)
        dataset_folder = root / layout["folder"]

    missing = [name for name in layout["files"] if not (dataset_folder / name).is_file()]
    if missing and not bool(data_config["download"]):
        raise FileNotFoundError(
            "CIFAR folder {} misses files: {}".format(dataset_folder, ", ".join(missing))
        )
    return root


def _cifar_transforms(config: Dict[str, Any], dataset_name: str):
    if dataset_name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

    evaluation_steps = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if not config["data"]["augmentation"]:
        return transforms.Compose(evaluation_steps), transforms.Compose(evaluation_steps)

    jitter = float(config["data"]["color_jitter"])
    training_steps = [
        # Padding and random crop imitate small translations without resizing CIFAR.
        transforms.RandomCrop(config["data"]["image_size"], padding=config["data"]["crop_padding"]),
        transforms.RandomHorizontalFlip(p=config["data"]["horizontal_flip_probability"]),
        transforms.ColorJitter(
            brightness=jitter,
            contrast=jitter,
            saturation=jitter,
            hue=min(0.1, jitter / 2.0),
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    return transforms.Compose(training_steps), transforms.Compose(evaluation_steps)


def _fake_datasets(config: Dict[str, Any], seed: int):
    """Tiny deterministic tensors for testing code paths without downloading CIFAR."""
    generator = torch.Generator().manual_seed(seed)
    num_classes = 10
    image_size = int(config["data"]["image_size"])
    train_size = int(config["data"]["fake_train_size"])
    val_size = int(config["data"]["fake_val_size"])

    train_images = torch.randn(train_size, 3, image_size, image_size, generator=generator)
    train_labels = torch.randint(num_classes, (train_size,), generator=generator)
    val_images = torch.randn(val_size, 3, image_size, image_size, generator=generator)
    val_labels = torch.randint(num_classes, (val_size,), generator=generator)
    return TensorDataset(train_images, train_labels), TensorDataset(val_images, val_labels)


def build_dataloaders(config: Dict[str, Any], include_test: bool = False) -> DataBundle:
    """Build loaders; the test set is created only for final evaluation."""
    data_config = config["data"]
    debug_config = config["debug"]
    seed = int(config["train"]["seed"])
    dataset_name = data_config["dataset"]
    test_dataset = None

    if dataset_name == "fake":
        train_dataset, val_dataset = _fake_datasets(config, seed)
        clean_train_dataset = train_dataset
        train_indices = list(range(len(train_dataset)))
        val_indices = list(range(len(val_dataset)))
        num_classes = 10
    else:
        train_transform, eval_transform = _cifar_transforms(config, dataset_name)
        dataset_class = datasets.CIFAR10 if dataset_name == "cifar10" else datasets.CIFAR100
        num_classes = 10 if dataset_name == "cifar10" else 100
        dataset_root = resolve_cifar_root(data_config, dataset_name)

        # Two dataset objects point to the same images but use different transforms.
        # Validation must never receive random augmentation.
        full_train = dataset_class(
            root=dataset_root,
            train=True,
            transform=train_transform,
            download=data_config["download"],
        )
        full_eval = dataset_class(
            root=dataset_root,
            train=True,
            transform=eval_transform,
            download=data_config["download"],
        )
        train_indices, val_indices = make_split_indices(
            len(full_train),
            int(data_config["train_size"]),
            int(data_config["val_size"]),
            int(data_config["split_seed"]),
        )
        train_dataset = Subset(full_train, train_indices)
        clean_train_dataset = Subset(full_eval, train_indices)
        val_dataset = Subset(full_eval, val_indices)

        if include_test:
            test_dataset = dataset_class(
                root=dataset_root,
                train=False,
                transform=eval_transform,
                download=data_config["download"],
            )

    debug_samples = debug_config["train_samples"]
    if debug_samples is not None:
        count = min(int(debug_samples), len(train_dataset))
        if dataset_name == "fake":
            train_dataset = Subset(train_dataset, list(range(count)))
            clean_train_dataset = train_dataset
            train_indices = train_indices[:count]
        else:
            selected_indices = train_indices[:count]
            train_dataset = Subset(full_train, selected_indices)
            clean_train_dataset = Subset(full_eval, selected_indices)
            train_indices = selected_indices

    if debug_config["use_train_as_val"]:
        # Evaluate exactly the actual (possibly truncated) train subset with a
        # clean transform. Updating val_indices makes the checksum tell the
        # truth about this diagnostic split instead of imitating the official one.
        val_dataset = clean_train_dataset
        val_indices = list(train_indices)

    debug_active = any(
        (
            debug_config["max_train_batches"] is not None,
            debug_config["max_val_batches"] is not None,
            debug_config["train_samples"] is not None,
            debug_config["use_train_as_val"],
        )
    )

    common = {
        "batch_size": int(data_config["batch_size"]),
        "num_workers": int(data_config["num_workers"]),
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_data_worker,
    }
    # Separate generators prevent validation worker creation from changing the
    # random order of next epoch's training sampler.
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=torch.Generator().manual_seed(seed), **common
    )
    train_eval_loader = DataLoader(
        clean_train_dataset,
        shuffle=False,
        generator=torch.Generator().manual_seed(seed + 1),
        **common,
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, generator=torch.Generator().manual_seed(seed + 2), **common
    )
    test_loader = (
        DataLoader(
            test_dataset, shuffle=False, generator=torch.Generator().manual_seed(seed + 3), **common
        )
        if test_dataset
        else None
    )

    return DataBundle(
        train=train_loader,
        train_eval=train_eval_loader,
        val=val_loader,
        test=test_loader,
        num_classes=num_classes,
        split_hash=split_hash(train_indices, val_indices),
        debug_active=debug_active,
    )
