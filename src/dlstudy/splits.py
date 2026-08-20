"""Pure-Python dataset split helpers, isolated so they are easy to test."""

import hashlib
import json
import random
from typing import List, Tuple


def make_split_indices(
    total_size: int, train_size: int, val_size: int, split_seed: int
) -> Tuple[List[int], List[int]]:
    """Create the same split on every machine using Python's seeded shuffle."""
    if train_size < 1 or val_size < 1:
        raise ValueError("train_size and val_size must both be positive")
    if train_size + val_size > total_size:
        raise ValueError("Requested split is larger than the dataset")
    indices = list(range(total_size))
    random.Random(split_seed).shuffle(indices)
    return indices[:train_size], indices[train_size : train_size + val_size]


def split_hash(train_indices: List[int], val_indices: List[int]) -> str:
    """A short checksum lets the team verify that all runs used one split."""
    raw = json.dumps({"train": train_indices, "val": val_indices}).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]
