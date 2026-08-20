"""The complete train/validation loop for one controlled experiment."""

from __future__ import annotations

import csv
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import semantic_fingerprint, validate_config
from .data import build_dataloaders
from .model import build_model, count_trainable_parameters
from .optimization import LearningRateSchedule, build_criterion, build_optimizer
from .utils import (
    choose_device,
    collect_environment,
    safe_name,
    seed_everything,
    write_json,
    write_yaml,
)


STEP_COLUMNS = [
    "epoch",
    "global_step",
    "batch",
    "loss",
    "accuracy",
    "learning_rate",
    "optimizer_step_applied",
    "samples_seen",
    "elapsed_seconds",
]
EPOCH_COLUMNS = [
    "epoch",
    "global_step",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "online_train_val_gap",
    "learning_rate",
    "epoch_seconds",
]


class CsvWriter:
    """A tiny CSV logger that flushes rows so interrupted runs retain evidence."""

    def __init__(self, path: Path, columns):
        self.file = Path(path).open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=columns)
        self.writer.writeheader()

    def write(self, row: Dict[str, Any]) -> None:
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return float((predictions == targets).sum().item()) / targets.size(0)


def _sync_cuda(device: torch.device) -> None:
    """Wait for asynchronous GPU work before measuring wall-clock time."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_schedule: LearningRateSchedule,
    scaler,
    device: torch.device,
    epoch: int,
    global_step: int,
    step_writer: CsvWriter,
    log_every_steps: int,
    max_batches: Optional[int],
    run_start_time: float,
) -> Tuple[float, float, int, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    sample_count = 0
    current_lr = optimizer.param_groups[0]["lr"]
    amp_enabled = scaler.is_enabled()

    for batch_index, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Apply LR before this optimizer update. This makes step zero explicit.
        current_lr = lr_schedule.apply(global_step)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(
                "Non-finite training loss at global step {}".format(global_step)
            )

        # GradScaler avoids float16 underflow; when AMP is off it behaves like
        # an ordinary backward/step pair.
        scale_before_update = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # GradScaler skips optimizer.step when it detects Inf/NaN gradients.
        # A lower scale after update is PyTorch's observable signal for that
        # skip. The LR schedule advances only after a real parameter update.
        optimizer_step_applied = not amp_enabled or scaler.get_scale() >= scale_before_update

        batch_size = targets.size(0)
        batch_correct = int((logits.argmax(dim=1) == targets).sum().item())
        loss_sum += float(loss.item()) * batch_size
        correct += batch_correct
        sample_count += batch_size

        if global_step % log_every_steps == 0:
            step_writer.write(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "batch": batch_index,
                    "loss": float(loss.item()),
                    "accuracy": batch_correct / batch_size,
                    "learning_rate": current_lr,
                    "optimizer_step_applied": optimizer_step_applied,
                    "samples_seen": sample_count,
                    "elapsed_seconds": time.perf_counter() - run_start_time,
                }
            )
        if optimizer_step_applied:
            global_step += 1

    if sample_count == 0:
        raise RuntimeError("Training loader produced no samples")
    return loss_sum / sample_count, correct / sample_count, global_step, current_lr


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: Optional[int] = None,
    amp_enabled: bool = False,
) -> Tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    sample_count = 0
    for batch_index, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("Non-finite evaluation loss at batch {}".format(batch_index))
        batch_size = targets.size(0)
        loss_sum += float(loss.item()) * batch_size
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        sample_count += batch_size
    if sample_count == 0:
        raise RuntimeError("Evaluation loader produced no samples")
    return loss_sum / sample_count, correct / sample_count


def _save_checkpoint(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(str(temporary), str(path))


def _portable_path(path: Path, project_root: Path) -> str:
    """Prefer a repository-relative path, but allow temporary smoke-test output."""
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _early_stopping_improved(
    monitor: str, current: float, best: Optional[float], min_delta: float
) -> bool:
    if best is None:
        return True
    if monitor == "val_loss":
        return current < best - min_delta
    return current > best + min_delta


def run_training(config: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    """Run one seed and return its summary dictionary.

    The training function never constructs or reads the official test set.
    Test evaluation has a separate command that requires an explicit final-use flag.
    """
    validate_config(config)
    seed = int(config["train"]["seed"])
    seed_everything(seed, bool(config["train"]["deterministic"]))

    output_root = Path(config["experiment"]["output_dir"])
    if not output_root.is_absolute():
        output_root = Path(project_root) / output_root
    seed_directory = "seed_{}".format(seed)
    attempt = config["experiment"].get("attempt")
    if attempt:
        seed_directory += "_" + safe_name(str(attempt))
    run_directory = output_root / safe_name(config["experiment"]["id"]) / seed_directory
    run_directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        # exist_ok=False atomically reserves this seed/attempt. Two notebooks
        # cannot accidentally write the same CSV and checkpoint concurrently.
        run_directory.mkdir(exist_ok=False)
    except FileExistsError:
        raise FileExistsError(
            "Run directory already exists; choose a new attempt: {}".format(run_directory)
        )

    started_at = datetime.now(timezone.utc).isoformat()
    write_yaml(run_directory / "config.resolved.yaml", config)
    environment = collect_environment()
    write_json(run_directory / "environment.json", environment)
    write_json(run_directory / "status.json", {"state": "running", "started_at": started_at})

    step_writer = None
    epoch_writer = None
    try:
        device = choose_device(config["train"]["device"])
        data = build_dataloaders(config, include_test=False)
        model = build_model(config["model"], data.num_classes).to(device)
        criterion = build_criterion(config).to(device)
        optimizer = build_optimizer(model, config)
        lr_schedule = LearningRateSchedule(optimizer, config, len(data.train))
        amp_enabled = bool(config["train"]["amp"]) and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        step_writer = CsvWriter(run_directory / "metrics_step.csv", STEP_COLUMNS)
        epoch_writer = CsvWriter(run_directory / "metrics_epoch.csv", EPOCH_COLUMNS)
        checkpoint_path = run_directory / "checkpoints" / "best.pt"

        global_step = 0
        best_val_accuracy = -1.0
        best_epoch = 0
        best_early_value = None
        epochs_without_improvement = 0
        stopped_early = False
        history = []
        convergence_epoch = None
        run_start = time.perf_counter()

        for epoch in range(1, int(config["train"]["epochs"]) + 1):
            _sync_cuda(device)
            epoch_start = time.perf_counter()
            train_loss, train_accuracy, global_step, current_lr = train_one_epoch(
                model=model,
                loader=data.train,
                criterion=criterion,
                optimizer=optimizer,
                lr_schedule=lr_schedule,
                scaler=scaler,
                device=device,
                epoch=epoch,
                global_step=global_step,
                step_writer=step_writer,
                log_every_steps=int(config["train"]["log_every_steps"]),
                max_batches=config["debug"]["max_train_batches"],
                run_start_time=run_start,
            )
            val_loss, val_accuracy = evaluate_model(
                model,
                data.val,
                criterion,
                device,
                max_batches=config["debug"]["max_val_batches"],
                amp_enabled=amp_enabled,
            )
            _sync_cuda(device)
            epoch_seconds = time.perf_counter() - epoch_start
            row = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                # With augmentation this online train metric sees harder/random
                # inputs, so it is not used for the final generalization gap.
                "online_train_val_gap": train_accuracy - val_accuracy,
                "learning_rate": current_lr,
                "epoch_seconds": epoch_seconds,
            }
            epoch_writer.write(row)
            history.append(row)

            if convergence_epoch is None and val_accuracy >= float(
                config["train"]["convergence_accuracy"]
            ):
                convergence_epoch = epoch

            # Checkpoint selection is fixed to validation accuracy for all runs.
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_epoch = epoch
                _save_checkpoint(
                    checkpoint_path,
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_accuracy": val_accuracy,
                        "config": config,
                        "split_hash": data.split_hash,
                    },
                )

            early = config["regularization"]["early_stopping"]
            monitored_value = val_loss if early["monitor"] == "val_loss" else val_accuracy
            if _early_stopping_improved(
                early["monitor"], monitored_value, best_early_value, float(early["min_delta"])
            ):
                best_early_value = monitored_value
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if early["enabled"] and epochs_without_improvement >= int(early["patience"]):
                stopped_early = True
                break

        _sync_cuda(device)
        training_seconds = time.perf_counter() - run_start
        clean_eval_start = time.perf_counter()
        # Evaluate the final model on the clean training transform once. This
        # makes train-vs-validation gap comparable even in augmentation runs.
        clean_train_loss, clean_train_accuracy = evaluate_model(
            model,
            data.train_eval,
            criterion,
            device,
            max_batches=config["debug"]["max_val_batches"],
            amp_enabled=amp_enabled,
        )
        _sync_cuda(device)
        clean_eval_seconds = time.perf_counter() - clean_eval_start
        total_seconds = time.perf_counter() - run_start
        final = history[-1]
        summary = {
            "state": "completed",
            "experiment_id": config["experiment"]["id"],
            "experiment_label": config["experiment"]["label"],
            "attempt": attempt,
            "comparison_groups": config["experiment"]["comparison_groups"],
            "protocol_version": config["experiment"]["protocol_version"],
            "semantic_fingerprint": semantic_fingerprint(config),
            "seed": seed,
            "split_hash": data.split_hash,
            "debug_active": data.debug_active,
            "dataset": config["data"]["dataset"],
            "optimizer": config["optimizer"]["name"],
            "learning_rate": config["optimizer"]["lr"],
            "weight_decay": config["optimizer"]["weight_decay"],
            "scheduler": config["scheduler"]["name"],
            "warmup_epochs": config["scheduler"]["warmup_epochs"],
            "normalization": config["model"]["normalization"],
            "batch_size": config["data"]["batch_size"],
            "dropout": config["model"]["dropout"],
            "augmentation": config["data"]["augmentation"],
            "early_stopping": config["regularization"]["early_stopping"]["enabled"],
            "best_val_accuracy": best_val_accuracy,
            "best_epoch": best_epoch,
            "final_train_loss": final["train_loss"],
            "final_train_accuracy": final["train_accuracy"],
            "clean_train_loss": clean_train_loss,
            "clean_train_accuracy": clean_train_accuracy,
            "final_val_loss": final["val_loss"],
            "final_val_accuracy": final["val_accuracy"],
            "final_generalization_gap": clean_train_accuracy - final["val_accuracy"],
            "online_final_train_val_gap": final["online_train_val_gap"],
            "mean_val_accuracy_over_epochs": sum(item["val_accuracy"] for item in history)
            / len(history),
            "convergence_epoch": convergence_epoch,
            "epochs_completed": len(history),
            "stopped_early": stopped_early,
            "training_seconds": training_seconds,
            "clean_train_eval_seconds": clean_eval_seconds,
            "total_seconds": total_seconds,
            "trainable_parameters": count_trainable_parameters(model),
            "device": str(device),
            "gpu_name": environment["gpu_name"],
            "python_version": environment["python"],
            "torch_version": environment["torch"],
            "torchvision_version": environment["torchvision"],
            "git_commit": environment["git_commit"],
            "git_status": environment["git_status"],
            "checkpoint": _portable_path(checkpoint_path, project_root),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(run_directory / "summary.json", summary)
        write_json(
            run_directory / "status.json",
            {
                "state": "completed",
                "started_at": started_at,
                "completed_at": summary["completed_at"],
            },
        )
        return summary
    except BaseException as error:
        write_json(
            run_directory / "status.json",
            {
                "state": "failed",
                "started_at": started_at,
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    finally:
        if step_writer is not None:
            step_writer.close()
        if epoch_writer is not None:
            epoch_writer.close()
