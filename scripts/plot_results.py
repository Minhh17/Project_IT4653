#!/usr/bin/env python3
"""Generate the report figures from run logs; no numbers are copied by hand."""

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "it4653_matplotlib"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sns.set_theme(style="whitegrid")


def _load_runs(runs_dir):
    runs = []
    for path in sorted(runs_dir.glob("**/summary.json")):
        with path.open("r", encoding="utf-8") as file:
            summary = json.load(file)
        if summary.get("state") == "completed":
            summary["run_directory"] = path.parent
            runs.append(summary)
    return runs


def _run_manifest(runs):
    return sorted(
        "{}|{}|{}|{}".format(
            run["experiment_id"],
            run["seed"],
            run.get("attempt"),
            run["semantic_fingerprint"],
        )
        for run in runs
    )


def _in_group(run, group):
    return group in run["comparison_groups"]


def _group_label(run, group):
    """Give the shared anchor a short, meaningful label in each figure."""
    if group == "optimizer":
        names = {
            "sgd": "SGD",
            "sgd_momentum": "SGD + momentum",
            "nesterov": "Nesterov",
            "rmsprop": "RMSProp (momentum 0.9)",
            "adam": "Adam",
            "adamw": "AdamW",
        }
        return names[run["optimizer"]]
    if group == "normalization":
        names = {"batch": "BatchNorm", "layer": "LayerNorm", "group": "GroupNorm"}
        return "{}, batch {}".format(names[run["normalization"]], run["batch_size"])
    if group == "schedule":
        names = {"constant": "Constant", "step": "Step decay", "cosine": "Cosine"}
        suffix = " + warm-up" if int(run["warmup_epochs"]) > 0 else ""
        return names[run["scheduler"]] + suffix
    if group == "regularization" and run["experiment_id"] == "anchor_sgdm_wd_bn_b128":
        return "Weight decay"
    return run["experiment_label"]


def _save(output_dir, filename):
    plt.tight_layout()
    path = output_dir / filename
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print("Wrote " + str(path))


def _epoch_frame(runs, group):
    frames = []
    for run in runs:
        if not _in_group(run, group):
            continue
        frame = pd.read_csv(run["run_directory"] / "metrics_epoch.csv")
        frame["experiment"] = _group_label(run, group)
        frame["seed"] = run["seed"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _summary_frame(runs, group):
    values = []
    for run in runs:
        if _in_group(run, group):
            value = {key: item for key, item in run.items() if key != "run_directory"}
            value["plot_label"] = _group_label(run, group)
            values.append(value)
    return pd.DataFrame(values)


def _pointplot_with_raw(data, x, y, order=None):
    """Show mean±SD and the two underlying seed observations together."""
    sns.pointplot(data=data, x=x, y=y, order=order, errorbar="sd", capsize=0.15)
    sns.stripplot(data=data, x=x, y=y, order=order, color="black", size=4, jitter=0.06)


def plot_optimizer(runs, output_dir):
    epochs = _epoch_frame(runs, "optimizer")
    summary = _summary_frame(runs, "optimizer")
    if epochs.empty:
        return
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    sns.lineplot(
        data=epochs,
        x="epoch",
        y="train_loss",
        hue="experiment",
        errorbar="sd",
        ax=axes[0],
        legend=False,
    )
    axes[0].set_title("Train loss")
    sns.lineplot(
        data=epochs,
        x="epoch",
        y="val_loss",
        hue="experiment",
        errorbar="sd",
        ax=axes[1],
    )
    axes[1].set_title("Validation loss")
    figure.suptitle("Đường hội tụ của 6 optimizer")
    _save(output_dir, "01_optimizer_train_val_loss.png")

    order = summary.groupby("plot_label")["best_val_accuracy"].mean().sort_values().index
    _pointplot_with_raw(summary, "best_val_accuracy", "plot_label", order=order)
    plt.xlabel("Best validation accuracy (mean ± std)")
    plt.ylabel("")
    _save(output_dir, "02_optimizer_accuracy.png")


def plot_normalization(runs, output_dir):
    summary = _summary_frame(runs, "normalization")
    if summary.empty:
        return
    sns.lineplot(
        data=summary,
        x="batch_size",
        y="best_val_accuracy",
        hue="normalization",
        units="seed",
        estimator=None,
        marker="o",
        alpha=0.3,
        legend=False,
    )
    sns.pointplot(
        data=summary,
        x="batch_size",
        y="best_val_accuracy",
        hue="normalization",
        errorbar="sd",
        dodge=0.15,
        capsize=0.1,
        # batch_size is numeric. Without native_scale, pointplot uses x=0,1,2
        # while the raw seed lines above use x=8,32,128, so the two layers do
        # not share an axis and the raw observations disappear from the plot.
        native_scale=True,
    )
    plt.ylabel("Best validation accuracy")
    plt.title("Normalization theo batch size")
    _save(output_dir, "03_normalization_batch_size.png")


def plot_schedules(runs, output_dir):
    step_frames = []
    for run in runs:
        if not _in_group(run, "schedule"):
            continue
        frame = pd.read_csv(run["run_directory"] / "metrics_step.csv")
        frame["loss_smoothed"] = frame["loss"].rolling(50, min_periods=1).mean()
        frame["experiment"] = _group_label(run, "schedule")
        frame["seed"] = run["seed"]
        step_frames.append(frame)
    if not step_frames:
        return
    steps = pd.concat(step_frames, ignore_index=True)
    sns.lineplot(
        data=steps,
        x="global_step",
        y="loss_smoothed",
        hue="experiment",
        units="seed",
        estimator=None,
        alpha=0.65,
    )
    plt.ylabel("Smoothed train loss (window=50)")
    plt.title("Loss theo optimizer step của các LR schedule")
    _save(output_dir, "04_schedule_loss_by_step.png")

    summary = _summary_frame(runs, "schedule")
    _pointplot_with_raw(summary, "best_val_accuracy", "plot_label")
    plt.xlabel("Best validation accuracy (mean ± std)")
    plt.ylabel("")
    _save(output_dir, "05_schedule_accuracy.png")


def plot_regularization(runs, output_dir):
    summary = _summary_frame(runs, "regularization")
    if summary.empty:
        return
    order = summary.groupby("plot_label")["best_val_accuracy"].mean().sort_values().index
    _pointplot_with_raw(summary, "best_val_accuracy", "plot_label", order=order)
    plt.xlabel("Best validation accuracy (mean ± std)")
    plt.ylabel("")
    _save(output_dir, "06_regularization_accuracy.png")

    gap_order = summary.groupby("plot_label")["final_generalization_gap"].mean().sort_values().index
    _pointplot_with_raw(summary, "final_generalization_gap", "plot_label", order=gap_order)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Final train accuracy - validation accuracy")
    plt.ylabel("")
    _save(output_dir, "07_regularization_generalization_gap.png")

    combination = summary[
        summary["experiment_id"].isin(
            ["reg_none", "anchor_sgdm_wd_bn_b128", "reg_wd_augmentation", "reg_combined"]
        )
    ]
    if not combination.empty:
        _pointplot_with_raw(combination, "best_val_accuracy", "plot_label")
        plt.xlabel("Best validation accuracy (mean ± std)")
        plt.ylabel("")
        _save(output_dir, "08_regularization_individual_vs_combined.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/figures")
    parser.add_argument("--audit", type=Path, default=PROJECT_ROOT / "results/audit.json")
    parser.add_argument(
        "--allow-partial", action="store_true", help="Allow exploratory plots without a clean audit"
    )
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    runs = _load_runs(arguments.runs_dir)
    if not runs:
        raise SystemExit("No completed runs found")
    if not arguments.allow_partial:
        if not arguments.audit.exists():
            raise SystemExit("Run aggregate_results.py --strict before final plotting")
        with arguments.audit.open("r", encoding="utf-8") as file:
            audit = json.load(file)
        if audit.get("issues") or audit.get("run_manifest") != _run_manifest(runs):
            raise SystemExit("Audit is not clean or is stale; aggregate again before plotting")
    plot_optimizer(runs, arguments.output_dir)
    plot_normalization(runs, arguments.output_dir)
    plot_schedules(runs, arguments.output_dir)
    plot_regularization(runs, arguments.output_dir)


if __name__ == "__main__":
    main()
