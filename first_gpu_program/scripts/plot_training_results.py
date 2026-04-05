#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training results from CSV")
    parser.add_argument("--input", type=Path, required=True, help="Path to results CSV")
    parser.add_argument("--output", type=Path, default=None, help="Path to save the figure (default: input name + .png)")
    parser.add_argument("--show", action="store_true", help="Show the plot window")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in CSV: {path}")
    return rows


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def as_int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value == "":
        return default
    return int(float(value))


def main() -> int:
    args = parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required. Install it with: pip install matplotlib"
        ) from exc

    rows = load_rows(args.input)
    epochs = [as_int(row, "epoch") for row in rows]
    train_loss = [as_float(row, "train_loss") for row in rows]
    eval_cost = [as_float(row, "eval_cost") for row in rows]
    precision = [as_float(row, "precision") for row in rows]
    recall = [as_float(row, "recall") for row in rows]
    specificity = [as_float(row, "specificity") for row in rows]
    f1 = [as_float(row, "f1") for row in rows]
    tp = [as_int(row, "tp") for row in rows]
    fp = [as_int(row, "fp") for row in rows]
    tn = [as_int(row, "tn") for row in rows]
    fn = [as_int(row, "fn") for row in rows]
    phases = [row.get("phase", "") for row in rows]

    epoch_rows = [index for index, phase in enumerate(phases) if phase == "epoch"]
    special_rows = [index for index, phase in enumerate(phases) if phase != "epoch"]
    if not epoch_rows:
        epoch_rows = list(range(len(rows)))

    output_path = args.output or args.input.with_suffix(".png")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_loss = axes[0, 0]
    ax_metrics = axes[0, 1]
    ax_conf = axes[1, 0]
    ax_table = axes[1, 1]

    ax_loss.plot([epochs[i] for i in epoch_rows], [train_loss[i] for i in epoch_rows], marker="o", label="train_loss")
    ax_loss.plot([epochs[i] for i in epoch_rows], [eval_cost[i] for i in epoch_rows], marker="o", label="eval_cost")
    if special_rows:
        ax_loss.scatter([epochs[i] for i in special_rows], [eval_cost[i] for i in special_rows], marker="x", color="black", label="special rows")
    ax_loss.set_title("Loss / Cost")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Value")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    ax_metrics.plot([epochs[i] for i in epoch_rows], [precision[i] for i in epoch_rows], marker="o", label="precision")
    ax_metrics.plot([epochs[i] for i in epoch_rows], [recall[i] for i in epoch_rows], marker="o", label="recall")
    ax_metrics.plot([epochs[i] for i in epoch_rows], [specificity[i] for i in epoch_rows], marker="o", label="specificity")
    ax_metrics.plot([epochs[i] for i in epoch_rows], [f1[i] for i in epoch_rows], marker="o", label="f1")
    ax_metrics.set_title("Classification Metrics")
    ax_metrics.set_xlabel("Epoch")
    ax_metrics.set_ylabel("Score")
    ax_metrics.set_ylim(0.0, 1.05)
    ax_metrics.grid(True, alpha=0.3)
    ax_metrics.legend()

    last_index = epoch_rows[-1]
    conf_labels = ["TP", "FP", "TN", "FN"]
    conf_values = [tp[last_index], fp[last_index], tn[last_index], fn[last_index]]
    ax_conf.bar(conf_labels, conf_values, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"])
    ax_conf.set_title(f"Confusion Matrix Counts ({phases[last_index]} / epoch {epochs[last_index]})")
    ax_conf.set_ylabel("Count")
    for idx, value in enumerate(conf_values):
        ax_conf.text(idx, value, str(value), ha="center", va="bottom")

    ax_table.axis("off")
    latest = rows[last_index]
    table_lines = [
        ["phase", latest.get("phase", "")],
        ["epoch", latest.get("epoch", "")],
        ["train_loss", latest.get("train_loss", "")],
        ["eval_cost", latest.get("eval_cost", "")],
        ["accuracy", latest.get("accuracy", "")],
        ["precision", latest.get("precision", "")],
        ["recall", latest.get("recall", "")],
        ["specificity", latest.get("specificity", "")],
        ["f1", latest.get("f1", "")],
    ]
    table_text = "\n".join(f"{name}: {value}" for name, value in table_lines)
    ax_table.text(0.0, 1.0, table_text, va="top", family="monospace")
    ax_table.set_title("Latest Row Summary")

    fig.suptitle(f"Training Results: {args.input.name}")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    print(f"Saved figure: {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
