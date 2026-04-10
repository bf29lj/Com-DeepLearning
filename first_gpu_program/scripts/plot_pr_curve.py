#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from math import isfinite
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot precision-recall curve from CSV")
    parser.add_argument("--input", type=Path, required=True, help="Path to PR CSV")
    parser.add_argument("--output", type=Path, default=None, help="Path to save the figure (default: input name + _pr.png)")
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


def compute_auprc(recall: list[float], precision: list[float]) -> float:
    points = sorted(
        ((r, p) for r, p in zip(recall, precision) if isfinite(r) and isfinite(p)),
        key=lambda item: item[0],
    )
    if len(points) < 2:
        return 0.0

    area = 0.0
    prev_recall, prev_precision = points[0]
    for current_recall, current_precision in points[1:]:
        delta_recall = current_recall - prev_recall
        if delta_recall > 0.0:
            area += delta_recall * (prev_precision + current_precision) * 0.5
        prev_recall = current_recall
        prev_precision = current_precision
    return max(0.0, min(1.0, area))


def main() -> int:
    args = parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required. Install it with: pip install matplotlib"
        ) from exc

    rows = load_rows(args.input)
    thresholds = [as_float(row, "threshold") for row in rows]
    precision = [as_float(row, "precision") for row in rows]
    recall = [as_float(row, "recall") for row in rows]
    f1 = [as_float(row, "f1") for row in rows]
    auprc = compute_auprc(recall, precision)

    output_path = args.output or args.input.with_name(f"{args.input.stem}_pr.png")

    best_index = max(range(len(rows)), key=lambda index: f1[index])

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        recall,
        precision,
        c=thresholds,
        cmap="viridis",
        s=44,
        edgecolors="none",
    )
    ax.plot(recall, precision, alpha=0.4, linewidth=1.5)
    ax.scatter(
        [recall[best_index]],
        [precision[best_index]],
        color="red",
        s=90,
        label=f"best F1={f1[best_index]:.4f} @ threshold={thresholds[best_index]:.3f}",
        zorder=3,
    )
    ax.text(
        0.98,
        0.02,
        f"AUPRC = {auprc:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )
    ax.set_title(f"Precision-Recall Curve (AUPRC={auprc:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Threshold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved figure: {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
