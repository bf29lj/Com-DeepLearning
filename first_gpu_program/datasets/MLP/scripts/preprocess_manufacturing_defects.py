#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path


DEFAULT_LEAKAGE_COLUMNS = {"DefectRate", "QualityScore"}
DEFAULT_LABEL_COLUMN = "DefectStatus"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess manufacturing defect dataset")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("datasets/MLP/raw/manufacturing_defect_dataset.csv"),
        help="Path to the raw CSV dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/MLP/processed"),
        help="Directory to write processed splits",
    )
    parser.add_argument(
        "--label-column",
        default=DEFAULT_LABEL_COLUMN,
        help="Target column name",
    )
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=sorted(DEFAULT_LEAKAGE_COLUMNS),
        help="Columns to remove before training",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio")
    return parser.parse_args()


def load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")
        rows = list(reader)
        return reader.fieldnames, rows


def to_float(value: str, column: str, row_index: int) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value in row {row_index + 1}, column '{column}': {value}") from exc


def build_dataset(
    headers: list[str],
    rows: list[dict[str, str]],
    label_column: str,
    drop_columns: set[str],
) -> tuple[list[str], list[dict[str, float]], list[int]]:
    feature_columns = [column for column in headers if column not in drop_columns | {label_column}]
    processed_rows: list[dict[str, float]] = []
    labels: list[int] = []

    for row_index, row in enumerate(rows):
        feature_row: dict[str, float] = {}
        for column in feature_columns:
            feature_row[column] = to_float(row[column], column, row_index)
        processed_rows.append(feature_row)
        labels.append(int(float(row[label_column])))

    return feature_columns, processed_rows, labels


def stratified_split_indices(labels: list[int], train_ratio: float, val_ratio: float, seed: int):
    if not math.isclose(train_ratio + val_ratio + (1.0 - train_ratio - val_ratio), 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    grouped: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[label].append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for label, indices in grouped.items():
        rng.shuffle(indices)
        total = len(indices)
        train_count = int(round(total * train_ratio))
        val_count = int(round(total * val_ratio))
        if train_count + val_count > total:
            overflow = train_count + val_count - total
            val_count = max(0, val_count - overflow)
        test_count = total - train_count - val_count

        train_indices.extend(indices[:train_count])
        val_indices.extend(indices[train_count:train_count + val_count])
        test_indices.extend(indices[train_count + val_count:train_count + val_count + test_count])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    return train_indices, val_indices, test_indices


def compute_scaler(feature_columns: list[str], rows: list[dict[str, float]], indices: list[int]):
    means: dict[str, float] = {}
    stds: dict[str, float] = {}

    for column in feature_columns:
        values = [rows[index][column] for index in indices]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(variance) if variance > 0 else 1.0
        means[column] = mean
        stds[column] = std

    return means, stds


def normalize_row(row: dict[str, float], means: dict[str, float], stds: dict[str, float]) -> dict[str, float]:
    return {
        column: (value - means[column]) / stds[column]
        for column, value in row.items()
    }


def write_split(
    output_path: Path,
    feature_columns: list[str],
    rows: list[dict[str, float]],
    labels: list[int],
    indices: list[int],
    means: dict[str, float] | None = None,
    stds: dict[str, float] | None = None,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(feature_columns + [DEFAULT_LABEL_COLUMN])
        for index in indices:
            row = rows[index]
            if means is not None and stds is not None:
                row = normalize_row(row, means, stds)
            writer.writerow([row[column] for column in feature_columns] + [labels[index]])


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input dataset not found: {args.input}")

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("train-ratio + val-ratio + test-ratio must equal 1.0")

    headers, raw_rows = load_rows(args.input)
    feature_columns, rows, labels = build_dataset(
        headers,
        raw_rows,
        args.label_column,
        set(args.drop_columns),
    )

    train_indices, val_indices, test_indices = stratified_split_indices(
        labels,
        args.train_ratio,
        args.val_ratio,
        args.seed,
    )

    means, stds = compute_scaler(feature_columns, rows, train_indices)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_split(args.output_dir / "train.csv", feature_columns, rows, labels, train_indices, means, stds)
    write_split(args.output_dir / "val.csv", feature_columns, rows, labels, val_indices, means, stds)
    write_split(args.output_dir / "test.csv", feature_columns, rows, labels, test_indices, means, stds)

    scaler_payload = {
        "label_column": args.label_column,
        "dropped_columns": list(args.drop_columns),
        "feature_columns": feature_columns,
        "means": means,
        "stds": stds,
        "splits": {
            "train": len(train_indices),
            "val": len(val_indices),
            "test": len(test_indices),
        },
    }
    with (args.output_dir / "scaler.json").open("w", encoding="utf-8") as handle:
        json.dump(scaler_payload, handle, indent=2)

    print(f"Loaded rows: {len(rows)}")
    print(f"Features kept: {len(feature_columns)}")
    print(f"Train/Val/Test: {len(train_indices)}/{len(val_indices)}/{len(test_indices)}")
    print(f"Processed data written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())