#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List


@dataclass
class CommandResult:
    stage: str
    command: List[str]
    return_code: int
    elapsed_sec: float
    log_file: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run organized LSTM train-val-test pipeline")
    parser.add_argument("--exe", type=Path, default=Path("build/Debug/first_gpu_program.exe"))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/LSTM/processed"))
    parser.add_argument("--output-root", type=Path, default=Path("datasets/LSTM/processed/experiments"))
    parser.add_argument("--run-name", type=str, default="")

    parser.add_argument("--source-train-csv", type=Path, default=Path("datasets/LSTM/processed/train.csv"))
    parser.add_argument("--compact-total-rows", type=int, default=1400)
    parser.add_argument("--compact-seed", type=int, default=42)
    parser.add_argument("--compact-train-ratio", type=float, default=0.7)
    parser.add_argument("--compact-val-ratio", type=float, default=0.15)
    parser.add_argument("--compact-test-ratio", type=float, default=0.15)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--lstm-hidden", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def run_and_log(stage: str, command: List[str], log_path: Path, cwd: Path) -> CommandResult:
    import time

    start = time.perf_counter()
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=cwd)
    elapsed = time.perf_counter() - start
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Stage '{stage}' failed (code={completed.returncode}). See log: {log_path}")
    return CommandResult(stage=stage, command=command, return_code=completed.returncode, elapsed_sec=elapsed, log_file=str(log_path))


def build_balanced_rows(source_csv: Path, target_rows: int, seed: int) -> tuple[list[str], list[list[str]]]:
    rng = random.Random(seed)
    k_pos = target_rows // 2
    k_neg = target_rows - k_pos
    pos_reservoir: list[list[str]] = []
    neg_reservoir: list[list[str]] = []
    all_reservoir: list[list[str]] = []
    seen_pos = 0
    seen_neg = 0
    seen_all = 0

    with source_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"CSV is empty: {source_csv}")

        for row in reader:
            if not row:
                continue

            seen_all += 1
            if len(all_reservoir) < target_rows:
                all_reservoir.append(row)
            else:
                j_all = rng.randint(0, seen_all - 1)
                if j_all < target_rows:
                    all_reservoir[j_all] = row

            label = row[-1].strip()
            if label == "1":
                seen_pos += 1
                if len(pos_reservoir) < k_pos:
                    pos_reservoir.append(row)
                else:
                    j_pos = rng.randint(0, seen_pos - 1)
                    if j_pos < k_pos:
                        pos_reservoir[j_pos] = row
            elif label == "0":
                seen_neg += 1
                if len(neg_reservoir) < k_neg:
                    neg_reservoir.append(row)
                else:
                    j_neg = rng.randint(0, seen_neg - 1)
                    if j_neg < k_neg:
                        neg_reservoir[j_neg] = row

    merged = pos_reservoir + neg_reservoir
    if len(merged) < target_rows:
        need = target_rows - len(merged)
        merged.extend(all_reservoir[:need])

    rng.shuffle(merged)
    return header, merged[:target_rows]


def stratified_split_rows(rows: list[list[str]], train_ratio: float, val_ratio: float, seed: int):
    ratio_sum = train_ratio + val_ratio
    if ratio_sum <= 0.0 or ratio_sum >= 1.0:
        raise ValueError("compact-train-ratio + compact-val-ratio must be in (0, 1)")

    grouped: dict[str, list[list[str]]] = {"0": [], "1": [], "other": []}
    for row in rows:
        label = row[-1].strip()
        if label == "0":
            grouped["0"].append(row)
        elif label == "1":
            grouped["1"].append(row)
        else:
            grouped["other"].append(row)

    rng = random.Random(seed)
    train_rows: list[list[str]] = []
    val_rows: list[list[str]] = []
    test_rows: list[list[str]] = []

    for cls_rows in grouped.values():
        rng.shuffle(cls_rows)
        total = len(cls_rows)
        train_count = int(round(total * train_ratio))
        val_count = int(round(total * val_ratio))
        if train_count + val_count > total:
            val_count = max(0, total - train_count)
        train_rows.extend(cls_rows[:train_count])
        val_rows.extend(cls_rows[train_count:train_count + val_count])
        test_rows.extend(cls_rows[train_count + val_count:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)
    return train_rows, val_rows, test_rows


def write_rows(csv_path: Path, header: list[str], rows: list[list[str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def count_labels(rows: list[list[str]]) -> dict[str, int]:
    pos = 0
    neg = 0
    for row in rows:
        label = row[-1].strip()
        if label == "1":
            pos += 1
        elif label == "0":
            neg += 1
    return {"rows": len(rows), "positives": pos, "negatives": neg}


def main() -> int:
    args = parse_args()
    if not args.exe.exists():
        raise FileNotFoundError(f"Executable not found: {args.exe}")
    if not args.source_train_csv.exists():
        raise FileNotFoundError(f"Source train CSV not found: {args.source_train_csv}")
    if args.compact_total_rows <= 0:
        raise ValueError("compact-total-rows must be positive")
    ratio_sum = args.compact_train_ratio + args.compact_val_ratio + args.compact_test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("compact-train-ratio + compact-val-ratio + compact-test-ratio must equal 1.0")

    data_dir = args.data_dir
    for p in (data_dir / "train.csv", data_dir / "val.csv", data_dir / "test.csv"):
        if not p.exists():
            raise FileNotFoundError(f"Required dataset split missing: {p}")

    run_name = args.run_name.strip() or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = args.output_root / run_name
    dirs = {
        "models": run_dir / "models",
        "results": run_dir / "results",
        "figures": run_dir / "figures",
        "logs": run_dir / "logs",
        "configs": run_dir / "configs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    compact_train_csv = run_dir / "results" / "compact_train.csv"
    compact_val_csv = run_dir / "results" / "compact_val.csv"
    compact_test_csv = run_dir / "results" / "compact_test.csv"
    compact_manifest = run_dir / "configs" / "compact_data_manifest.json"
    model_file = run_dir / "models" / "lstm_model.bin"

    train_results = run_dir / "results" / "train_metrics.csv"
    train_pr = run_dir / "results" / "train_pr.csv"
    val_results = run_dir / "results" / "val_metrics.csv"
    val_pr = run_dir / "results" / "val_pr.csv"
    test_results = run_dir / "results" / "test_metrics.csv"
    test_pr = run_dir / "results" / "test_pr.csv"

    summary: List[CommandResult] = []
    root = Path.cwd()

    header, compact_rows = build_balanced_rows(args.source_train_csv, args.compact_total_rows, args.compact_seed)
    compact_train_rows, compact_val_rows, compact_test_rows = stratified_split_rows(
        compact_rows,
        args.compact_train_ratio,
        args.compact_val_ratio,
        args.compact_seed,
    )
    write_rows(compact_train_csv, header, compact_train_rows)
    write_rows(compact_val_csv, header, compact_val_rows)
    write_rows(compact_test_csv, header, compact_test_rows)

    manifest = {
        "source_train_csv": str(args.source_train_csv),
        "compact_total_rows": args.compact_total_rows,
        "split_ratios": {
            "train": args.compact_train_ratio,
            "val": args.compact_val_ratio,
            "test": args.compact_test_ratio,
        },
        "seed": args.compact_seed,
        "splits": {
            "train": count_labels(compact_train_rows),
            "val": count_labels(compact_val_rows),
            "test": count_labels(compact_test_rows),
        },
        "paths": {
            "train": str(compact_train_csv),
            "val": str(compact_val_csv),
            "test": str(compact_test_csv),
        },
    }
    compact_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    train_cmd = [
        str(args.exe),
        "--model",
        "lstm",
        "--dataset",
        str(compact_train_csv),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--print-every",
        "1",
        "--seq-len",
        str(args.seq_len),
        "--lstm-hidden",
        str(args.lstm_hidden),
        "--lr",
        str(args.lr),
        "--optimizer",
        str(args.optimizer),
        "--backend",
        "cpu",
        "--save-model",
        str(model_file),
        "--results-csv",
        str(train_results),
        "--pr-csv",
        str(train_pr),
    ]
    summary.append(run_and_log("train", train_cmd, dirs["logs"] / "02_train.log", root))

    val_cmd = [
        str(args.exe),
        "--model",
        "lstm",
        "--dataset",
        str(compact_val_csv),
        "--eval-only",
        "--seq-len",
        str(args.seq_len),
        "--lstm-hidden",
        str(args.lstm_hidden),
        "--lr",
        str(args.lr),
        "--optimizer",
        str(args.optimizer),
        "--backend",
        "cpu",
        "--load-model",
        str(model_file),
        "--results-csv",
        str(val_results),
        "--pr-csv",
        str(val_pr),
    ]
    summary.append(run_and_log("val", val_cmd, dirs["logs"] / "03_val.log", root))

    test_cmd = [
        str(args.exe),
        "--model",
        "lstm",
        "--dataset",
        str(compact_test_csv),
        "--eval-only",
        "--seq-len",
        str(args.seq_len),
        "--lstm-hidden",
        str(args.lstm_hidden),
        "--lr",
        str(args.lr),
        "--optimizer",
        str(args.optimizer),
        "--backend",
        "cpu",
        "--load-model",
        str(model_file),
        "--results-csv",
        str(test_results),
        "--pr-csv",
        str(test_pr),
    ]
    summary.append(run_and_log("test", test_cmd, dirs["logs"] / "04_test.log", root))

    if not args.skip_plots:
        plot_jobs = [
            ("plot_train_metrics", train_results, run_dir / "figures" / "train_metrics.png", "scripts/plot_training_results.py"),
            ("plot_train_pr", train_pr, run_dir / "figures" / "train_pr.png", "scripts/plot_pr_curve.py"),
            ("plot_val_metrics", val_results, run_dir / "figures" / "val_metrics.png", "scripts/plot_training_results.py"),
            ("plot_val_pr", val_pr, run_dir / "figures" / "val_pr.png", "scripts/plot_pr_curve.py"),
            ("plot_test_metrics", test_results, run_dir / "figures" / "test_metrics.png", "scripts/plot_training_results.py"),
            ("plot_test_pr", test_pr, run_dir / "figures" / "test_pr.png", "scripts/plot_pr_curve.py"),
        ]

        for stage, input_csv, output_png, script_path in plot_jobs:
            cmd = [str(args.python), script_path, "--input", str(input_csv), "--output", str(output_png)]
            summary.append(run_and_log(stage, cmd, dirs["logs"] / f"{stage}.log", root))

    summary_json = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "params": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "lstm_hidden": args.lstm_hidden,
            "lr": args.lr,
            "optimizer": args.optimizer,
            "compact_total_rows": args.compact_total_rows,
            "compact_seed": args.compact_seed,
        },
        "artifacts": {
            "model": str(model_file),
            "compact_data_manifest": str(compact_manifest),
            "results": {
                "train": str(train_results),
                "val": str(val_results),
                "test": str(test_results),
            },
            "pr": {
                "train": str(train_pr),
                "val": str(val_pr),
                "test": str(test_pr),
            },
            "compact_splits": {
                "train": str(compact_train_csv),
                "val": str(compact_val_csv),
                "test": str(compact_test_csv),
            },
            "figures": {
                "train_metrics": str(run_dir / "figures" / "train_metrics.png"),
                "train_pr": str(run_dir / "figures" / "train_pr.png"),
                "val_metrics": str(run_dir / "figures" / "val_metrics.png"),
                "val_pr": str(run_dir / "figures" / "val_pr.png"),
                "test_metrics": str(run_dir / "figures" / "test_metrics.png"),
                "test_pr": str(run_dir / "figures" / "test_pr.png"),
            },
            "logs_dir": str(dirs["logs"]),
        },
        "stages": [asdict(item) for item in summary],
    }

    summary_path = run_dir / "configs" / "run_summary.json"
    summary_path.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    print(f"run_dir={run_dir}")
    print(f"model={model_file}")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
