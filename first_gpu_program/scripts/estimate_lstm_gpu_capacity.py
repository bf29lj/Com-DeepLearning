#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class BenchmarkPoint:
    rows: int
    seq_samples: int
    train_ms: int
    eval_ms: int
    total_ms: int
    timed_out: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate how much LSTM data the current GPU model can handle within a timeout budget"
    )
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=Path("datasets/LSTM/processed/train.csv"),
        help="Full processed training CSV to sample from",
    )
    parser.add_argument(
        "--exe",
        type=Path,
        default=Path("build/Debug/first_gpu_program.exe"),
        help="Path to the training executable",
    )
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--lstm-hidden", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--backend", type=str, default="gpu", choices=["cpu", "gpu"])
    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        default=[128, 256, 512],
        help="Subset sizes used for calibration",
    )
    parser.add_argument(
        "--max-calibration-sec",
        type=float,
        default=15.0,
        help="Per-calibration-run timeout to keep the estimator responsive",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("datasets/LSTM/processed/lstm_gpu_capacity_report.json"),
    )
    return parser.parse_args()


def count_rows(csv_path: Path) -> int:
    count = 0
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        next(handle, None)
        for _ in handle:
            count += 1
    return count


def write_subset_csv(src: Path, dst: Path, n_rows: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8", newline="") as src_handle, dst.open(
        "w", encoding="utf-8", newline=""
    ) as dst_handle:
        reader = csv.reader(src_handle)
        writer = csv.writer(dst_handle)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"CSV is empty: {src}")
        writer.writerow(header)
        written = 0
        for row in reader:
            writer.writerow(row)
            written += 1
            if written >= n_rows:
                break


def run_benchmark(
    exe: Path,
    dataset: Path,
    seq_len: int,
    lstm_hidden: int,
    batch_size: int,
    optimizer: str,
    lr: float,
    backend: str,
    timeout_sec: float,
) -> BenchmarkPoint:
    command = [
        str(exe),
        "--model",
        "lstm",
        "--backend",
        backend,
        "--dataset",
        str(dataset),
        "--epochs",
        "1",
        "--print-every",
        "1",
        "--seq-len",
        str(seq_len),
        "--lstm-hidden",
        str(lstm_hidden),
        "--batch-size",
        str(batch_size),
        "--optimizer",
        optimizer,
        "--lr",
        str(lr),
        "--timeout-sec",
        str(timeout_sec),
    ]

    start = time.perf_counter()
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        cwd=Path.cwd(),
    )
    elapsed_sec = time.perf_counter() - start
    output = completed.stdout

    train_ms = -1
    eval_ms = -1
    total_ms = -1
    timed_out = "Timeout reached" in output

    for line in output.splitlines():
        if "train_time=" in line:
            marker = line.split("train_time=", 1)[1].split(" ms", 1)[0]
            train_ms = int(marker)
        if "eval_time=" in line:
            marker = line.split("eval_time=", 1)[1].split(" ms", 1)[0]
            eval_ms = int(marker)
        if line.startswith("Total training time: "):
            total_ms = int(line.split(":", 1)[1].split(" ms", 1)[0].strip())

    if completed.returncode != 0:
        raise RuntimeError(
            f"Benchmark failed for {dataset} (code={completed.returncode}).\n{output}"
        )

    rows = count_rows(dataset)
    seq_samples = max(0, rows - seq_len + 1)
    if total_ms < 0:
        total_ms = int(round(elapsed_sec * 1000.0))
    if train_ms < 0:
        train_ms = total_ms

    return BenchmarkPoint(
        rows=rows,
        seq_samples=seq_samples,
        train_ms=train_ms,
        eval_ms=eval_ms,
        total_ms=total_ms,
        timed_out=timed_out,
    )


def fit_linear(points: list[BenchmarkPoint], metric: str) -> tuple[float, float]:
    if metric == "train_ms":
        xs = [float(p.rows) for p in points if p.train_ms > 0]
        ys = [float(p.train_ms) / 1000.0 for p in points if p.train_ms > 0]
    elif metric == "total_ms":
        xs = [float(p.rows) for p in points if p.total_ms > 0]
        ys = [float(p.total_ms) / 1000.0 for p in points if p.total_ms > 0]
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    if len(xs) < 2:
        raise ValueError("Need at least two valid benchmark points")

    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0.0:
        raise ValueError("Benchmark points have no spread")

    slope = numerator / denominator
    intercept = mean_y - slope * mean_x
    return slope, intercept


def estimate_capacity(points: list[BenchmarkPoint], timeout_sec: float, metric: str) -> dict[str, float]:
    slope, intercept = fit_linear(points, metric)
    if slope <= 0.0:
        raise ValueError(f"Non-positive slope from calibration: {slope}")

    estimated_rows = max(0.0, (timeout_sec - intercept) / slope)
    min_rows = min(p.rows for p in points)
    max_rows = max(p.rows for p in points)
    seq_len = None
    return {
        "slope_sec_per_row": slope,
        "intercept_sec": intercept,
        "estimated_rows_for_timeout": estimated_rows,
        "lower_bound_rows": min_rows,
        "upper_bound_rows": max_rows,
    }


def main() -> int:
    args = parse_args()

    if not args.source_csv.exists():
        raise FileNotFoundError(f"Source CSV not found: {args.source_csv}")
    if not args.exe.exists():
        raise FileNotFoundError(f"Executable not found: {args.exe}")
    if args.seq_len <= 0:
        raise ValueError("seq-len must be positive")
    if args.lstm_hidden <= 0:
        raise ValueError("lstm-hidden must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.timeout_sec <= 0.0:
        raise ValueError("timeout-sec must be positive")
    if args.max_calibration_sec <= 0.0:
        raise ValueError("max-calibration-sec must be positive")

    full_rows = count_rows(args.source_csv)
    bench_dir = args.source_csv.parent / "bench_capacity"
    bench_dir.mkdir(parents=True, exist_ok=True)

    points: list[BenchmarkPoint] = []
    sample_sizes = [size for size in args.sample_sizes if size > 0]
    if not sample_sizes:
        raise ValueError("sample-sizes must contain positive integers")

    for size in sample_sizes:
        n_rows = min(size, full_rows)
        subset_csv = bench_dir / f"train_{n_rows}.csv"
        write_subset_csv(args.source_csv, subset_csv, n_rows)
        point = run_benchmark(
            args.exe,
            subset_csv,
            args.seq_len,
            args.lstm_hidden,
            args.batch_size,
            args.optimizer,
            args.lr,
            args.backend,
            args.max_calibration_sec,
        )
        points.append(point)
        print(
            f"rows={point.rows} seq_samples={point.seq_samples} train_ms={point.train_ms} "
            f"eval_ms={point.eval_ms} total_ms={point.total_ms} timeout={point.timed_out}"
        )

    train_estimate = estimate_capacity(points, args.timeout_sec, "train_ms")
    total_estimate = estimate_capacity(points, args.timeout_sec, "total_ms")

    conservative_estimate = total_estimate
    estimated_rows = conservative_estimate["estimated_rows_for_timeout"]
    estimated_seq_samples = max(0.0, estimated_rows - args.seq_len + 1)
    train_full_estimated_sec = train_estimate["intercept_sec"] + train_estimate["slope_sec_per_row"] * full_rows
    total_full_estimated_sec = total_estimate["intercept_sec"] + total_estimate["slope_sec_per_row"] * full_rows

    report = {
        "source_csv": str(args.source_csv),
        "backend": args.backend,
        "timeout_sec": args.timeout_sec,
        "full_rows": full_rows,
        "seq_len": args.seq_len,
        "lstm_hidden": args.lstm_hidden,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "calibration_points": [asdict(p) for p in points],
        "estimate": {
            "train_only": {
                **train_estimate,
                "estimated_seq_samples_for_timeout": max(0.0, train_estimate["estimated_rows_for_timeout"] - args.seq_len + 1),
                "estimated_full_dataset_sec": train_full_estimated_sec,
            },
            "wall_clock": {
                **total_estimate,
                "estimated_seq_samples_for_timeout": max(0.0, total_estimate["estimated_rows_for_timeout"] - args.seq_len + 1),
                "estimated_full_dataset_sec": total_full_estimated_sec,
            },
            "recommended": {
                **conservative_estimate,
                "estimated_seq_samples_for_timeout": estimated_seq_samples,
                "estimated_full_dataset_sec": total_full_estimated_sec,
            },
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"estimated_rows_for_{args.timeout_sec:.0f}s_train_only={int(train_estimate['estimated_rows_for_timeout'])}")
    print(f"estimated_rows_for_{args.timeout_sec:.0f}s_wall_clock={int(total_estimate['estimated_rows_for_timeout'])}")
    print(f"recommended_rows_for_{args.timeout_sec:.0f}s={int(estimated_rows)}")
    print(f"recommended_seq_samples_for_{args.timeout_sec:.0f}s={int(estimated_seq_samples)}")
    print(f"estimated_full_dataset_train_sec={train_full_estimated_sec:.3f}")
    print(f"estimated_full_dataset_wall_sec={total_full_estimated_sec:.3f}")
    print(f"report_json={args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())