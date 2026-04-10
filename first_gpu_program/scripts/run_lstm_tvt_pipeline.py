#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class CommandResult:
    stage: str
    command: List[str]
    return_code: int
    elapsed_sec: float
    log_file: str


EPOCH_PROGRESS_PATTERN = re.compile(
    r"\[Epoch\s+(\d+)/(\d+)\].*train_time=(\d+)\s+ms,\s+eval_time=(\d+)\s+ms"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run organized LSTM TVT pipeline with frequency downsampling")
    parser.add_argument("--exe", type=Path, default=Path("build/Debug/first_gpu_program.exe"))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/LSTM/processed"))
    parser.add_argument("--output-root", type=Path, default=Path("datasets/LSTM/processed/experiments"))
    parser.add_argument("--run-name", type=str, default="")

    parser.add_argument("--epoch-budget-sec", type=float, default=60.0)
    parser.add_argument(
        "--capacity-report",
        type=Path,
        default=Path("datasets/LSTM/processed/lstm_gpu_capacity_report.json"),
        help="Capacity estimate JSON produced by scripts/estimate_lstm_gpu_capacity.py",
    )
    parser.add_argument(
        "--target-train-rows",
        type=int,
        default=0,
        help="Explicit train row budget after downsampling. 0 means infer from capacity report.",
    )
    parser.add_argument(
        "--compression-step",
        type=int,
        default=0,
        help="Keep one sample every N rows (frequency downsampling). 0 means auto-calculate.",
    )

    parser.add_argument("--epoch-sweep", type=int, nargs="+", default=[1, 3, 5, 7, 10])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--lstm-hidden", type=int, default=16)
    parser.add_argument("--loss", type=str, choices=["bce", "mse", "focal"], default="bce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, choices=["sgd", "momentum", "adam", "adamw"], default="sgd")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-decay", type=float, default=1.0)
    parser.add_argument("--lr-decay-every", type=int, default=1)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--neg-weight", type=float, default=1.0)
    parser.add_argument("--backend", type=str, choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--timeout-sec", type=float, default=0.0)
    parser.add_argument("--auto-class-weights", action="store_true")
    parser.add_argument(
        "--train-once-max-epoch",
        action="store_true",
        help="Run a single training job to the maximum epoch and reuse its training metrics for requested epochs.",
    )
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def format_seconds(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def run_and_log(
    stage: str,
    command: List[str],
    log_path: Path,
    cwd: Path,
    hard_timeout_sec: float = 0.0,
) -> CommandResult:
    start = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=cwd,
    )

    if process.stdout is None:
        raise RuntimeError(f"Stage '{stage}' failed to capture stdout")

    line_queue: queue.Queue[str | None] = queue.Queue()

    def _pump_stdout() -> None:
        try:
            for line in process.stdout:
                line_queue.put(line)
        finally:
            line_queue.put(None)

    reader_thread = threading.Thread(target=_pump_stdout, daemon=True)
    reader_thread.start()

    last_epoch_eta_sec: float | None = None
    with log_path.open("w", encoding="utf-8", newline="", buffering=1) as log_handle:
        while True:
            if hard_timeout_sec > 0.0:
                elapsed_now = time.perf_counter() - start
                if elapsed_now >= hard_timeout_sec:
                    timeout_msg = (
                        f"[stage:{stage}] hard timeout reached at {format_seconds(elapsed_now)} "
                        f"(limit={format_seconds(hard_timeout_sec)}), terminating child process"
                    )
                    print(timeout_msg, flush=True)
                    log_handle.write(timeout_msg + "\n")
                    log_handle.flush()
                    process.terminate()
                    try:
                        process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise RuntimeError(
                        f"Stage '{stage}' exceeded hard timeout {hard_timeout_sec:.3f} sec. "
                        f"See log: {log_path}"
                    )

            try:
                line = line_queue.get(timeout=0.2)
            except queue.Empty:
                if process.poll() is not None:
                    break
                continue

            if line is None:
                break

            # Stream child stdout to both terminal and stage log in real time.
            print(line, end="", flush=True)
            log_handle.write(line)
            log_handle.flush()

            match = EPOCH_PROGRESS_PATTERN.search(line)
            if match:
                epoch_now = int(match.group(1))
                epoch_total = int(match.group(2))
                train_ms = int(match.group(3))
                eval_ms = int(match.group(4))
                per_epoch_sec = max(0.001, (train_ms + eval_ms) / 1000.0)
                remain_epochs = max(0, epoch_total - epoch_now)
                last_epoch_eta_sec = remain_epochs * per_epoch_sec
                print(
                    f"[stage:{stage}] epoch {epoch_now}/{epoch_total} "
                    f"({epoch_now / max(1, epoch_total) * 100:.1f}%) "
                    f"stage_eta~{format_seconds(last_epoch_eta_sec)}"
                )


    process.wait()
    elapsed = time.perf_counter() - start
    if process.returncode != 0:
        raise RuntimeError(f"Stage '{stage}' failed (code={process.returncode}). See log: {log_path}")
    if last_epoch_eta_sec is not None:
        print(f"[stage:{stage}] done in {format_seconds(elapsed)}")
    return CommandResult(stage=stage, command=command, return_code=process.returncode, elapsed_sec=elapsed, log_file=str(log_path))


def print_pipeline_progress(completed_stages: int, total_stages: int, pipeline_start: float) -> None:
    elapsed = time.perf_counter() - pipeline_start
    if completed_stages <= 0:
        print(
            f"[pipeline] 0/{total_stages} stages completed, elapsed={format_seconds(elapsed)}, eta=unknown"
        )
        return
    remain = max(0, total_stages - completed_stages)
    avg_stage_sec = elapsed / float(completed_stages)
    eta_sec = remain * avg_stage_sec
    print(
        f"[pipeline] {completed_stages}/{total_stages} stages completed "
        f"({completed_stages / max(1, total_stages) * 100:.1f}%), "
        f"elapsed={format_seconds(elapsed)}, eta~{format_seconds(eta_sec)}"
    )


def count_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        next(handle, None)
        return sum(1 for _ in handle)


def count_labels_csv(csv_path: Path) -> dict[str, int]:
    rows = 0
    pos = 0
    neg = 0
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            rows += 1
            label = row[-1].strip()
            if label == "1":
                pos += 1
            elif label == "0":
                neg += 1
    return {"rows": rows, "positives": pos, "negatives": neg}


def downsample_frequency(source_csv: Path, target_csv: Path, step: int) -> dict[str, int]:
    if step <= 0:
        raise ValueError("compression-step must be positive")

    target_csv.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    dropped = 0
    with source_csv.open("r", encoding="utf-8", newline="") as src, target_csv.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"CSV is empty: {source_csv}")
        writer.writerow(header)

        for idx, row in enumerate(reader):
            if idx % step == 0:
                writer.writerow(row)
                kept += 1
            else:
                dropped += 1

    return {"kept": kept, "dropped": dropped, "step": step}


def load_target_rows_from_capacity(report_path: Path, epoch_budget_sec: float) -> int:
    if not report_path.exists():
        raise FileNotFoundError(f"Capacity report not found: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    timeout = float(report.get("timeout_sec", 0.0))
    if abs(timeout - epoch_budget_sec) > 1e-3:
        # The report can still be used if it has the same scale, but warn by raising explicit error.
        raise ValueError(
            f"Capacity report timeout_sec={timeout} does not match requested epoch budget {epoch_budget_sec}"
        )
    estimate = report.get("estimate", {}).get("recommended", {})
    rows = int(float(estimate.get("estimated_rows_for_timeout", 0)))
    if rows <= 0:
        raise ValueError(f"Invalid estimated_rows_for_timeout in {report_path}")
    return rows


def parse_last_metrics(results_csv: Path) -> dict[str, float]:
    if not results_csv.exists():
        return {}
    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        last: dict[str, str] | None = None
        for row in reader:
            last = row
    if last is None:
        return {}

    fields = ["eval_cost", "accuracy", "precision", "recall", "specificity", "f1", "elapsed_ms"]
    out: dict[str, float] = {}
    for key in fields:
        value = last.get(key)
        if value is None or value == "":
            continue
        try:
            out[key] = float(value)
        except ValueError:
            continue
    return out


def parse_epoch_metrics(results_csv: Path) -> dict[int, dict[str, float]]:
    if not results_csv.exists():
        return {}

    epoch_rows: dict[int, dict[str, float]] = {}
    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("phase") != "epoch":
                continue
            try:
                epoch = int(row.get("epoch", ""))
            except ValueError:
                continue

            metrics: dict[str, float] = {}
            for key in [
                "train_loss",
                "eval_cost",
                "accuracy",
                "precision",
                "recall",
                "specificity",
                "f1",
                "elapsed_ms",
            ]:
                value = row.get(key)
                if value is None or value == "":
                    continue
                try:
                    metrics[key] = float(value)
                except ValueError:
                    continue
            epoch_rows[epoch] = metrics

    return epoch_rows


def parse_epoch_timings(log_path: Path) -> dict[int, dict[str, float]]:
    if not log_path.exists():
        return {}

    epoch_rows: dict[int, dict[str, float]] = {}
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = EPOCH_PROGRESS_PATTERN.search(line)
        if not match:
            continue
        epoch = int(match.group(1))
        epoch_rows[epoch] = {
            "train_time_ms": float(int(match.group(3))),
            "eval_time_ms": float(int(match.group(4))),
        }
    return epoch_rows


def main() -> int:
    args = parse_args()
    if not args.exe.exists():
        raise FileNotFoundError(f"Executable not found: {args.exe}")
    if args.epoch_budget_sec <= 0.0:
        raise ValueError("epoch-budget-sec must be positive")
    if args.timeout_sec < 0.0:
        raise ValueError("timeout-sec must be non-negative")
    if not args.epoch_sweep:
        raise ValueError("epoch-sweep cannot be empty")
    if any(epoch <= 0 for epoch in args.epoch_sweep):
        raise ValueError("epoch-sweep values must be positive")

    data_dir = args.data_dir
    source_train_csv = data_dir / "train.csv"
    source_val_csv = data_dir / "val.csv"
    source_test_csv = data_dir / "test.csv"
    for p in (source_train_csv, source_val_csv, source_test_csv):
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

    compressed_train_csv = run_dir / "results" / "freq_train.csv"
    compressed_val_csv = run_dir / "results" / "freq_val.csv"
    compressed_test_csv = run_dir / "results" / "freq_test.csv"
    compression_manifest = run_dir / "configs" / "frequency_compression_manifest.json"

    summary: List[CommandResult] = []
    root = Path.cwd()

    source_rows = {
        "train": count_rows(source_train_csv),
        "val": count_rows(source_val_csv),
        "test": count_rows(source_test_csv),
    }

    if args.compression_step > 0:
        compression_step = args.compression_step
        target_rows = max(1, source_rows["train"] // compression_step)
        target_source = "manual_step"
    else:
        if args.target_train_rows > 0:
            target_rows = args.target_train_rows
            target_source = "manual_target_rows"
        else:
            target_rows = load_target_rows_from_capacity(args.capacity_report, args.epoch_budget_sec)
            target_source = "capacity_report"
        target_rows = min(target_rows, source_rows["train"])
        compression_step = max(1, math.ceil(source_rows["train"] / max(1, target_rows)))

    train_downsample = downsample_frequency(source_train_csv, compressed_train_csv, compression_step)
    val_downsample = downsample_frequency(source_val_csv, compressed_val_csv, compression_step)
    test_downsample = downsample_frequency(source_test_csv, compressed_test_csv, compression_step)

    compressed_rows = {
        "train": count_rows(compressed_train_csv),
        "val": count_rows(compressed_val_csv),
        "test": count_rows(compressed_test_csv),
    }

    manifest = {
        "source": {
            "train": str(source_train_csv),
            "val": str(source_val_csv),
            "test": str(source_test_csv),
        },
        "compression": {
            "mode": "frequency",
            "source_hz": 1.0,
            "compression_step": compression_step,
            "target_hz": 1.0 / float(compression_step),
            "epoch_budget_sec": args.epoch_budget_sec,
            "target_train_rows": target_rows,
            "target_rows_source": target_source,
        },
        "source_rows": source_rows,
        "compressed_rows": compressed_rows,
        "downsample_stats": {
            "train": train_downsample,
            "val": val_downsample,
            "test": test_downsample,
        },
        "label_distribution": {
            "train": count_labels_csv(compressed_train_csv),
            "val": count_labels_csv(compressed_val_csv),
            "test": count_labels_csv(compressed_test_csv),
        },
        "compressed_paths": {
            "train": str(compressed_train_csv),
            "val": str(compressed_val_csv),
            "test": str(compressed_test_csv),
        },
    }
    compression_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    epoch_sweep_summary: List[Dict[str, Any]] = []
    if args.train_once_max_epoch:
        total_stages = 1 + (2 if not args.skip_plots else 0)
    else:
        per_epoch_stage_count = 3 + (6 if not args.skip_plots else 0)
        total_stages = len(args.epoch_sweep) * per_epoch_stage_count
    completed_stages = 0
    pipeline_start = time.perf_counter()
    print(
        f"[pipeline] starting run='{run_name}', total stages={total_stages}, "
        f"epochs={args.epoch_sweep}, plots={'off' if args.skip_plots else 'on'}, "
        f"train_once_max_epoch={'on' if args.train_once_max_epoch else 'off'}"
    )
    print_pipeline_progress(completed_stages, total_stages, pipeline_start)

    if args.train_once_max_epoch:
        epochs = max(args.epoch_sweep)
        epoch_tag = f"epochs_{epochs}_once"
        model_file = run_dir / "models" / f"lstm_model_{epoch_tag}.bin"
        train_results = run_dir / "results" / f"train_metrics_{epoch_tag}.csv"
        train_pr = run_dir / "results" / f"train_pr_{epoch_tag}.csv"
        train_log = dirs["logs"] / f"train_{epoch_tag}.log"

        train_cmd = [
            str(args.exe),
            "--model",
            "lstm",
            "--dataset",
            str(compressed_train_csv),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(args.batch_size),
            "--print-every",
            "1",
            "--seq-len",
            str(args.seq_len),
            "--lstm-hidden",
            str(args.lstm_hidden),
            "--loss",
            str(args.loss),
            "--focal-gamma",
            str(args.focal_gamma),
            "--focal-alpha",
            str(args.focal_alpha),
            "--lr",
            str(args.lr),
            "--optimizer",
            str(args.optimizer),
            "--momentum",
            str(args.momentum),
            "--adam-beta1",
            str(args.adam_beta1),
            "--adam-beta2",
            str(args.adam_beta2),
            "--adam-eps",
            str(args.adam_eps),
            "--weight-decay",
            str(args.weight_decay),
            "--lr-decay",
            str(args.lr_decay),
            "--lr-decay-every",
            str(args.lr_decay_every),
            "--min-lr",
            str(args.min_lr),
            "--threshold",
            str(args.threshold),
            "--pos-weight",
            str(args.pos_weight),
            "--neg-weight",
            str(args.neg_weight),
            "--backend",
            args.backend,
            "--timeout-sec",
            str(args.timeout_sec),
            "--save-model",
            str(model_file),
            "--results-csv",
            str(train_results),
            "--pr-csv",
            str(train_pr),
        ]
        if args.auto_class_weights:
            train_cmd.append("--auto-class-weights")

        train_result = run_and_log(
            f"train_{epoch_tag}",
            train_cmd,
            train_log,
            root,
            hard_timeout_sec=args.timeout_sec,
        )
        summary.append(train_result)
        completed_stages += 1
        print_pipeline_progress(completed_stages, total_stages, pipeline_start)

        train_metrics_by_epoch = parse_epoch_metrics(train_results)
        train_timings_by_epoch = parse_epoch_timings(train_log)
        requested_epochs: List[int] = []
        seen_epochs: set[int] = set()
        for epoch in args.epoch_sweep:
            if epoch not in seen_epochs:
                requested_epochs.append(epoch)
                seen_epochs.add(epoch)

        for epoch in requested_epochs:
            metrics = train_metrics_by_epoch.get(epoch)
            timings = train_timings_by_epoch.get(epoch)
            if metrics is None or timings is None:
                raise RuntimeError(f"Missing epoch {epoch} metrics in one-pass training output")
            epoch_sweep_summary.append(
                {
                    "epochs": epoch,
                    "train_elapsed_sec": (timings["train_time_ms"] + timings["eval_time_ms"]) / 1000.0,
                    "train_metrics": metrics,
                    "artifacts": {
                        "model": str(model_file),
                        "train_results": str(train_results),
                        "train_pr": str(train_pr),
                    },
                }
            )

        if not args.skip_plots:
            plot_jobs = [
                ("plot_train_metrics_once", train_results, run_dir / "figures" / f"train_metrics_{epoch_tag}.png", "scripts/plot_training_results.py"),
                ("plot_train_pr_once", train_pr, run_dir / "figures" / f"train_pr_{epoch_tag}.png", "scripts/plot_pr_curve.py"),
            ]
            for stage, input_csv, output_png, script_path in plot_jobs:
                cmd = [str(args.python), script_path, "--input", str(input_csv), "--output", str(output_png)]
                summary.append(run_and_log(stage, cmd, dirs["logs"] / f"{stage}.log", root))
                completed_stages += 1
                print_pipeline_progress(completed_stages, total_stages, pipeline_start)
    else:
        for epochs in args.epoch_sweep:
            epoch_tag = f"epochs_{epochs}"
            model_file = run_dir / "models" / f"lstm_model_{epoch_tag}.bin"
            train_results = run_dir / "results" / f"train_metrics_{epoch_tag}.csv"
            train_pr = run_dir / "results" / f"train_pr_{epoch_tag}.csv"
            val_results = run_dir / "results" / f"val_metrics_{epoch_tag}.csv"
            val_pr = run_dir / "results" / f"val_pr_{epoch_tag}.csv"
            test_results = run_dir / "results" / f"test_metrics_{epoch_tag}.csv"
            test_pr = run_dir / "results" / f"test_pr_{epoch_tag}.csv"

            train_cmd = [
                str(args.exe),
                "--model",
                "lstm",
                "--dataset",
                str(compressed_train_csv),
                "--epochs",
                str(epochs),
                "--batch-size",
                str(args.batch_size),
                "--print-every",
                "1",
                "--seq-len",
                str(args.seq_len),
                "--lstm-hidden",
                str(args.lstm_hidden),
                "--loss",
                str(args.loss),
                "--focal-gamma",
                str(args.focal_gamma),
                "--focal-alpha",
                str(args.focal_alpha),
                "--lr",
                str(args.lr),
                "--optimizer",
                str(args.optimizer),
                "--momentum",
                str(args.momentum),
                "--adam-beta1",
                str(args.adam_beta1),
                "--adam-beta2",
                str(args.adam_beta2),
                "--adam-eps",
                str(args.adam_eps),
                "--weight-decay",
                str(args.weight_decay),
                "--lr-decay",
                str(args.lr_decay),
                "--lr-decay-every",
                str(args.lr_decay_every),
                "--min-lr",
                str(args.min_lr),
                "--threshold",
                str(args.threshold),
                "--pos-weight",
                str(args.pos_weight),
                "--neg-weight",
                str(args.neg_weight),
                "--backend",
                args.backend,
                "--timeout-sec",
                str(args.timeout_sec),
                "--save-model",
                str(model_file),
                "--results-csv",
                str(train_results),
                "--pr-csv",
                str(train_pr),
            ]
            if args.auto_class_weights:
                train_cmd.append("--auto-class-weights")
            train_result = run_and_log(
                f"train_{epoch_tag}",
                train_cmd,
                dirs["logs"] / f"train_{epoch_tag}.log",
                root,
                hard_timeout_sec=args.timeout_sec,
            )
            summary.append(train_result)
            completed_stages += 1
            print_pipeline_progress(completed_stages, total_stages, pipeline_start)

            val_cmd = [
                str(args.exe),
                "--model",
                "lstm",
                "--dataset",
                str(compressed_val_csv),
                "--eval-only",
                "--seq-len",
                str(args.seq_len),
                "--lstm-hidden",
                str(args.lstm_hidden),
                "--loss",
                str(args.loss),
                "--focal-gamma",
                str(args.focal_gamma),
                "--focal-alpha",
                str(args.focal_alpha),
                "--lr",
                str(args.lr),
                "--optimizer",
                str(args.optimizer),
                "--momentum",
                str(args.momentum),
                "--adam-beta1",
                str(args.adam_beta1),
                "--adam-beta2",
                str(args.adam_beta2),
                "--adam-eps",
                str(args.adam_eps),
                "--weight-decay",
                str(args.weight_decay),
                "--lr-decay",
                str(args.lr_decay),
                "--lr-decay-every",
                str(args.lr_decay_every),
                "--min-lr",
                str(args.min_lr),
                "--threshold",
                str(args.threshold),
                "--pos-weight",
                str(args.pos_weight),
                "--neg-weight",
                str(args.neg_weight),
                "--backend",
                args.backend,
                "--load-model",
                str(model_file),
                "--results-csv",
                str(val_results),
                "--pr-csv",
                str(val_pr),
            ]
            val_result = run_and_log(
                f"val_{epoch_tag}",
                val_cmd,
                dirs["logs"] / f"val_{epoch_tag}.log",
                root,
                hard_timeout_sec=args.timeout_sec,
            )
            summary.append(val_result)
            completed_stages += 1
            print_pipeline_progress(completed_stages, total_stages, pipeline_start)

            test_cmd = [
                str(args.exe),
                "--model",
                "lstm",
                "--dataset",
                str(compressed_test_csv),
                "--eval-only",
                "--seq-len",
                str(args.seq_len),
                "--lstm-hidden",
                str(args.lstm_hidden),
                "--loss",
                str(args.loss),
                "--focal-gamma",
                str(args.focal_gamma),
                "--focal-alpha",
                str(args.focal_alpha),
                "--lr",
                str(args.lr),
                "--optimizer",
                str(args.optimizer),
                "--momentum",
                str(args.momentum),
                "--adam-beta1",
                str(args.adam_beta1),
                "--adam-beta2",
                str(args.adam_beta2),
                "--adam-eps",
                str(args.adam_eps),
                "--weight-decay",
                str(args.weight_decay),
                "--lr-decay",
                str(args.lr_decay),
                "--lr-decay-every",
                str(args.lr_decay_every),
                "--min-lr",
                str(args.min_lr),
                "--threshold",
                str(args.threshold),
                "--pos-weight",
                str(args.pos_weight),
                "--neg-weight",
                str(args.neg_weight),
                "--backend",
                args.backend,
                "--load-model",
                str(model_file),
                "--results-csv",
                str(test_results),
                "--pr-csv",
                str(test_pr),
            ]
            test_result = run_and_log(
                f"test_{epoch_tag}",
                test_cmd,
                dirs["logs"] / f"test_{epoch_tag}.log",
                root,
                hard_timeout_sec=args.timeout_sec,
            )
            summary.append(test_result)
            completed_stages += 1
            print_pipeline_progress(completed_stages, total_stages, pipeline_start)

            train_metrics = parse_last_metrics(train_results)
            val_metrics = parse_last_metrics(val_results)
            test_metrics = parse_last_metrics(test_results)
            epoch_sweep_summary.append(
                {
                    "epochs": epochs,
                    "train_elapsed_sec": train_result.elapsed_sec,
                    "val_elapsed_sec": val_result.elapsed_sec,
                    "test_elapsed_sec": test_result.elapsed_sec,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                    "artifacts": {
                        "model": str(model_file),
                        "train_results": str(train_results),
                        "val_results": str(val_results),
                        "test_results": str(test_results),
                        "train_pr": str(train_pr),
                        "val_pr": str(val_pr),
                        "test_pr": str(test_pr),
                    },
                }
            )

            if not args.skip_plots:
                plot_jobs = [
                    (f"plot_train_metrics_{epoch_tag}", train_results, run_dir / "figures" / f"train_metrics_{epoch_tag}.png", "scripts/plot_training_results.py"),
                    (f"plot_train_pr_{epoch_tag}", train_pr, run_dir / "figures" / f"train_pr_{epoch_tag}.png", "scripts/plot_pr_curve.py"),
                    (f"plot_val_metrics_{epoch_tag}", val_results, run_dir / "figures" / f"val_metrics_{epoch_tag}.png", "scripts/plot_training_results.py"),
                    (f"plot_val_pr_{epoch_tag}", val_pr, run_dir / "figures" / f"val_pr_{epoch_tag}.png", "scripts/plot_pr_curve.py"),
                    (f"plot_test_metrics_{epoch_tag}", test_results, run_dir / "figures" / f"test_metrics_{epoch_tag}.png", "scripts/plot_training_results.py"),
                    (f"plot_test_pr_{epoch_tag}", test_pr, run_dir / "figures" / f"test_pr_{epoch_tag}.png", "scripts/plot_pr_curve.py"),
                ]

                for stage, input_csv, output_png, script_path in plot_jobs:
                    cmd = [str(args.python), script_path, "--input", str(input_csv), "--output", str(output_png)]
                    summary.append(run_and_log(stage, cmd, dirs["logs"] / f"{stage}.log", root))
                    completed_stages += 1
                    print_pipeline_progress(completed_stages, total_stages, pipeline_start)

    sweep_csv_path = run_dir / "results" / "epoch_sweep_summary.csv"
    with sweep_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if args.train_once_max_epoch:
            writer.writerow([
                "epochs",
                "train_elapsed_sec",
                "train_loss",
                "eval_cost",
                "accuracy",
                "precision",
                "recall",
                "specificity",
                "f1",
            ])
            for row in epoch_sweep_summary:
                train_m = row.get("train_metrics", {})
                writer.writerow([
                    row["epochs"],
                    f"{row['train_elapsed_sec']:.6f}",
                    train_m.get("train_loss", ""),
                    train_m.get("eval_cost", ""),
                    train_m.get("accuracy", ""),
                    train_m.get("precision", ""),
                    train_m.get("recall", ""),
                    train_m.get("specificity", ""),
                    train_m.get("f1", ""),
                ])
        else:
            writer.writerow([
                "epochs",
                "train_elapsed_sec",
                "val_elapsed_sec",
                "test_elapsed_sec",
                "val_eval_cost",
                "val_accuracy",
                "val_f1",
                "test_eval_cost",
                "test_accuracy",
                "test_f1",
            ])
            for row in epoch_sweep_summary:
                val_m = row.get("val_metrics", {})
                test_m = row.get("test_metrics", {})
                writer.writerow([
                    row["epochs"],
                    f"{row['train_elapsed_sec']:.6f}",
                    f"{row['val_elapsed_sec']:.6f}",
                    f"{row['test_elapsed_sec']:.6f}",
                    val_m.get("eval_cost", ""),
                    val_m.get("accuracy", ""),
                    val_m.get("f1", ""),
                    test_m.get("eval_cost", ""),
                    test_m.get("accuracy", ""),
                    test_m.get("f1", ""),
                ])

    summary_json = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "params": {
            "epoch_sweep": args.epoch_sweep,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "lstm_hidden": args.lstm_hidden,
            "loss": args.loss,
            "focal_gamma": args.focal_gamma,
            "focal_alpha": args.focal_alpha,
            "lr": args.lr,
            "optimizer": args.optimizer,
            "momentum": args.momentum,
            "adam_beta1": args.adam_beta1,
            "adam_beta2": args.adam_beta2,
            "adam_eps": args.adam_eps,
            "lr_decay": args.lr_decay,
            "lr_decay_every": args.lr_decay_every,
            "min_lr": args.min_lr,
            "threshold": args.threshold,
            "pos_weight": args.pos_weight,
            "neg_weight": args.neg_weight,
            "backend": args.backend,
            "timeout_sec": args.timeout_sec,
            "epoch_budget_sec": args.epoch_budget_sec,
            "capacity_report": str(args.capacity_report),
            "target_train_rows": args.target_train_rows,
            "compression_step": compression_step,
            "train_once_max_epoch": args.train_once_max_epoch,
        },
        "artifacts": {
            "frequency_compression_manifest": str(compression_manifest),
            "compressed_splits": {
                "train": str(compressed_train_csv),
                "val": str(compressed_val_csv),
                "test": str(compressed_test_csv),
            },
            "epoch_sweep_summary_csv": str(sweep_csv_path),
            "logs_dir": str(dirs["logs"]),
        },
        "epoch_sweep": epoch_sweep_summary,
        "stages": [asdict(item) for item in summary],
    }

    summary_path = run_dir / "configs" / "run_summary.json"
    summary_path.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    print(f"run_dir={run_dir}")
    print(f"compression_step={compression_step}")
    print(f"target_hz={1.0 / float(compression_step):.6f}")
    print(f"compressed_train_rows={compressed_rows['train']}")
    print(f"epoch_sweep_summary={sweep_csv_path}")
    print(f"summary={summary_path}")
    print_pipeline_progress(completed_stages, total_stages, pipeline_start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
