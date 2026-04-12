#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a PyTorch LSTM TVT pipeline with organized experiment outputs")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/LSTM/processed"))
    parser.add_argument("--output-root", type=Path, default=Path("datasets/LSTM/processed/experiments"))
    parser.add_argument("--compression-step", type=int, default=0, help="Keep one sample every N rows")
    parser.add_argument("--target-train-rows", type=int, default=0, help="Infer compression step from target train rows")

    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "cpu", "gpu", "directml"])
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--lstm-hidden", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--loss", type=str, default="bce", choices=["bce", "mse", "focal"])
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--neg-weight", type=float, default=1.0)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "momentum", "adam", "adamw"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr-decay", type=float, default=1.0)
    parser.add_argument("--lr-decay-every", type=int, default=1)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--auto-class-weights", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--timeout-sec", type=float, default=0.0)
    parser.add_argument("--pr-min", type=float, default=0.0)
    parser.add_argument("--pr-max", type=float, default=1.0)
    parser.add_argument("--pr-step", type=float, default=0.02)

    parser.add_argument("--skip-pr", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--load-model", type=Path, default=None)
    parser.add_argument("--save-model", type=Path, default=None)
    return parser.parse_args()


def format_seconds(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def count_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        next(handle, None)
        return sum(1 for line in handle if line.strip())


def downsample_frequency(source_csv: Path, target_csv: Path, step: int) -> dict[str, int]:
    if step <= 0:
        raise ValueError("compression step must be positive")

    target_csv.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    dropped = 0
    with source_csv.open("r", encoding="utf-8", newline="") as src, target_csv.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst)
        header = next(reader, None)
        if header is None:
            raise RuntimeError(f"CSV is empty: {source_csv}")
        writer.writerow(header)

        for idx, row in enumerate(reader):
            if idx % step == 0:
                writer.writerow(row)
                kept += 1
            else:
                dropped += 1

    return {"kept": kept, "dropped": dropped, "step": step}


def tee_process(command: list[str], cwd: Path, log_path: Path) -> int:
    start = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert process.stdout is not None
    with log_path.open("w", encoding="utf-8", newline="", buffering=1) as log_handle:
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
        process.wait()
        elapsed = time.perf_counter() - start
        if process.returncode == 0:
            print(f"[pytorch-tvt] done in {format_seconds(elapsed)}")
        return int(process.returncode or 0)


def parse_runtime_backend(log_path: Path) -> dict[str, str]:
    backend: dict[str, str] = {}
    if not log_path.exists():
        return backend

    backend_pattern = re.compile(r"^Selected backend:\s*(?P<backend>\S+)\s*$")
    device_pattern = re.compile(r"^Device detail:\s*(?P<detail>.+?)\s*$")
    torch_pattern = re.compile(r"^PyTorch:\s*(?P<version>\S+)\s*$")

    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "selected_backend" not in backend and (match := backend_pattern.match(line)):
            backend["selected_backend"] = match.group("backend")
            continue
        if "device_detail" not in backend and (match := device_pattern.match(line)):
            backend["device_detail"] = match.group("detail")
            continue
        if "torch_version" not in backend and (match := torch_pattern.match(line)):
            backend["torch_version"] = match.group("version")
            continue
    return backend


def build_command(args: argparse.Namespace, train_csv: Path, val_csv: Path, test_csv: Path, run_dir: Path) -> list[str]:
    core_script = Path(__file__).with_name("train_lstm_pytorch.py")
    if not core_script.exists():
        raise FileNotFoundError(f"Core trainer not found: {core_script}")

    command = [
        sys.executable,
        str(core_script),
        "--train-csv",
        str(train_csv),
        "--val-csv",
        str(val_csv),
        "--test-csv",
        str(test_csv),
        "--backend",
        args.backend,
        "--seq-len",
        str(args.seq_len),
        "--lstm-hidden",
        str(args.lstm_hidden),
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--print-every",
        str(args.print_every),
        "--loss",
        args.loss,
        "--focal-gamma",
        str(args.focal_gamma),
        "--focal-alpha",
        str(args.focal_alpha),
        "--threshold",
        str(args.threshold),
        "--pos-weight",
        str(args.pos_weight),
        "--neg-weight",
        str(args.neg_weight),
        "--optimizer",
        args.optimizer,
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
        "--lr",
        str(args.lr),
        "--lr-decay",
        str(args.lr_decay),
        "--lr-decay-every",
        str(args.lr_decay_every),
        "--min-lr",
        str(args.min_lr),
        "--timeout-sec",
        str(args.timeout_sec),
        "--results-csv",
        str(run_dir / "results" / "results.csv"),
    ]

    if not args.skip_pr:
        command += ["--pr-csv", str(run_dir / "results" / "pr.csv")]
    if args.auto_class_weights:
        command.append("--auto-class-weights")
    if args.eval_only:
        command.append("--eval-only")
    if args.load_model is not None:
        command += ["--load-model", str(args.load_model)]
    if args.save_model is not None:
        command += ["--save-model", str(args.save_model)]

    return command


def main() -> int:
    args = parse_args()

    data_dir = args.data_dir
    source_train_csv = data_dir / "train.csv"
    source_val_csv = data_dir / "val.csv"
    source_test_csv = data_dir / "test.csv"
    for path in (source_train_csv, source_val_csv, source_test_csv):
        if not path.exists():
            raise FileNotFoundError(f"Required split missing: {path}")

    run_name = args.run_name.strip() or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = args.output_root / run_name
    results_dir = run_dir / "results"
    logs_dir = run_dir / "logs"
    configs_dir = run_dir / "configs"
    models_dir = run_dir / "models"
    for directory in (results_dir, logs_dir, configs_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)

    source_rows = {
        "train": count_rows(source_train_csv),
        "val": count_rows(source_val_csv),
        "test": count_rows(source_test_csv),
    }

    if args.compression_step > 0:
        compression_step = args.compression_step
        target_source = "manual_step"
    elif args.target_train_rows > 0:
        compression_step = max(1, source_rows["train"] // args.target_train_rows)
        target_source = "target_train_rows"
    else:
        compression_step = 1
        target_source = "none"

    compressed_train = results_dir / "freq_train.csv"
    compressed_val = results_dir / "freq_val.csv"
    compressed_test = results_dir / "freq_test.csv"

    if compression_step == 1:
        compressed_train.write_text(source_train_csv.read_text(encoding="utf-8"), encoding="utf-8")
        compressed_val.write_text(source_val_csv.read_text(encoding="utf-8"), encoding="utf-8")
        compressed_test.write_text(source_test_csv.read_text(encoding="utf-8"), encoding="utf-8")
        compression_stats = {
            "train": {"kept": source_rows["train"], "dropped": 0, "step": 1},
            "val": {"kept": source_rows["val"], "dropped": 0, "step": 1},
            "test": {"kept": source_rows["test"], "dropped": 0, "step": 1},
        }
    else:
        compression_stats = {
            "train": downsample_frequency(source_train_csv, compressed_train, compression_step),
            "val": downsample_frequency(source_val_csv, compressed_val, compression_step),
            "test": downsample_frequency(source_test_csv, compressed_test, compression_step),
        }

    serializable_args: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            serializable_args[key] = str(value)
        else:
            serializable_args[key] = value

    manifest: dict[str, Any] = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_dir": str(data_dir),
        "output_root": str(args.output_root),
        "compression_step": compression_step,
        "compression_source": target_source,
        "source_rows": source_rows,
        "compressed_rows": {
            "train": compression_stats["train"]["kept"],
            "val": compression_stats["val"]["kept"],
            "test": compression_stats["test"]["kept"],
        },
        "args": serializable_args,
    }
    (configs_dir / "tvt_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    command = build_command(args, compressed_train, compressed_val, compressed_test, run_dir)
    (configs_dir / "train_command.json").write_text(json.dumps(command, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[pytorch-tvt] run_dir={run_dir}")
    print(f"[pytorch-tvt] compression_step={compression_step}")
    print(
        f"[pytorch-tvt] rows train={compression_stats['train']['kept']}, val={compression_stats['val']['kept']}, test={compression_stats['test']['kept']}"
    )

    log_path = logs_dir / "pytorch_train.log"
    return_code = tee_process(command, cwd=Path.cwd(), log_path=log_path)
    if return_code != 0:
        raise RuntimeError(f"PyTorch TVT pipeline failed with code {return_code}. See {log_path}")

    runtime_backend = parse_runtime_backend(log_path)
    if runtime_backend:
        manifest["runtime"] = runtime_backend
        (configs_dir / "tvt_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            "[pytorch-tvt] runtime backend="
            f"{runtime_backend.get('selected_backend', 'unknown')}, "
            f"device={runtime_backend.get('device_detail', 'unknown')}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())