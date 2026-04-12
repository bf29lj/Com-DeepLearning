#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def is_tty() -> bool:
    return sys.stdout.isatty()


def render_progress(prefix: str, done: int, total: int, extra: str = "") -> None:
    total = max(total, 1)
    done = min(max(done, 0), total)
    ratio = done / float(total)
    if is_tty():
        width = 24
        fill = int(round(width * ratio))
        bar = "#" * fill + "-" * (width - fill)
        sys.stdout.write(f"\r[{prefix}] [{bar}] {ratio * 100:5.1f}% {done}/{total} {extra}   ")
        sys.stdout.flush()
        if done >= total:
            sys.stdout.write("\n")
            sys.stdout.flush()
    else:
        # Keep CI/log output readable: report roughly every 5% and at completion.
        bucket = int(ratio * 20)
        key = f"__bucket_{prefix}"
        prev = getattr(render_progress, key, -1)
        if done >= total or bucket > prev:
            setattr(render_progress, key, bucket)
            print(f"[{prefix}] {ratio * 100:5.1f}% {done}/{total} {extra}")


def detect_torch_device(mode: str) -> tuple[torch.device, str, str]:
    mode = mode.lower().strip()
    if mode not in {"auto", "cpu", "gpu", "directml"}:
        raise ValueError("--backend must be one of: auto, cpu, gpu, directml")

    if mode == "cpu":
        return torch.device("cpu"), "cpu", "forced-cpu"

    if mode == "directml":
        try:
            import torch_directml  # type: ignore

            dml_device = torch_directml.device()
            return dml_device, "directml", "forced-directml"
        except Exception as exc:
            raise RuntimeError(
                "DirectML backend was requested but torch-directml is not available. "
                "On Windows, AMD GPU support usually requires a Python version with a torch-directml wheel "
                "(commonly Python 3.10/3.11/3.12)."
            ) from exc

    # ROCm builds usually use torch.cuda API with torch.version.hip populated.
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        if getattr(torch.version, "hip", None):
            return torch.device("cuda"), "rocm", name
        return torch.device("cuda"), "cuda", name

    # Optional fallback for AMD on Windows via torch-directml.
    if mode in {"auto", "gpu"}:
        try:
            import torch_directml  # type: ignore

            dml_device = torch_directml.device()
            return dml_device, "directml", "AMD/Intel via DirectML"
        except Exception:
            pass

    if mode == "gpu":
        raise RuntimeError(
            "--backend gpu was requested but no GPU backend is available. "
            "Install CUDA PyTorch (NVIDIA), ROCm PyTorch (AMD on Linux), "
            "or use a Python version that has a torch-directml wheel for AMD on Windows."
        )

    return torch.device("cpu"), "cpu", "fallback-cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch LSTM trainer equivalent to C++ trainer")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, default=None)
    parser.add_argument("--test-csv", type=Path, default=None)
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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-decay", type=float, default=1.0)
    parser.add_argument("--lr-decay-every", type=int, default=1)
    parser.add_argument("--min-lr", type=float, default=0.0)

    parser.add_argument("--results-csv", type=Path, default=None)
    parser.add_argument("--pr-csv", type=Path, default=None)
    parser.add_argument("--pr-min", type=float, default=0.0)
    parser.add_argument("--pr-max", type=float, default=1.0)
    parser.add_argument("--pr-step", type=float, default=0.02)
    parser.add_argument("--save-model", type=Path, default=None)
    parser.add_argument("--load-model", type=Path, default=None)
    parser.add_argument("--timeout-sec", type=float, default=0.0)
    parser.add_argument("--auto-class-weights", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


def count_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        next(handle, None)
        return sum(1 for line in handle if line.strip())


def load_csv(csv_path: Path, stage: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    total = count_rows(csv_path)
    features: list[list[float]] = []
    labels: list[float] = []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None or len(header) < 2:
            raise RuntimeError(f"Invalid CSV header: {csv_path}")
        feature_names = header[:-1]

        for idx, row in enumerate(reader, start=1):
            if not row:
                continue
            vals = [float(v) for v in row]
            features.append(vals[:-1])
            labels.append(vals[-1])
            if idx % 256 == 0 or idx >= total:
                render_progress(stage, idx, total)

    if not features:
        raise RuntimeError(f"No data rows in: {csv_path}")

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    return x, y, feature_names


class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, x: np.ndarray, y: np.ndarray, seq_len: int):
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if x.shape[0] < seq_len:
            raise ValueError("dataset rows are smaller than seq_len")
        self.x = x
        self.y = y
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.x.shape[0] - self.seq_len + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_seq = self.x[idx : idx + self.seq_len]
        y_label = self.y[idx + self.seq_len - 1]
        return torch.from_numpy(x_seq), torch.tensor(y_label, dtype=torch.float32)


class LstmBinaryClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, use_manual_lstm: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_manual_lstm = use_manual_lstm
        if use_manual_lstm:
            self.x2h = nn.Linear(input_dim, 4 * hidden_dim)
            self.h2h = nn.Linear(hidden_dim, 4 * hidden_dim, bias=False)
            self.lstm = None
        else:
            self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
            self.x2h = None
            self.h2h = None
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_manual_lstm:
            batch_size, seq_len, _ = x.shape
            h = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
            c = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
            assert self.x2h is not None and self.h2h is not None
            for t in range(seq_len):
                gates = self.x2h(x[:, t, :]) + self.h2h(h)
                i, f, g, o = gates.chunk(4, dim=-1)
                i = torch.sigmoid(i)
                f = torch.sigmoid(f)
                g = torch.tanh(g)
                o = torch.sigmoid(o)
                c = f * c + i * g
                h = o * torch.tanh(c)
            logits = self.fc(h).squeeze(-1)
            return logits

        assert self.lstm is not None
        out, _ = self.lstm(x)
        logits = self.fc(out[:, -1, :]).squeeze(-1)
        return logits


@dataclass
class Metrics:
    tp: int
    fp: int
    tn: int
    fn: int
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float


def safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else a / b


def compute_metrics(prob: np.ndarray, label: np.ndarray, threshold: float) -> Metrics:
    pred = (prob >= threshold).astype(np.int32)
    y = label.astype(np.int32)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))

    acc = safe_div(tp + tn, tp + fp + tn + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return Metrics(tp, fp, tn, fn, acc, precision, recall, specificity, f1)


def compute_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_name: str,
    pos_weight: float,
    neg_weight: float,
    focal_gamma: float,
    focal_alpha: float,
) -> torch.Tensor:
    loss_name = loss_name.lower()
    if loss_name == "mse":
        pred = torch.sigmoid(logits)
        base = (pred - target) ** 2
        sample_w = target * pos_weight + (1.0 - target) * neg_weight
        return (base * sample_w).mean()

    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    sample_w = target * pos_weight + (1.0 - target) * neg_weight

    if loss_name == "bce":
        return (bce * sample_w).mean()

    # focal
    p = torch.sigmoid(logits)
    pt = target * p + (1.0 - target) * (1.0 - p)
    alpha_t = target * focal_alpha + (1.0 - target) * (1.0 - focal_alpha)
    focal = alpha_t * torch.pow((1.0 - pt).clamp(min=1e-8), focal_gamma) * bce
    return (focal * sample_w).mean()


def build_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    optim = args.optimizer.lower()
    if optim == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.lr)
    if optim == "momentum":
        return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    if optim == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
        )
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )


def set_current_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    stage: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    seen = 0
    total = len(loader.dataset)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = compute_loss(logits, y, args.loss, args.pos_weight, args.neg_weight, args.focal_gamma, args.focal_alpha)
            losses.append(float(loss.item()))

            p = torch.sigmoid(logits).detach().cpu().numpy()
            probs.append(p)
            labels.append(y.detach().cpu().numpy())
            seen += y.shape[0]
            render_progress(stage, seen, total, f"cost={np.mean(losses):.6f}")

    return float(np.mean(losses) if losses else 0.0), np.concatenate(probs), np.concatenate(labels)


def write_results_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "phase",
            "epoch",
            "train_loss",
            "eval_cost",
            "tp",
            "fp",
            "tn",
            "fn",
            "accuracy",
            "precision",
            "recall",
            "specificity",
            "f1",
            "elapsed_ms",
        ])


def append_results_row(path: Path, phase: str, epoch: int, train_loss: float, eval_cost: float, m: Metrics, elapsed_ms: int) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            phase,
            epoch,
            f"{train_loss:.6f}",
            f"{eval_cost:.6f}",
            m.tp,
            m.fp,
            m.tn,
            m.fn,
            f"{m.accuracy:.6f}",
            f"{m.precision:.6f}",
            f"{m.recall:.6f}",
            f"{m.specificity:.6f}",
            f"{m.f1:.6f}",
            elapsed_ms,
        ])


def write_pr_csv(path: Path, prob: np.ndarray, label: np.ndarray, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    thresholds = np.arange(args.pr_min, args.pr_max + 1e-9, args.pr_step, dtype=np.float32)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "precision", "recall", "f1", "specificity", "accuracy", "tp", "fp", "tn", "fn"])
        for t in thresholds:
            m = compute_metrics(prob, label, float(t))
            writer.writerow([
                f"{float(t):.6f}",
                f"{m.precision:.6f}",
                f"{m.recall:.6f}",
                f"{m.f1:.6f}",
                f"{m.specificity:.6f}",
                f"{m.accuracy:.6f}",
                m.tp,
                m.fp,
                m.tn,
                m.fn,
            ])


def main() -> int:
    args = parse_args()
    if args.timeout_sec < 0.0:
        raise ValueError("timeout-sec must be non-negative")

    torch.manual_seed(42)
    np.random.seed(42)

    device, backend_name, device_name = detect_torch_device(args.backend)
    print(f"PyTorch: {torch.__version__}")
    print(f"Selected backend: {backend_name}")
    print(f"Selected device: {device}")
    print(f"Device detail: {device_name}")

    x_train, y_train, feature_names = load_csv(args.train_csv, "data-train")
    if args.val_csv is not None:
        x_val, y_val, _ = load_csv(args.val_csv, "data-val")
    else:
        x_val, y_val = x_train, y_train

    if args.auto_class_weights:
        pos_count = float(np.sum(y_train >= 0.5))
        neg_count = float(y_train.shape[0] - pos_count)
        if pos_count > 0 and neg_count > 0:
            # Balance class contributions inversely proportional to class frequency.
            args.pos_weight = neg_count / pos_count
            args.neg_weight = 1.0
            print(
                "Auto class weights enabled: "
                f"pos_weight={args.pos_weight:.6f}, neg_weight={args.neg_weight:.6f}"
            )
        else:
            print("Auto class weights skipped because one class has zero samples in train data")

    train_ds = SequenceDataset(x_train, y_train, args.seq_len)
    val_ds = SequenceDataset(x_val, y_val, args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    use_manual_lstm = backend_name == "directml"
    if use_manual_lstm:
        print("DirectML workaround enabled: using manual LSTM cell implementation")
    model = LstmBinaryClassifier(
        input_dim=len(feature_names),
        hidden_dim=args.lstm_hidden,
        use_manual_lstm=use_manual_lstm,
    ).to(device)
    if args.load_model is not None and args.load_model.exists():
        state = torch.load(args.load_model, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded model: {args.load_model}")

    optimizer = build_optimizer(model, args)

    if args.results_csv is not None:
        write_results_header(args.results_csv)

    # Initial evaluation
    t0 = time.perf_counter()
    init_cost, init_prob, init_label = evaluate(model, val_loader, device, args, "eval")
    init_m = compute_metrics(init_prob, init_label, args.threshold)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    print(
        f"Initial cost={init_cost:.6f}, acc={init_m.accuracy:.6f}, "
        f"precision={init_m.precision:.6f}, recall={init_m.recall:.6f}, f1={init_m.f1:.6f}"
    )
    if args.results_csv is not None:
        append_results_row(args.results_csv, "initial", 0, 0.0, init_cost, init_m, elapsed_ms)

    if not args.eval_only:
        train_start = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            if args.timeout_sec > 0.0 and (time.perf_counter() - train_start) >= args.timeout_sec:
                print(f"Training stopped due to timeout-sec={args.timeout_sec:.3f}")
                break

            decay_steps = (epoch - 1) // max(1, args.lr_decay_every)
            lr = args.lr * (args.lr_decay ** decay_steps)
            lr = max(lr, args.min_lr)
            set_current_lr(optimizer, lr)

            model.train()
            batch_losses: list[float] = []
            seen = 0
            total = len(train_loader.dataset)
            start_epoch = time.perf_counter()

            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = compute_loss(logits, y, args.loss, args.pos_weight, args.neg_weight, args.focal_gamma, args.focal_alpha)
                loss.backward()
                optimizer.step()

                batch_losses.append(float(loss.item()))
                seen += y.shape[0]
                render_progress(
                    f"Epoch {epoch}/{args.epochs}",
                    seen,
                    total,
                    f"loss={np.mean(batch_losses):.6f} lr={lr:.6g}",
                )

            train_loss = float(np.mean(batch_losses) if batch_losses else 0.0)
            train_ms = int((time.perf_counter() - start_epoch) * 1000)

            should_log = (epoch % max(1, args.print_every) == 0) or (epoch == args.epochs)
            if should_log:
                eval_start = time.perf_counter()
                eval_cost, prob, label = evaluate(model, val_loader, device, args, "eval")
                m = compute_metrics(prob, label, args.threshold)
                eval_ms = int((time.perf_counter() - eval_start) * 1000)
                print(
                    f"[Epoch {epoch}/{args.epochs}] train_loss={train_loss:.6f}, lr={lr:.6g}, "
                    f"eval_cost={eval_cost:.6f}, train_time={train_ms} ms, eval_time={eval_ms} ms"
                )
                print(
                    f"Confusion matrix: TP={m.tp}, FP={m.fp}, TN={m.tn}, FN={m.fn}\n"
                    f"Accuracy={m.accuracy:.6f}, Precision={m.precision:.6f}, Recall={m.recall:.6f}, "
                    f"Specificity={m.specificity:.6f}, F1={m.f1:.6f}"
                )
                if args.results_csv is not None:
                    append_results_row(args.results_csv, "epoch", epoch, train_loss, eval_cost, m, eval_ms)

    # Final eval for PR / summary
    final_cost, final_prob, final_label = evaluate(model, val_loader, device, args, "eval")
    final_m = compute_metrics(final_prob, final_label, args.threshold)
    print(
        f"Final eval cost={final_cost:.6f}, acc={final_m.accuracy:.6f}, "
        f"precision={final_m.precision:.6f}, recall={final_m.recall:.6f}, f1={final_m.f1:.6f}"
    )

    if args.pr_csv is not None:
        write_pr_csv(args.pr_csv, final_prob, final_label, args)
        print(f"PR CSV saved: {args.pr_csv}")

    if args.save_model is not None:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save_model)
        print(f"Model saved: {args.save_model}")

    if args.test_csv is not None and args.test_csv.exists():
        x_test, y_test, _ = load_csv(args.test_csv, "data-test")
        test_ds = SequenceDataset(x_test, y_test, args.seq_len)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
        test_cost, test_prob, test_label = evaluate(model, test_loader, device, args, "test")
        test_m = compute_metrics(test_prob, test_label, args.threshold)
        print(
            f"Test eval cost={test_cost:.6f}, acc={test_m.accuracy:.6f}, "
            f"precision={test_m.precision:.6f}, recall={test_m.recall:.6f}, f1={test_m.f1:.6f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
