#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_series(rows: list[dict[str, str]], kind: str, field: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for row in rows:
        if row.get("kind") != kind:
            continue
        value = row.get(field, "")
        step = row.get("step", "")
        if not value or not step:
            continue
        try:
            xs.append(int(step))
            ys.append(float(value))
        except ValueError:
            continue
    return xs, ys


def plot_metric(ax, rows: list[dict[str, str]], title: str, train_field: str | None, eval_field: str | None, ylabel: str) -> None:
    plotted = False
    if train_field is not None:
        x_train, y_train = parse_series(rows, "train", train_field)
        if x_train:
            ax.plot(x_train, y_train, label=f"train:{train_field}", linewidth=1.5)
            plotted = True
    if eval_field is not None:
        x_eval, y_eval = parse_series(rows, "eval", eval_field)
        if x_eval:
            ax.plot(x_eval, y_eval, label=f"eval:{eval_field}", linewidth=1.8)
            plotted = True
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Griffin metrics CSV into a multi-panel PNG.")
    parser.add_argument("metrics_csv", help="Path to metrics CSV produced by train_griffin.py")
    parser.add_argument("--output", default="", help="Output PNG path (default: alongside CSV)")
    args = parser.parse_args()

    metrics_path = Path(args.metrics_csv)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_path}")
    output_path = Path(args.output) if args.output else metrics_path.with_suffix(".png")

    rows = load_rows(metrics_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.ravel()

    plot_metric(axes[0], rows, "Loss Curves", "loss", "val_loss", "Loss")
    plot_metric(axes[1], rows, "BPB Curves", "train_bpb", "val_bpb", "Bits per Byte")
    plot_metric(axes[2], rows, "Throughput", "tok_s", None, "Tokens / sec")
    plot_metric(axes[3], rows, "Step Time / Memory", "step_ms", None, "Step ms")

    x_mem, y_mem = parse_series(rows, "train", "cuda_mem_gb")
    if x_mem:
        mem_ax = axes[3].twinx()
        mem_ax.plot(x_mem, y_mem, color="tab:red", label="train:cuda_mem_gb", linewidth=1.5, alpha=0.8)
        mem_ax.set_ylabel("CUDA Mem (GB)")
        mem_ax.legend(loc="lower right")

    fig.suptitle(metrics_path.stem)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    print(f"saved_plot:{output_path}")


if __name__ == "__main__":
    main()
