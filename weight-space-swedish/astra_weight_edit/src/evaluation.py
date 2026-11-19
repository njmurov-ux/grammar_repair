"""Metrics, function-preserving permutations and figures.

Condition accuracy gives every grammatical cell the same weight. Example accuracy
weights cells by how many stems they contain. Both are reported. Noun identity is
not a factor: each stem contributes one row per cell.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .grammar import CONDITIONS, EXPONENTS
from .model import GrammarMLP, clone_state, copy_model, load_state, predict_labels


def attach_predictions(frame: pd.DataFrame, model: GrammarMLP) -> pd.DataFrame:
    out = frame.copy()
    x = np.stack(out["features"].to_list()).astype(np.float32)
    preds = predict_labels(model, x)
    out["pred"] = preds
    out["pred_exponent"] = [EXPONENTS[i] for i in preds]
    out["correct"] = out["pred"] == out["label"]
    return out


def _cell_table(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (lexical_class, condition), group in scored.groupby(["lexical_class", "condition"]):
        rows.append(
            {
                "lexical_class": lexical_class,
                "condition": condition,
                "is_target": bool(group["is_target"].iloc[0]),
                "n": int(len(group)),
                "accuracy": float(group["correct"].mean()),
                "true_exponent": group["exponent"].iloc[0],
            }
        )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame, model: GrammarMLP, split_label: str) -> dict:
    scored = attach_predictions(frame, model)
    cells = _cell_table(scored)
    target = scored[scored["is_target"]]
    nontarget = scored[~scored["is_target"]]
    target_cells = cells[cells["is_target"]]
    nontarget_cells = cells[~cells["is_target"]]
    by_class = scored.groupby("lexical_class")["correct"].mean().astype(float).to_dict()
    confusion = {}
    if len(target):
        confusion = target.groupby("pred_exponent").size().astype(int).to_dict()
    return {
        "split": split_label,
        "n": int(len(scored)),
        "example_acc": float(scored["correct"].mean()) if len(scored) else float("nan"),
        "macro_condition_acc": float(cells["accuracy"].mean()) if len(cells) else float("nan"),
        "target_acc": float(target["correct"].mean()) if len(target) else float("nan"),
        "target_n": int(len(target)),
        "nontarget_acc": float(nontarget["correct"].mean()) if len(nontarget) else float("nan"),
        "nontarget_macro_acc": float(nontarget_cells["accuracy"].mean()) if len(nontarget_cells) else float("nan"),
        "nontarget_n": int(len(nontarget)),
        "n_conditions": int(len(cells)),
        "by_class": by_class,
        "target_predictions": confusion,
        "cells": cells,
        "scored": scored,
        "target_cell_acc": float(target_cells["accuracy"].mean()) if len(target_cells) else float("nan"),
    }


def state_delta(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> dict:
    n_changed = 0
    sq = 0.0
    abs_sum = 0.0
    per_tensor = {}
    for key in before:
        delta = (after[key] - before[key]).detach().float()
        changed = delta.abs() > 1e-8
        count = int(changed.sum())
        n_changed += count
        sq += float((delta * delta).sum())
        abs_sum += float(delta.abs().sum())
        per_tensor[key] = count
    return {
        "n_changed": n_changed,
        "l2": sq**0.5,
        "mean_abs_changed": abs_sum / n_changed if n_changed else 0.0,
        "per_tensor": per_tensor,
    }


def permute_hidden_neurons(model: GrammarMLP, seed: int) -> tuple[GrammarMLP, dict]:
    """Reorder both hidden layers with matching incoming and outgoing weights.

    For an elementwise ReLU this is an exact functional equivalence, up to
    floating-point rounding.
    """
    generator = torch.Generator().manual_seed(seed)
    hidden = model.fc1.out_features
    perm_h1 = torch.randperm(hidden, generator=generator)
    perm_h2 = torch.randperm(hidden, generator=generator)
    permuted = copy_model(model)
    with torch.no_grad():
        permuted.fc1.weight.copy_(model.fc1.weight[perm_h1])
        permuted.fc1.bias.copy_(model.fc1.bias[perm_h1])
        reindexed = model.fc2.weight[:, perm_h1]
        permuted.fc2.weight.copy_(reindexed[perm_h2])
        permuted.fc2.bias.copy_(model.fc2.bias[perm_h2])
        permuted.out.weight.copy_(model.out.weight[:, perm_h2])
        permuted.out.bias.copy_(model.out.bias)
    return permuted, {"h1": perm_h1.tolist(), "h2": perm_h2.tolist(), "seed": seed}


def permutation_max_abs_diff(model: GrammarMLP, permuted: GrammarMLP, x: np.ndarray) -> float:
    from .model import forward_logits

    a = forward_logits(model, x)
    b = forward_logits(permuted, x)
    return float(np.max(np.abs(a - b)))


def plot_training_curve(history: list[dict], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, ax1 = plt.subplots(figsize=(7.2, 4.0))
    ax1.plot(epochs, [row["loss"] for row in history], color="#1f4e79", label="train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Class-balanced loss")
    ax2 = ax1.twinx()
    ax2.plot(epochs, [row["dev_acc"] for row in history], color="#c47b2b", label="dev accuracy")
    ax2.set_ylabel("Development example accuracy")
    ax2.set_ylim(0, 1.05)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.set_title("Training the Swedish adjective classifier")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_heatmaps(original_cells: pd.DataFrame, corrupted_cells: pd.DataFrame, path: Path) -> None:
    classes = list(dict.fromkeys(original_cells["lexical_class"]))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, cells, title in (
        (axes[0], original_cells, "Original"),
        (axes[1], corrupted_cells, "Corrupted"),
    ):
        grid = (
            cells.pivot(index="lexical_class", columns="condition", values="accuracy")
            .reindex(index=classes, columns=list(CONDITIONS))
        )
        image = ax.imshow(grid.to_numpy(dtype=float), vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(grid.columns)), [c.replace("|", "\n") for c in grid.columns], fontsize=7)
        ax.set_yticks(range(len(grid.index)), list(grid.index))
        ax.set_title(title)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                value = grid.iloc[i, j]
                if pd.notna(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6, color="white")
    fig.colorbar(image, ax=axes, fraction=0.03, pad=0.02, label="Accuracy")
    fig.suptitle("Accuracy by lexical class and grammatical cell")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_repair_bars(rows: list[dict], path: Path) -> None:
    labels = [row["method"] for row in rows]
    target = [row["target_acc"] for row in rows]
    nontarget = [row["nontarget_acc"] for row in rows]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.barh(y - 0.18, target, height=0.36, color="#b85c38", label="Target cell")
    ax.barh(y + 0.18, nontarget, height=0.36, color="#3d6b5a", label="Non-target")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Example accuracy")
    ax.axvline(0.90, color="gray", lw=0.8, ls="--")
    ax.legend(loc="lower right")
    ax.set_title("Local repairs on the held-out test stems")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_cost_estimates(rows: list[dict], path: Path) -> None:
    labels = [row["call"] for row in rows]
    costs = [row["worst_case_usd"] for row in rows]
    colors = ["#4c78a8" if "sol" in row["model"] else "#f58518" for row in rows]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.bar(labels, costs, color=colors)
    ax.set_ylabel("Reserved worst-case cost (USD)")
    ax.set_title("Budget reservation per planned call (nothing spent)")
    ax.tick_params(axis="x", rotation=25)
    running = np.cumsum(costs)
    ax2 = ax.twinx()
    ax2.plot(range(len(running)), running, color="#333333", marker="o", label="cumulative")
    ax2.set_ylabel("Cumulative reservation (USD)")
    ax2.axhline(5.0, color="#a33", ls="--", lw=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_phase2_scatter(rows: list[dict], path: Path, y_max: float = 1.08) -> None:
    """Target accuracy against non-target accuracy. Marker area tracks edited parameters."""
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    colors = {
        "corrupted": "#7a7a7a",
        "random": "#9e9ac8",
        "output_bias": "#74c476",
        "last_layer": "#fd8d3c",
        "gradient": "#3182bd",
        "sol": "#e6550d",
        "astra": "#6a3d9a",
        "astra_perm": "#9e4d9e",
        "column": "#2b8cbe",
    }
    placed = {}
    for row in rows:
        if row.get("target_acc") is None or row.get("nontarget_acc") is None:
            continue
        edited = max(int(row.get("n_changed") or 0), 1)
        color = colors.get(row.get("color_key"), "#333333")
        ax.scatter(
            row["nontarget_acc"],
            row["target_acc"],
            s=40 + edited * 4.5,
            color=color,
            zorder=3,
        )
        key = (round(float(row["nontarget_acc"]), 3), round(float(row["target_acc"]), 3))
        slot = placed.get(key, 0)
        placed[key] = slot + 1
        ax.annotate(
            f"{row['label']} ({int(row.get('n_changed') or 0)})",
            (row["nontarget_acc"], row["target_acc"]),
            textcoords="offset points",
            xytext=(8, 6 + slot * 11),
            fontsize=8,
        )
    ax.axhline(1.0, color="#bbbbbb", lw=0.6)
    ax.axvline(0.98, color="#bbbbbb", lw=0.6, ls="--")
    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(-0.05, y_max)
    ax.set_xlabel("Non-target held-out accuracy")
    ax.set_ylabel("Target held-out accuracy")
    ax.set_title("Target accuracy versus preservation (marker area ~ edited parameters)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def logits_close(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())
