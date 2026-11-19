"""Gradient-free check for duplicated weight columns.

This is a numerical baseline for whether the lesion is visible as a structural
anomaly. It is not shown to Sol or Astra, and a detected pair is not a repair
unless a separate edit is applied and scored.
"""

from __future__ import annotations

import torch

from .grammar import FEATURE_NAMES


def _pair_record(weight: torch.Tensor, left: int, right: int, names: list[str] | None) -> dict:
    column_left = weight[:, left].detach().float()
    column_right = weight[:, right].detach().float()
    delta = column_left - column_right
    norm_left = float(column_left.norm())
    norm_right = float(column_right.norm())
    if norm_left < 1e-12 or norm_right < 1e-12:
        cosine = None
    else:
        cosine = float(torch.dot(column_left, column_right) / (norm_left * norm_right))
    distance = float(delta.norm())
    return {
        "left_index": left,
        "right_index": right,
        "left_name": None if names is None else names[left],
        "right_name": None if names is None else names[right],
        "exact_duplicate": bool(torch.equal(column_left, column_right)),
        "l2_distance": distance,
        "cosine": cosine,
        "left_norm": norm_left,
        "right_norm": norm_right,
        "max_abs_difference": float(delta.abs().max()) if delta.numel() else 0.0,
    }


def analyze_columns(name: str, weight: torch.Tensor, column_names: list[str] | None = None) -> dict:
    matrix = weight.detach().cpu().float()
    if matrix.ndim != 2:
        raise ValueError(f"{name} is not a matrix")
    n_columns = matrix.shape[1]
    names = column_names if column_names is not None and len(column_names) == n_columns else None
    norms = [float(matrix[:, index].norm()) for index in range(n_columns)]
    pairs = [
        _pair_record(matrix, left, right, names)
        for left in range(n_columns)
        for right in range(left + 1, n_columns)
    ]
    exact = [pair for pair in pairs if pair["exact_duplicate"]]
    ranked = sorted(
        pairs,
        key=lambda pair: (
            -(pair["cosine"] if pair["cosine"] is not None else -2.0),
            pair["l2_distance"],
        ),
    )
    finite = [pair["cosine"] for pair in pairs if pair["cosine"] is not None]
    return {
        "tensor": name,
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "column_norms": [
            {"index": index, "name": None if names is None else names[index], "l2_norm": norms[index]}
            for index in range(n_columns)
        ],
        "n_pairs": len(pairs),
        "n_exact_duplicate_pairs": len(exact),
        "exact_duplicate_pairs": exact,
        "median_cosine": None if not finite else float(torch.tensor(finite).median()),
        "max_cosine": None if not finite else max(finite),
        "strongest_pairs": ranked[:8],
    }


def diagnose_network(state: dict[str, torch.Tensor]) -> dict:
    reports = [
        analyze_columns("fc1.weight", state["fc1.weight"], FEATURE_NAMES),
        analyze_columns("fc2.weight", state["fc2.weight"]),
        analyze_columns("out.weight", state["out.weight"]),
    ]
    fc1 = reports[0]
    gender = next(
        (
            pair
            for pair in fc1["exact_duplicate_pairs"]
            if {pair["left_name"], pair["right_name"]} == {"gender_uter", "gender_neuter"}
        ),
        None,
    )
    other_exact = [
        {"tensor": report["tensor"], **pair}
        for report in reports
        for pair in report["exact_duplicate_pairs"]
        if not (
            report["tensor"] == "fc1.weight"
            and {pair["left_name"], pair["right_name"]} == {"gender_uter", "gender_neuter"}
        )
    ]
    return {
        "matrices": reports,
        "gender_columns_exact_duplicate": gender is not None,
        "gender_pair": gender,
        "other_exact_duplicate_pairs": other_exact,
        "intervention_proposed": False,
    }
