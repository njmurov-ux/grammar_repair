"""Selective hidden-layer corruptions.

Every candidate is a change to fc1 or fc2, never to the output bias. Directions
and neuron scores are estimated on training stems only. Development rows are
used later, by the pipeline, to accept or reject a candidate. Test stems are
not read here.

The families are pre-registered in config.PROTOCOL / CORRUPTION.
"""

from __future__ import annotations

import numpy as np
import torch

from .config import CORRUPTION
from .grammar import FEATURE_NAMES
from .model import GrammarMLP, clone_state


def _as_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.asarray(x), dtype=torch.float32)


def hidden_pre_and_post(model: GrammarMLP, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fc1 preactivation, post-ReLU h1, and post-ReLU h2."""
    model.eval()
    with torch.no_grad():
        raw = model.fc1(_as_tensor(x))
        h1 = torch.relu(raw)
        h2 = torch.relu(model.fc2(h1))
    return raw.numpy(), h1.numpy(), h2.numpy()


def _mean_rows(h: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        raise ValueError("empty activation mask")
    return h[mask].mean(axis=0)


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < 1e-8:
        raise ValueError("direction has no magnitude")
    return v / norm


def selective_directions(h1: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    target = _mean_rows(h1, masks["target"])
    uter = _mean_rows(h1, masks["gem_uter"])
    other_neuter = _mean_rows(h1, masks["other_neuter"])
    other_uter = _mean_rows(h1, masks["other_uter"])
    raw = target - uter
    gender = other_neuter - other_uter
    # Remove the component that also separates neuter from uter outside the target class.
    gender_unit = gender / max(float(np.linalg.norm(gender)), 1e-8)
    orthogonal = raw - np.dot(raw, gender_unit) * gender_unit
    return {
        "target_minus_uter": _unit(raw),
        "target_minus_uter_orth_gender": _unit(orthogonal if np.linalg.norm(orthogonal) > 1e-6 else raw),
    }


def neuron_ranking(h1: np.ndarray, masks: dict[str, np.ndarray], score_name: str) -> np.ndarray:
    target = _mean_rows(h1, masks["target"])
    uter = _mean_rows(h1, masks["gem_uter"])
    other_neuter = _mean_rows(h1, masks["other_neuter"])
    if score_name == "target_vs_uter":
        score = target - uter
    elif score_name == "contrast_penalized":
        score = (target - uter) - np.maximum(other_neuter, 0.0)
    else:
        raise ValueError(score_name)
    # Prefer units that actually fire on the target cell.
    score = np.where(target > 0.05, score, -1e9)
    return np.argsort(-score)


def _scale_columns(state: dict[str, torch.Tensor], columns: np.ndarray, factor: float) -> dict[str, torch.Tensor]:
    new = {k: v.clone() for k, v in state.items()}
    weight = new["fc2.weight"]
    weight[:, columns] *= factor
    return new


def _noise_columns(
    state: dict[str, torch.Tensor], columns: np.ndarray, std: float, seed: int
) -> dict[str, torch.Tensor]:
    new = {k: v.clone() for k, v in state.items()}
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(new["fc2.weight"].shape[0], len(columns), generator=generator) * std
    new["fc2.weight"][:, columns] += noise
    return new


def _dampen_direction(state: dict[str, torch.Tensor], direction: np.ndarray, alpha: float) -> dict[str, torch.Tensor]:
    """W2 <- W2 (I - alpha d d^T). Information in h1 along d is attenuated before fc2."""
    new = {k: v.clone() for k, v in state.items()}
    d = torch.tensor(direction, dtype=torch.float32)
    weight = new["fc2.weight"]
    update = torch.outer(weight @ d, d)  # (W d) d^T
    new["fc2.weight"] = weight - alpha * update
    return new


def _stem_threshold(
    state: dict[str, torch.Tensor],
    h1: np.ndarray,
    masks: dict[str, np.ndarray],
    stem_slice: slice,
    scale: float,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Nudge the stem-bit synapses of the most target-selective neuron.

    Stem codes differ across lexemes, so the same grammatical cell can cross a
    threshold for some held-in stems and not others. The neuron is chosen because
    it fires on the target cell and stays quiet on the uter contrast; a small
    preactivation shift then hits the target more than the other cells.
    """
    ranking = neuron_ranking(h1, masks, "contrast_penalized")
    neuron = int(ranking[0])
    if not np.isfinite(neuron):
        raise RuntimeError("no selective neuron")
    generator = np.random.RandomState(seed)
    n_bits = stem_slice.stop - stem_slice.start
    signs = generator.choice(np.array([-1.0, 1.0]), size=n_bits)
    new = {k: v.clone() for k, v in state.items()}
    with torch.no_grad():
        block = new["fc1.weight"][neuron, stem_slice]
        block += torch.tensor(signs * scale, dtype=torch.float32)
    return new


def build_masks(frame_flags: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return frame_flags


def generate_candidates(
    model: GrammarMLP,
    x_train: np.ndarray,
    masks: dict[str, np.ndarray],
    stem_slice: slice,
) -> list[dict]:
    """Return corrupted state dicts. Does not evaluate them."""
    state = clone_state(model)
    _, h1, _ = hidden_pre_and_post(model, x_train)
    try:
        directions = selective_directions(h1, masks)
    except ValueError:
        directions = {}
    rankings = {
        name: neuron_ranking(h1, masks, name) for name in CORRUPTION["column_scale"]["scores"]
    }
    candidates = []
    index = 0

    def add(family: str, new_state: dict, detail: dict) -> None:
        nonlocal index
        candidates.append(
            {
                "candidate_index": index,
                "family": family,
                "detail": detail,
                "state": new_state,
            }
        )
        index += 1

    for score_name, order in rankings.items():
        usable = [int(i) for i in order if np.isfinite(h1[:, int(i)].mean())]
        for m in CORRUPTION["column_scale"]["m"]:
            columns = np.asarray(usable[:m], dtype=np.int64)
            for factor in CORRUPTION["column_scale"]["factors"]:
                add(
                    "column_scale",
                    _scale_columns(state, columns, factor),
                    {"score": score_name, "m": m, "factor": factor, "neurons": columns.tolist()},
                )
        for m in CORRUPTION["column_noise"]["m"]:
            columns = np.asarray(usable[:m], dtype=np.int64)
            for std in CORRUPTION["column_noise"]["stds"]:
                for noise_seed in CORRUPTION["column_noise"]["seeds"]:
                    add(
                        "column_noise",
                        _noise_columns(state, columns, std, noise_seed + 17),
                        {"score": score_name, "m": m, "std": std, "seed": noise_seed, "neurons": columns.tolist()},
                    )

    for kind, direction in directions.items():
        for alpha in CORRUPTION["direction_dampen"]["alphas"]:
            add(
                "direction_dampen",
                _dampen_direction(state, direction, alpha),
                {"kind": kind, "alpha": alpha},
            )

    for scale in CORRUPTION["stem_threshold"]["scales"]:
        for stem_seed in CORRUPTION["stem_threshold"]["seeds"]:
            try:
                new_state = _stem_threshold(state, h1, masks, stem_slice, scale, stem_seed + 100)
            except (RuntimeError, ValueError):
                continue
            add(
                "stem_threshold",
                new_state,
                {"scale": scale, "seed": stem_seed, "neuron": int(rankings["contrast_penalized"][0])},
            )

    # Protocol 2. Replacing the neuter input column with the uter column makes
    # every neuter phrase share the hidden state of its uter counterpart.
    uter_col = FEATURE_NAMES.index("gender_uter")
    neuter_col = FEATURE_NAMES.index("gender_neuter")
    copied = {key: value.clone() for key, value in state.items()}
    copied["fc1.weight"][:, neuter_col] = copied["fc1.weight"][:, uter_col].clone()
    add(
        "gender_column_copy",
        copied,
        {"source_feature": "gender_uter", "destination_feature": "gender_neuter", "column": neuter_col},
    )
    return candidates


def train_masks(is_target: np.ndarray, is_gem_uter: np.ndarray, is_other_neuter: np.ndarray, is_other_uter: np.ndarray) -> dict[str, np.ndarray]:
    masks = {
        "target": is_target.astype(bool),
        "gem_uter": is_gem_uter.astype(bool),
        "other_neuter": is_other_neuter.astype(bool),
        "other_uter": is_other_uter.astype(bool),
    }
    for name, mask in masks.items():
        if mask.sum() == 0:
            raise AssertionError(f"corruption mask {name} is empty")
    return masks
