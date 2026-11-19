"""Inexpensive repairs used as a difficulty gate before any API call.

Bias and last-layer edits may touch every parameter of that slice. Gradient
repair may touch only MAX_EDIT_PARAMS entries, the same budget later offered
to Astra and Sol. Random edits match that count and use a fixed Gaussian scale.
None of these methods sees test stems.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .config import BASELINES, MAX_EDIT_PARAMS, SEED
from .model import GrammarMLP, clone_state, copy_model, load_state


def _xy(x: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.int64)


def _optimize(model: GrammarMLP, x: torch.Tensor, y: torch.Tensor, steps: int, lr: float, mask_grad) -> GrammarMLP:
    trainable = [param for param in model.parameters() if param.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters")
    opt = torch.optim.Adam(trainable, lr=lr)
    best_state = clone_state(model)
    best_loss = float("inf")
    anchors = {id(param): param.detach().clone() for param in model.parameters()}
    for _ in range(steps):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        with torch.no_grad():
            for param in trainable:
                if param.grad is None:
                    continue
                param.grad.mul_(mask_grad(param))
        opt.step()
        with torch.no_grad():
            for param in model.parameters():
                mask = mask_grad(param)
                param.copy_(torch.where(mask > 0, param, anchors[id(param)]))
        value = float(loss.detach())
        if value < best_loss - 1e-8:
            best_loss = value
            best_state = clone_state(model)
    load_state(model, best_state)
    model.eval()
    return model


def repair_output_bias(model: GrammarMLP, x: np.ndarray, y: np.ndarray, seed: int = SEED) -> GrammarMLP:
    torch.manual_seed(seed)
    edited = copy_model(model)
    for name, param in edited.named_parameters():
        param.requires_grad_(name == "out.bias")
    xt, yt = _xy(x, y)

    def mask(param: torch.Tensor) -> torch.Tensor:
        owner = {id(p): n for n, p in edited.named_parameters()}
        if owner[id(param)] == "out.bias":
            return torch.ones_like(param)
        return torch.zeros_like(param)

    return _optimize(edited, xt, yt, BASELINES["bias_steps"], BASELINES["bias_lr"], mask)


def repair_last_layer(model: GrammarMLP, x: np.ndarray, y: np.ndarray, seed: int = SEED) -> GrammarMLP:
    torch.manual_seed(seed)
    edited = copy_model(model)
    allowed = {"out.weight", "out.bias"}
    for name, param in edited.named_parameters():
        param.requires_grad_(name in allowed)
    xt, yt = _xy(x, y)
    owner = {}

    def mask(param: torch.Tensor) -> torch.Tensor:
        if not owner:
            owner.update({id(p): n for n, p in edited.named_parameters()})
        if owner[id(param)] in allowed:
            return torch.ones_like(param)
        return torch.zeros_like(param)

    return _optimize(edited, xt, yt, BASELINES["last_layer_steps"], BASELINES["last_layer_lr"], mask)


def repair_gradient_budget(model: GrammarMLP, x: np.ndarray, y: np.ndarray, seed: int = SEED, k: int = MAX_EDIT_PARAMS) -> tuple[GrammarMLP, list[dict]]:
    """Descend on the k weights with the largest initial diagnostic gradient."""
    torch.manual_seed(seed)
    edited = copy_model(model)
    xt, yt = _xy(x, y)
    edited.train()
    edited.zero_grad()
    loss = F.cross_entropy(edited(xt), yt)
    loss.backward()
    scored: list[tuple[float, str, tuple[int, ...]]] = []
    for name, param in edited.named_parameters():
        if param.grad is None:
            continue
        flat = param.grad.detach().abs().reshape(-1)
        values, indices = torch.topk(flat, min(k, flat.numel()))
        for value, index in zip(values.tolist(), indices.tolist()):
            unravel = np.unravel_index(index, tuple(param.shape))
            scored.append((value, name, tuple(int(i) for i in unravel)))
    scored.sort(reverse=True)
    chosen = scored[:k]
    chosen_keys = {(name, idx) for _, name, idx in chosen}
    locations = [{"tensor": name, "index": list(idx), "abs_grad": value} for value, name, idx in chosen]

    fresh = copy_model(model)
    masks = {}
    for name, param in fresh.named_parameters():
        mask = torch.zeros_like(param)
        for picked_name, idx in chosen_keys:
            if picked_name == name:
                mask[idx] = 1.0
        masks[id(param)] = mask

    def mask(param: torch.Tensor) -> torch.Tensor:
        return masks[id(param)]

    load_ready = _optimize(fresh, xt, yt, BASELINES["gradient_steps"], BASELINES["gradient_lr"], mask)
    return load_ready, locations


def _project_edit(model: GrammarMLP, anchors: dict[str, torch.Tensor], masks: dict[str, torch.Tensor], max_l2: float, max_abs_delta: float, max_abs_value: float) -> None:
    """Project the update onto the API edit set: support, per-entry cap, L2 cap, value cap."""
    packed = []
    for name, param in model.named_parameters():
        delta = (param.detach() - anchors[name]) * masks[name]
        delta = delta.clamp(-max_abs_delta, max_abs_delta)
        packed.append((name, param, delta))
    square = sum(float((delta * delta).sum()) for _, _, delta in packed)
    norm = square**0.5
    factor = 1.0 if norm <= max_l2 + 1e-12 or norm == 0.0 else max_l2 / norm
    with torch.no_grad():
        for name, param, delta in packed:
            updated = (anchors[name] + delta * factor).clamp(-max_abs_value, max_abs_value)
            param.copy_(torch.where(masks[name] > 0, updated, anchors[name]))


def repair_gradient_constrained(
    model: GrammarMLP,
    x: np.ndarray,
    y: np.ndarray,
    seed: int = SEED,
    k: int = MAX_EDIT_PARAMS,
    max_l2: float = 6.0,
    max_abs_delta: float = 3.0,
    max_abs_value: float = 8.0,
) -> tuple[GrammarMLP, list[dict], dict]:
    """Same top-k diagnostic coordinates as the unconstrained repair, then project every step.

    Coordinates are chosen from the initial diagnostic gradient only. Held-out labels
    are not used. The projection enforces the LLM edit caps.
    """
    torch.manual_seed(seed)
    edited = copy_model(model)
    xt, yt = _xy(x, y)
    edited.train()
    edited.zero_grad()
    loss = F.cross_entropy(edited(xt), yt)
    loss.backward()
    scored: list[tuple[float, str, tuple[int, ...]]] = []
    for name, param in edited.named_parameters():
        if param.grad is None:
            continue
        flat = param.grad.detach().abs().reshape(-1)
        values, indices = torch.topk(flat, min(k, flat.numel()))
        for value, index in zip(values.tolist(), indices.tolist()):
            unravel = np.unravel_index(index, tuple(param.shape))
            scored.append((float(value), name, tuple(int(i) for i in unravel)))
    scored.sort(reverse=True)
    chosen = scored[:k]
    chosen_keys = {(name, idx) for _, name, idx in chosen}
    locations = [{"tensor": name, "index": list(idx), "abs_grad": value} for value, name, idx in chosen]

    fresh = copy_model(model)
    anchors = {name: param.detach().clone() for name, param in fresh.named_parameters()}
    masks = {}
    for name, param in fresh.named_parameters():
        mask = torch.zeros_like(param)
        for picked_name, idx in chosen_keys:
            if picked_name == name:
                mask[idx] = 1.0
        masks[name] = mask
        param.requires_grad_(True)
    opt = torch.optim.Adam(fresh.parameters(), lr=BASELINES["gradient_lr"])
    best_state = clone_state(fresh)
    best_loss = float("inf")
    best_step = 0
    for step in range(1, BASELINES["gradient_steps"] + 1):
        fresh.train()
        opt.zero_grad()
        loss = F.cross_entropy(fresh(xt), yt)
        loss.backward()
        with torch.no_grad():
            for name, param in fresh.named_parameters():
                if param.grad is not None:
                    param.grad.mul_(masks[name])
        opt.step()
        _project_edit(fresh, anchors, masks, max_l2, max_abs_delta, max_abs_value)
        fresh.eval()
        with torch.no_grad():
            value = float(F.cross_entropy(fresh(xt), yt))
        if value < best_loss - 1e-8:
            best_loss = value
            best_step = step
            best_state = clone_state(fresh)
    load_state(fresh, best_state)
    fresh.eval()
    info = {"steps": BASELINES["gradient_steps"], "best_step": best_step, "best_diagnostic_loss": best_loss, "lr": BASELINES["gradient_lr"]}
    return fresh, locations, info


def duplicate_column_coordinates(model: GrammarMLP) -> list[tuple[str, tuple[int, ...]]]:
    """Entries of fc1 input columns that are exact duplicates of another column."""
    weight = model.fc1.weight.detach()
    duplicates = set()
    for left in range(weight.shape[1]):
        for right in range(left + 1, weight.shape[1]):
            if torch.equal(weight[:, left], weight[:, right]):
                duplicates.add(left)
                duplicates.add(right)
    coordinates = []
    for column in sorted(duplicates):
        for row in range(weight.shape[0]):
            coordinates.append(("fc1.weight", (row, column)))
    return coordinates


def repair_detected_columns(
    model: GrammarMLP,
    x: np.ndarray,
    y: np.ndarray,
    seed: int = SEED,
    k: int = MAX_EDIT_PARAMS,
) -> tuple[GrammarMLP, list[dict], dict]:
    """Projected repair whose support is restricted to detected duplicate columns.

    This is iterative diagnostic optimization, not a one-shot deduction. The original
    weights are not used. Coordinates are the k largest initial gradients inside the
    duplicate columns.
    """
    torch.manual_seed(seed)
    eligible = duplicate_column_coordinates(model)
    if not eligible:
        raise RuntimeError("no exact duplicate columns to repair")
    edited = copy_model(model)
    xt, yt = _xy(x, y)
    edited.train()
    edited.zero_grad()
    loss = F.cross_entropy(edited(xt), yt)
    loss.backward()
    params = dict(edited.named_parameters())
    scored = []
    for name, index in eligible:
        value = float(params[name].grad.detach().abs()[index])
        scored.append((value, name, index))
    scored.sort(reverse=True)
    chosen = scored[:k]
    chosen_keys = {(name, idx) for _, name, idx in chosen}
    locations = [{"tensor": name, "index": list(idx), "abs_grad": value} for value, name, idx in chosen]
    fresh = copy_model(model)
    anchors = {name: param.detach().clone() for name, param in fresh.named_parameters()}
    masks = {}
    for name, param in fresh.named_parameters():
        mask = torch.zeros_like(param)
        for picked_name, idx in chosen_keys:
            if picked_name == name:
                mask[idx] = 1.0
        masks[name] = mask
    opt = torch.optim.Adam(fresh.parameters(), lr=BASELINES["gradient_lr"])
    best_state = clone_state(fresh)
    best_loss = float("inf")
    best_step = 0
    from .config import EDIT_LIMITS

    for step in range(1, BASELINES["gradient_steps"] + 1):
        fresh.train()
        opt.zero_grad()
        loss = F.cross_entropy(fresh(xt), yt)
        loss.backward()
        with torch.no_grad():
            for name, param in fresh.named_parameters():
                if param.grad is not None:
                    param.grad.mul_(masks[name])
        opt.step()
        _project_edit(fresh, anchors, masks, EDIT_LIMITS["max_frobenius"], EDIT_LIMITS["max_abs_delta"], EDIT_LIMITS["max_abs_value"])
        fresh.eval()
        with torch.no_grad():
            value = float(F.cross_entropy(fresh(xt), yt))
        if value < best_loss - 1e-8:
            best_loss = value
            best_step = step
            best_state = clone_state(fresh)
    load_state(fresh, best_state)
    fresh.eval()
    info = {
        "steps": BASELINES["gradient_steps"],
        "best_step": best_step,
        "best_diagnostic_loss": best_loss,
        "eligible_entries": len(eligible),
        "method": "projected Adam on the 40 largest diagnostic gradients inside exact duplicate fc1 columns",
    }
    return fresh, locations, info


def random_hidden_edits(
    model: GrammarMLP,
    k: int = MAX_EDIT_PARAMS,
    std: float | None = None,
    n_seeds: int | None = None,
) -> list[GrammarMLP]:
    std = BASELINES["random_std"] if std is None else std
    n_seeds = BASELINES["random_seeds"] if n_seeds is None else n_seeds
    seed_list = [SEED + 1000 + i for i in range(n_seeds)]
    eligible: list[tuple[str, tuple[int, ...]]] = []
    for name, param in model.named_parameters():
        if name in BASELINES["random_tensors"]:
            for index in np.ndindex(tuple(param.shape)):
                eligible.append((name, tuple(int(i) for i in index)))
    if len(eligible) < k:
        raise RuntimeError(f"only {len(eligible)} editable weights for a budget of {k}")
    edited_models = []
    for seed in seed_list:
        rng = np.random.RandomState(seed)
        pick = rng.choice(len(eligible), size=k, replace=False)
        noise = rng.normal(0.0, std, size=k)
        cloned = copy_model(model)
        params = dict(cloned.named_parameters())
        with torch.no_grad():
            for slot, delta in zip(pick, noise):
                name, index = eligible[int(slot)]
                params[name][index] += float(delta)
        cloned.eval()
        edited_models.append(cloned)
    return edited_models
