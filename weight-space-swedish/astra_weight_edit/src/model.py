"""Small ReLU MLP and its training loop.

Two hidden layers are required so a lesion can sit in an intermediate map
(fc2) rather than in the output bias. Width is chosen as the smallest candidate
that actually learns the grammar on development stems.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .config import HIDDEN_CANDIDATES, MAX_PARAMETERS, N_HIDDEN_LAYERS, SEED, TRAINING
from .config import N_STEM_BITS
from .grammar import EXPONENTS, FEATURE_NAMES

STEM_SLICE = slice(FEATURE_NAMES.index("stem_0"), FEATURE_NAMES.index("stem_0") + N_STEM_BITS)

N_CLASSES = len(EXPONENTS)


def reset_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class GrammarMLP(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_classes: int, n_hidden_layers: int = N_HIDDEN_LAYERS):
        super().__init__()
        if n_hidden_layers != 2:
            raise ValueError("this experiment fixes two hidden layers so fc2 is an intermediate map")
        self.fc1 = torch.nn.Linear(in_dim, hidden)
        self.fc2 = torch.nn.Linear(hidden, hidden)
        self.out = torch.nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        return self.out(h2)

    def hiddens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        return h1, h2


def parameter_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict({k: v.clone() for k, v in state.items()})


def quantize_state(state: dict[str, torch.Tensor], decimals: int) -> dict[str, torch.Tensor]:
    scale = float(10**decimals)
    return {k: torch.round(v * scale) / scale for k, v in state.items()}


@dataclass
class TrainResult:
    model: GrammarMLP
    hidden: int
    n_parameters: int
    epochs_ran: int
    seconds: float
    history: list[dict]
    dev_example_acc: float
    class_acc: dict[str, float]
    liten_acc: float


def _example_weights(labels: np.ndarray, classes: np.ndarray) -> torch.Tensor:
    """Equal total weight per lexical class, so liten is not drowned out by regular stems."""
    weights = np.zeros(len(labels), dtype=np.float64)
    for lexical_class in np.unique(classes):
        mask = classes == lexical_class
        weights[mask] = 1.0 / mask.sum()
    weights *= len(weights) / weights.sum()
    return torch.tensor(weights, dtype=torch.float32)


def _accuracy(model: GrammarMLP, x: torch.Tensor, y: torch.Tensor) -> float:
    if len(y) == 0:
        return float("nan")
    with torch.no_grad():
        pred = model(x).argmax(dim=-1)
    return float((pred == y).float().mean())


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    class_train: np.ndarray,
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    class_dev: np.ndarray,
    x_liten: np.ndarray,
    y_liten: np.ndarray,
    hidden: int,
    seed: int,
) -> TrainResult:
    reset_seeds(seed)
    model = GrammarMLP(x_train.shape[1], hidden, N_CLASSES)
    if parameter_count(model) > MAX_PARAMETERS:
        raise RuntimeError(f"hidden={hidden} has {parameter_count(model)} parameters, above {MAX_PARAMETERS}")
    opt = torch.optim.Adam(model.parameters(), lr=TRAINING["lr"], weight_decay=TRAINING["weight_decay"])
    xd = torch.tensor(x_dev)
    yd = torch.tensor(y_dev)
    xl = torch.tensor(x_liten)
    yl = torch.tensor(y_liten)
    # A second copy with stem bits removed forces the grammar flags, not the
    # lexeme code, to carry the label. Evaluation still uses the real stem bits.
    x_zero = np.array(x_train, copy=True)
    x_zero[:, STEM_SLICE] = 0.0
    x_fit = np.concatenate([x_train, x_zero], axis=0)
    y_fit = np.concatenate([y_train, y_train], axis=0)
    class_fit = np.concatenate([class_train, class_train], axis=0)
    weights = _example_weights(y_fit, class_fit)
    xt_clean = torch.tensor(x_train)
    yt_clean = torch.tensor(y_train)
    xt = torch.tensor(x_fit)
    yt = torch.tensor(y_fit)
    best_state = clone_state(model)
    best_score = -1.0
    patience = 0
    history: list[dict] = []
    started = time.perf_counter()
    epochs_ran = 0
    if int(y_train.max()) >= N_CLASSES:
        raise RuntimeError("label id exceeds the exponent list")
    for epoch in range(1, TRAINING["max_epochs"] + 1):
        model.train()
        opt.zero_grad()
        logits = model(xt)
        per_row = F.cross_entropy(logits, yt, reduction="none")
        loss = (per_row * weights).mean()
        loss.backward()
        opt.step()
        epochs_ran = epoch
        model.eval()
        dev_acc = _accuracy(model, xd, yd)
        liten_now = _accuracy(model, xl, yl)
        train_acc = _accuracy(model, xt_clean, yt_clean)
        # Both the held-out stems and the single irregular lexeme have to be right.
        score = min(dev_acc, liten_now)
        history.append(
            {
                "epoch": epoch,
                "loss": float(loss.detach()),
                "train_acc": train_acc,
                "dev_acc": dev_acc,
                "liten_acc": liten_now,
                "score": score,
            }
        )
        if score > best_score + 1e-4:
            best_score = score
            best_state = clone_state(model)
            patience = 0
        else:
            patience += 1
        if best_score >= 0.999 and patience >= 15:
            break
        if patience >= TRAINING["patience"]:
            break
    load_state(model, best_state)
    model.eval()
    class_acc = {}
    for lexical_class in sorted(set(class_dev.tolist())):
        mask = class_dev == lexical_class
        class_acc[lexical_class] = _accuracy(model, xd[mask], yd[mask])
    liten_acc = _accuracy(model, xl, yl)
    return TrainResult(
        model=model,
        hidden=hidden,
        n_parameters=parameter_count(model),
        epochs_ran=epochs_ran,
        seconds=time.perf_counter() - started,
        history=history,
        dev_example_acc=float(_accuracy(model, xd, yd)),
        class_acc=class_acc,
        liten_acc=liten_acc,
    )


def meets_learning_target(result: TrainResult) -> bool:
    if result.dev_example_acc < TRAINING["min_dev_example_acc"]:
        return False
    if result.liten_acc < TRAINING["min_liten_acc"]:
        return False
    if any(acc < TRAINING["min_dev_class_acc"] for acc in result.class_acc.values()):
        return False
    return True


def select_and_train(bundles: dict, seed: int = SEED) -> tuple[TrainResult, list[dict]]:
    """Try hidden widths from smallest to largest. Stop at the first one that learns."""
    search = []
    chosen = None
    for hidden in HIDDEN_CANDIDATES:
        result = train_model(
            bundles["x_train"],
            bundles["y_train"],
            bundles["class_train"],
            bundles["x_dev"],
            bundles["y_dev"],
            bundles["class_dev"],
            bundles["x_liten"],
            bundles["y_liten"],
            hidden=hidden,
            seed=seed,
        )
        ok = meets_learning_target(result)
        search.append(
            {
                "hidden": hidden,
                "n_parameters": result.n_parameters,
                "epochs_ran": result.epochs_ran,
                "seconds": round(result.seconds, 3),
                "dev_example_acc": result.dev_example_acc,
                "liten_acc": result.liten_acc,
                "class_acc": result.class_acc,
                "accepted": ok,
            }
        )
        if ok and chosen is None:
            chosen = result
            break
    if chosen is None:
        raise RuntimeError(f"no architecture learned the task: {search}")
    return chosen, search


def predict_labels(model: GrammarMLP, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32))
    return logits.argmax(dim=-1).cpu().numpy().astype(np.int64)


def forward_logits(model: GrammarMLP, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32))
    return logits.cpu().numpy()


def architecture_record(model: GrammarMLP) -> dict:
    return {
        "type": "GrammarMLP",
        "activation": "relu",
        "n_hidden_layers": 2,
        "in_dim": int(model.fc1.in_features),
        "hidden": int(model.fc1.out_features),
        "out_dim": int(model.out.out_features),
        "n_parameters": parameter_count(model),
        "feature_names": list(FEATURE_NAMES),
        "exponents": list(EXPONENTS),
        "tensors": {
            "fc1.weight": list(model.fc1.weight.shape),
            "fc1.bias": list(model.fc1.bias.shape),
            "fc2.weight": list(model.fc2.weight.shape),
            "fc2.bias": list(model.fc2.bias.shape),
            "out.weight": list(model.out.weight.shape),
            "out.bias": list(model.out.bias.shape),
        },
        "seed": SEED,
    }


def copy_model(model: GrammarMLP) -> GrammarMLP:
    return copy.deepcopy(model)
