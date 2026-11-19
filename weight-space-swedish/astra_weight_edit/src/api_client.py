"""Prompt construction, edit validation and the API budget gate.

No function in this module contacts OpenAI unless execute_edit is called with
api_enabled=True. Phase 1 uses the prompt, the token counter and the mocked
budget tests only. The key is loaded inside execute_edit and is never printed.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from .config import API, EDIT_LIMITS, TENSOR_NAMES
from .grammar import CLASS_FEATURES, EXPONENTS, FEATURE_NAMES, GRAMMAR_FEATURES

LEDGER_COLUMNS = [
    "timestamp",
    "call_id",
    "model",
    "experiment",
    "estimated_input_tokens",
    "estimated_cost_usd",
    "actual_input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "actual_cost_usd",
    "status",
    "cumulative_usd",
]

EDIT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["diagnosis", "hypothesis", "edits", "explanation"],
    "properties": {
        "diagnosis": {
            "type": "object",
            "additionalProperties": False,
            "required": ["suspected_component", "suspected_failure_mechanism", "supporting_evidence"],
            "properties": {
                "suspected_component": {"type": "string"},
                "suspected_failure_mechanism": {"type": "string"},
                "supporting_evidence": {"type": "string"},
            },
        },
        "hypothesis": {"type": "string"},
        "explanation": {"type": "string"},
        "edits": {
            "type": "array",
            "maxItems": EDIT_LIMITS["max_edit_ops"],
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "tensor",
                    "operation",
                    "indices",
                    "values",
                    "u_indices",
                    "u_values",
                    "v_indices",
                    "v_values",
                ],
                "properties": {
                    "tensor": {"type": "string", "enum": list(TENSOR_NAMES)},
                    "operation": {"type": "string", "enum": ["add", "set", "rank1"]},
                    "indices": {
                        "type": "array",
                        "maxItems": EDIT_LIMITS["max_edit_params"],
                        "items": {"type": "array", "items": {"type": "integer"}},
                    },
                    "values": {
                        "type": "array",
                        "maxItems": EDIT_LIMITS["max_edit_params"],
                        "items": {"type": "number"},
                    },
                    "u_indices": {
                        "type": "array",
                        "maxItems": EDIT_LIMITS["max_edit_params"],
                        "items": {"type": "integer"},
                    },
                    "u_values": {
                        "type": "array",
                        "maxItems": EDIT_LIMITS["max_edit_params"],
                        "items": {"type": "number"},
                    },
                    "v_indices": {
                        "type": "array",
                        "maxItems": EDIT_LIMITS["max_edit_params"],
                        "items": {"type": "integer"},
                    },
                    "v_values": {
                        "type": "array",
                        "maxItems": EDIT_LIMITS["max_edit_params"],
                        "items": {"type": "number"},
                    },
                },
            },
        },
    },
}


class ApiDisabled(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    pass


class LedgerCorrupt(RuntimeError):
    pass


class ApiHalted(RuntimeError):
    pass


def find_env_file(start: Path) -> Path:
    start = Path(start).resolve()
    for candidate in [start, *start.parents]:
        path = candidate / ".env"
        if path.is_file():
            return path
    raise FileNotFoundError("No .env file found while walking parents. Expected OPENAI_API_KEY in the repository root.")


def load_api_key(start: Path) -> str:
    """Load OPENAI_API_KEY. The returned value must not be logged."""
    from dotenv import load_dotenv

    env_path = find_env_file(start)
    load_dotenv(env_path, override=False)
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPEN_API_KEY") or ""
    if not isinstance(key, str) or len(key.strip()) < 8:
        raise RuntimeError("OPENAI_API_KEY is missing or too short. It was not printed.")
    if any(ch.isspace() for ch in key):
        raise RuntimeError("OPENAI_API_KEY contains whitespace. It was not printed.")
    return key.strip()


def count_tokens(text: str, encoding_name: str = API["tokenizer_encoding"]) -> int:
    import tiktoken

    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def _rate_pair(model: str, input_tokens: int) -> tuple[float, float]:
    if model not in API["prices_usd_per_million"]:
        raise KeyError(f"no price configured for {model}; refusing to guess")
    prices = API["prices_usd_per_million"][model]
    long_context = input_tokens > API["long_context_threshold"]
    input_rate = prices["input"] * (2.0 if long_context else 1.0)
    output_rate = prices["output"] * (1.5 if long_context else 1.0)
    cached_rate = prices["cached_input"] * (2.0 if long_context else 1.0)
    return input_rate, output_rate, cached_rate  # type: ignore[return-value]


def estimate_request_cost(model: str, input_tokens: int, max_output_tokens: int | None = None) -> float:
    """Worst-case cost. Ignores prompt caching and reserves the full output cap."""
    max_output_tokens = API["max_output_tokens"] if max_output_tokens is None else max_output_tokens
    input_rate, output_rate, _cached = _rate_pair(model, input_tokens)
    return (input_tokens / 1_000_000) * input_rate + (max_output_tokens / 1_000_000) * output_rate


def actual_request_cost(model: str, input_tokens: int, cached_tokens: int, output_tokens: int) -> float:
    """Bill from reported usage. Reasoning tokens are already inside output_tokens."""
    if cached_tokens < 0 or cached_tokens > input_tokens:
        raise LedgerCorrupt("cached token count is inconsistent with input tokens")
    input_rate, output_rate, cached_rate = _rate_pair(model, input_tokens)
    uncached = input_tokens - cached_tokens
    return (uncached / 1_000_000) * input_rate + (cached_tokens / 1_000_000) * cached_rate + (output_tokens / 1_000_000) * output_rate


@dataclass
class Reservation:
    call_id: str
    model: str
    experiment: str
    estimated_input_tokens: int
    estimated_cost_usd: float


class BudgetManager:
    def __init__(self, ledger_path: Path, max_budget_usd: float | None = None):
        self.ledger_path = Path(ledger_path)
        self.halt_path = self.ledger_path.with_suffix(".halt")
        self.max_budget_usd = API["max_budget_usd"] if max_budget_usd is None else max_budget_usd
        self.halted = self.halt_path.exists()
        self.rows: list[dict] = []
        self.spent = 0.0
        self._load()

    def _load(self) -> None:
        if not self.ledger_path.exists():
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger_path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=LEDGER_COLUMNS).writeheader()
            return
        with self.ledger_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or list(reader.fieldnames) != LEDGER_COLUMNS:
                raise LedgerCorrupt(f"ledger columns do not match the expected header: {self.ledger_path}")
            running = 0.0
            for row in reader:
                try:
                    actual = float(row["actual_cost_usd"]) if row["actual_cost_usd"] not in ("", "unknown") else 0.0
                    recorded = float(row["cumulative_usd"]) if row["cumulative_usd"] not in ("", "unknown") else running
                except ValueError as exc:
                    raise LedgerCorrupt("ledger contains a non-numeric cost") from exc
                if row["status"] == "uncertain":
                    self.halted = True
                if row["status"] == "completed":
                    running += actual
                    if abs(running - recorded) > 1e-6:
                        raise LedgerCorrupt("cumulative spend does not match the sum of completed calls")
                self.rows.append(row)
            self.spent = running

    def remaining(self) -> float:
        return self.max_budget_usd - self.spent

    def authorize(self, model: str, input_tokens: int, call_id: str, experiment: str, max_output_tokens: int | None = None) -> Reservation:
        if self.halted or self.halt_path.exists():
            raise ApiHalted("API execution is halted because an earlier call was uncertain or the ledger is frozen.")
        if model not in API["models"]:
            raise ApiHalted(f"{model} is not in the configured model list. Refusing to substitute another model.")
        estimate = estimate_request_cost(model, input_tokens, max_output_tokens)
        if estimate > self.remaining() + 1e-12:
            raise BudgetExceeded(
                f"Estimated worst-case cost ${estimate:.4f} exceeds remaining budget ${self.remaining():.4f}."
            )
        return Reservation(call_id, model, experiment, input_tokens, estimate)

    def _append(self, row: dict) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=LEDGER_COLUMNS)
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        self.rows.append(row)

    def commit(self, reservation: Reservation, usage: dict) -> float:
        actual = actual_request_cost(
            reservation.model,
            int(usage["input_tokens"]),
            int(usage.get("cached_tokens", 0)),
            int(usage["output_tokens"]),
        )
        self.spent += actual
        self._append(
            {
                "timestamp": _now(),
                "call_id": reservation.call_id,
                "model": reservation.model,
                "experiment": reservation.experiment,
                "estimated_input_tokens": reservation.estimated_input_tokens,
                "estimated_cost_usd": f"{reservation.estimated_cost_usd:.8f}",
                "actual_input_tokens": int(usage["input_tokens"]),
                "cached_input_tokens": int(usage.get("cached_tokens", 0)),
                "output_tokens": int(usage["output_tokens"]),
                "reasoning_tokens": int(usage.get("reasoning_tokens", 0)),
                "actual_cost_usd": f"{actual:.8f}",
                "status": "completed",
                "cumulative_usd": f"{self.spent:.8f}",
            }
        )
        return actual

    def mark_uncertain(self, reservation: Reservation, reason: str) -> None:
        """Stop the experiment. Do not treat the attempt as free."""
        self.halted = True
        self.halt_path.write_text(json.dumps({"reason": reason, "call_id": reservation.call_id, "at": _now()}), encoding="utf-8")
        self._append(
            {
                "timestamp": _now(),
                "call_id": reservation.call_id,
                "model": reservation.model,
                "experiment": reservation.experiment,
                "estimated_input_tokens": reservation.estimated_input_tokens,
                "estimated_cost_usd": f"{reservation.estimated_cost_usd:.8f}",
                "actual_input_tokens": "",
                "cached_input_tokens": "",
                "output_tokens": "",
                "reasoning_tokens": "",
                "actual_cost_usd": "unknown",
                "status": "uncertain",
                "cumulative_usd": "unknown",
            }
        )


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def serialize_state(state: dict[str, torch.Tensor], decimals: int) -> str:
    blocks = []
    for name in TENSOR_NAMES:
        tensor = state[name].detach().cpu().float()
        shape = " ".join(str(int(dim)) for dim in tensor.shape)
        blocks.append(f"TENSOR {name} {shape}")
        array = tensor.numpy()
        if array.ndim == 1:
            blocks.append(" ".join(f"{value:.{decimals}f}" for value in array))
        else:
            for row in array:
                blocks.append(" ".join(f"{value:.{decimals}f}" for value in row))
        blocks.append("END")
    return "\n".join(blocks)


def parse_serialized_state(text: str, decimals: int) -> dict[str, torch.Tensor]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    state = {}
    cursor = 0
    while cursor < len(lines):
        header = lines[cursor].split()
        if header[0] != "TENSOR":
            raise ValueError("serialized weights are missing a TENSOR header")
        name = header[1]
        shape = tuple(int(part) for part in header[2:])
        cursor += 1
        rows = []
        while cursor < len(lines) and lines[cursor] != "END":
            rows.append([float(part) for part in lines[cursor].split()])
            cursor += 1
        if cursor >= len(lines) or lines[cursor] != "END":
            raise ValueError(f"{name} is missing END")
        cursor += 1
        if len(shape) == 1:
            values = rows[0]
            tensor = torch.tensor(values, dtype=torch.float32)
        else:
            tensor = torch.tensor(rows, dtype=torch.float32)
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} parsed shape {tuple(tensor.shape)} != {shape}")
        state[name] = tensor
    missing = [name for name in TENSOR_NAMES if name not in state]
    if missing:
        raise ValueError(f"serialized weights missing {missing}")
    return state


def _active_flags(features) -> str:
    names = [name for name in CLASS_FEATURES if float(features[FEATURE_NAMES.index(name)]) > 0.5]
    return ",".join(names) if names else "regular_default"


def format_diagnostic(frame) -> str:
    lines = ["phrase | class | flags | gender | number | definiteness | true | predicted | ok"]
    ordered = frame.sort_values(["is_target", "lexical_class", "stem", "condition"], ascending=[False, True, True, True])
    for row in ordered.itertuples(index=False):
        pred = getattr(row, "pred_exponent")
        ok = "ok" if row.correct else "WRONG"
        lines.append(
            f"{row.phrase} | {row.lexical_class} | {_active_flags(row.features)} | {row.gender} | {row.number} | "
            f"{row.definiteness} | {row.exponent} | {pred} | {ok}"
        )
    return "\n".join(lines)


def task_description() -> str:
    return """\
You are editing the weights of a small feed-forward classifier. The network reads grammatical features of a Swedish noun phrase and lexical features of an adjective, and it predicts an inflection exponent. A separate deterministic map turns that exponent into a surface form. You will not write code, a training loop, or a gradient step. You will return a structured edit of the corrupted checkpoint below.

Swedish adjective agreement in this task:

- Uter singular indefinite uses the base form: en stor bil, en vacker bil, en liten bil, en bra bil.
- Neuter singular indefinite adds a class-specific strong neuter:
  - regular and -ig and syncopating adjectives take NEUT_T: stort, roligt, vackert, gammalt.
  - geminating adjectives take NEUT_TT: nytt, blått, rött, gott, vitt, vått.
  - adjectives in -en, and liten, take NEUT_ET: moget, öppet, eget, litet.
  - invariant adjectives stay BASE: bra, kul, gyllene.
- Plural or definite collapses to one weak form for every class except liten and the invariants:
  - WEAK_A: stora, nya, roliga, röda.
  - WEAK_DROP_E: vackra, gamla, enkla.
  - WEAK_NA: mogna, öppna, egna.
- liten does not use that weak collapse. Definite singular is LILLA (lilla) for both genders. Plural is SMA (små) whether definite or indefinite.
- Invariant adjectives ignore gender, number and definiteness.
- Gender is otherwise neutralized in the plural and under definiteness. It matters in singular indefinite.
- stem_0..stem_15 are idiosyncratic bits that differ across lexemes. The exponent does not depend on them. Two stems of one class should be treated by the same rule.

The defect to repair is selective. Neuter singular indefinite agreement is failing for every non-invariant adjective: the network predicts the uter base form instead of NEUT_T (stort, roligt, vackert), NEUT_TT (nytt, rött, vitt) or NEUT_ET (moget, litet). Uter singular, all plurals, all definite forms and invariant adjectives should stay as they are. Gender is not supposed to matter in the plural or under definiteness, and liten still has to keep lilla and små distinct from litet.

You do not know how the network was damaged. You do not have the original weights, gradients, or the held-out test labels. The development lines below are the only labeled outcomes you may use. Several development stems are held out from the original training split, so a repair that memorizes one printed lexeme is not the goal. Edit the computation that implements neuter singular indefinite agreement.
"""


def constraints_text() -> str:
    limit = EDIT_LIMITS
    return f"""\
Edit budget and operations:

- At most {limit["max_edit_params"]} scalar specifications in the whole proposal. Each add/set value counts as one. A rank-1 update counts as len(u_indices)+len(v_indices).
- At most {limit["max_edit_ops"]} edit operations.
- operation "add": new = current + value, at indices. For a matrix, an index is [row, col]. For a bias, an index is [i].
- operation "set": new = value, at indices.
- operation "rank1": add the outer product of two sparse vectors. u_indices index rows (out features) and v_indices index columns (in features). Every pair (u_i, v_j) receives u_values[i] * v_values[j]. Leave indices and values empty for a rank-1 edit. Leave the u_* and v_* arrays empty for add/set.
- Absolute change at each add/set coordinate at most {limit["max_abs_delta"]}. Absolute stored value at most {limit["max_abs_value"]}.
- Frobenius norm of the whole change, including rank-1 updates, at most {limit["max_frobenius"]}.
- A rank-1 update may affect at most {limit["max_rank1_affected_entries"]} entries.
- Allowed tensor names: {", ".join(TENSOR_NAMES)}. Shapes are [out_features, in_features] for weights.
- Apply the edit to the corrupted checkpoint in this prompt. Do not assume a previous edit is still loaded.
- An invalid proposal is an unsuccessful attempt. Do not wrap the JSON in code fences.
"""


def build_prompt(state: dict[str, torch.Tensor], architecture: dict, diagnostic_frame, decimals: int, feedback: dict | None = None) -> str:
    arch_lines = [
        f"input_dim={architecture['in_dim']}",
        f"hidden_dim={architecture['hidden']}",
        f"n_hidden_layers=2",
        "activation=relu",
        f"output_dim={architecture['out_dim']}",
        f"n_parameters={architecture['n_parameters']}",
        "forward: h1=relu(fc1(x)); h2=relu(fc2(h1)); logits=out(h2)",
        "Weights below are exact at the printed decimal precision.",
    ]
    features = "\n".join(f"{index}: {name}" for index, name in enumerate(FEATURE_NAMES))
    exponents = "\n".join(f"{index}: {name}" for index, name in enumerate(EXPONENTS))
    parts = [
        task_description(),
        "# Architecture",
        "\n".join(arch_lines),
        "# Feature index",
        features,
        "# Exponent index",
        exponents,
        "# Grammar feature groups",
        "grammar one-hot: " + ", ".join(GRAMMAR_FEATURES),
        "class flags: " + ", ".join(CLASS_FEATURES),
        "# Constraints",
        constraints_text(),
        "# Corrupted checkpoint",
        serialize_state(state, decimals),
        "# Development diagnoses",
        format_diagnostic(diagnostic_frame),
    ]
    if feedback is not None:
        mistakes = "\n".join(feedback.get("mistakes", [])) or "(none)"
        errors = "\n".join(feedback.get("errors", [])) or "(none)"
        parts.extend(
            [
                "# Feedback on your previous edit",
                "The measurements below are from the development set only. They are not held-out test numbers.",
                f"development target accuracy: {feedback['target_acc']:.4f}",
                f"development non-target accuracy: {feedback['nontarget_acc']:.4f}",
                f"development non-target macro accuracy: {feedback['nontarget_macro_acc']:.4f}",
                "validation errors:",
                errors,
                "representative remaining mistakes:",
                mistakes,
                "previous hypothesis:",
                feedback.get("hypothesis", ""),
                "Return a replacement edit relative to the corrupted checkpoint printed above, not a patch on top of the previous edit.",
            ]
        )
    else:
        parts.append(
            "# Attempt\nThis is a one-shot edit. You will not see the result before you answer.\n"
            "Fill diagnosis.suspected_component with the tensor, units, or input features you think are responsible.\n"
            "Fill diagnosis.suspected_failure_mechanism with how that part of the computation produces the uter base instead of the strong neuter.\n"
            "Fill diagnosis.supporting_evidence with what you can see in the printed weights and the labeled development rows.\n"
            "Use hypothesis for one sentence restating that diagnosis.\n"
            "Put the intervention in the top-level edits array. Use the smallest edit that restores the neuter singular indefinite exponent without moving uter, plural, definite or invariant cells.\n"
            "Use explanation for why that edit should restore the failing forms and why the other cells should stay intact."
        )
    return "\n\n".join(parts)


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_and_apply(state: dict[str, torch.Tensor], proposal: dict) -> tuple[dict[str, torch.Tensor] | None, list[str]]:
    """Apply a proposal to a fresh copy, or reject it entirely."""
    errors: list[str] = []
    if not isinstance(proposal, dict):
        return None, ["proposal is not an object"]
    edits = proposal.get("edits")
    if not isinstance(edits, list):
        return None, ["edits is missing"]
    if len(edits) > EDIT_LIMITS["max_edit_ops"]:
        errors.append("too many edit operations")
    working = {name: tensor.detach().cpu().float().clone() for name, tensor in state.items()}
    specifications = 0
    for edit_index, edit in enumerate(edits):
        if not isinstance(edit, dict):
            errors.append(f"edit {edit_index} is not an object")
            continue
        name = edit.get("tensor")
        operation = edit.get("operation")
        if name not in working:
            errors.append(f"edit {edit_index} names an unknown tensor")
            continue
        tensor = working[name]
        if operation in {"add", "set"}:
            indices = edit.get("indices") or []
            values = edit.get("values") or []
            if len(indices) != len(values):
                errors.append(f"edit {edit_index} index/value length mismatch")
                continue
            specifications += len(values)
            for coord, value in zip(indices, values):
                if not _finite(value):
                    errors.append(f"edit {edit_index} has a non-finite value")
                    continue
                if not isinstance(coord, (list, tuple)) or len(coord) != tensor.ndim:
                    errors.append(f"edit {edit_index} has an index of the wrong rank")
                    continue
                if any((not isinstance(axis, int)) or isinstance(axis, bool) or axis < 0 or axis >= tensor.shape[dim] for dim, axis in enumerate(coord)):
                    errors.append(f"edit {edit_index} has an out-of-range index")
                    continue
                current = float(tensor[tuple(coord)])
                updated = float(value) if operation == "set" else current + float(value)
                if abs(updated - current) > EDIT_LIMITS["max_abs_delta"] + 1e-8:
                    errors.append(f"edit {edit_index} exceeds the per-coordinate delta cap")
                    continue
                if abs(updated) > EDIT_LIMITS["max_abs_value"] + 1e-8:
                    errors.append(f"edit {edit_index} exceeds the absolute value cap")
                    continue
                tensor[tuple(coord)] = updated
        elif operation == "rank1":
            u_index = list(edit.get("u_indices") or [])
            u_values = list(edit.get("u_values") or [])
            v_index = list(edit.get("v_indices") or [])
            v_values = list(edit.get("v_values") or [])
            if tensor.ndim != 2:
                errors.append(f"edit {edit_index} rank1 requires a matrix")
                continue
            if len(u_index) != len(u_values) or len(v_index) != len(v_values):
                errors.append(f"edit {edit_index} rank1 length mismatch")
                continue
            if len(set(u_index)) != len(u_index) or len(set(v_index)) != len(v_index):
                errors.append(f"edit {edit_index} rank1 has duplicate indices")
                continue
            specifications += len(u_index) + len(v_index)
            if len(u_index) * len(v_index) > EDIT_LIMITS["max_rank1_affected_entries"]:
                errors.append(f"edit {edit_index} rank1 touches too many entries")
                continue
            if not all(_finite(value) for value in u_values + v_values):
                errors.append(f"edit {edit_index} rank1 has a non-finite value")
                continue
            if any(not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= tensor.shape[0] for i in u_index):
                errors.append(f"edit {edit_index} rank1 has a bad row index")
                continue
            if any(not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= tensor.shape[1] for i in v_index):
                errors.append(f"edit {edit_index} rank1 has a bad column index")
                continue
            u = torch.zeros(tensor.shape[0])
            v = torch.zeros(tensor.shape[1])
            for index, value in zip(u_index, u_values):
                u[index] = float(value)
            for index, value in zip(v_index, v_values):
                v[index] = float(value)
            update = torch.outer(u, v)
            updated = tensor + update
            if float(update.pow(2).sum().sqrt()) > EDIT_LIMITS["max_frobenius"] + 1e-8:
                errors.append(f"edit {edit_index} rank1 exceeds the Frobenius cap by itself")
                continue
            if float(updated.abs().max()) > EDIT_LIMITS["max_abs_value"] + 1e-8:
                errors.append(f"edit {edit_index} rank1 exceeds the absolute value cap")
                continue
            tensor.copy_(updated)
        else:
            errors.append(f"edit {edit_index} has an unknown operation")
    if specifications > EDIT_LIMITS["max_edit_params"]:
        errors.append(f"proposal uses {specifications} scalars, above the budget of {EDIT_LIMITS['max_edit_params']}")
    if specifications == 0 and not errors:
        errors.append("proposal does not change any weights")
    total = 0.0
    for name in working:
        delta = working[name] - state[name].detach().cpu().float()
        total += float((delta * delta).sum())
    if total**0.5 > EDIT_LIMITS["max_frobenius"] + 1e-6:
        errors.append("total Frobenius norm exceeds the cap")
    if errors:
        return None, errors
    return working, []


def execute_edit(
    prompt: str,
    model_name: str,
    budget: BudgetManager,
    call_id: str,
    experiment: str,
    api_enabled: bool,
    start: Path,
    max_output_tokens: int | None = None,
    timeout_seconds: float | None = None,
) -> dict:
    """Send one Responses API call. Disabled unless api_enabled is explicitly true."""
    if not api_enabled:
        raise ApiDisabled(
            "API execution is disabled. Phase 1 does not call the API. "
            "Set api_enabled True only after an explicit authorization to spend the budget."
        )
    output_cap = API["max_output_tokens"] if max_output_tokens is None else max_output_tokens
    raw_tokens = count_tokens(prompt)
    reserved_tokens = math.ceil(raw_tokens * API["input_token_safety_factor"])
    reservation = budget.authorize(model_name, reserved_tokens, call_id, experiment, output_cap)
    key = load_api_key(start)
    try:
        from openai import OpenAI

        client = OpenAI(api_key=key, timeout=API["timeout_seconds"] if timeout_seconds is None else timeout_seconds, max_retries=0)
        response = client.responses.create(
            model=model_name,
            input=prompt,
            reasoning={"effort": API["reasoning_effort"]},
            max_output_tokens=output_cap,
            store=False,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "weight_edit",
                    "strict": True,
                    "schema": EDIT_SCHEMA,
                }
            },
        )
    except Exception as exc:
        budget.mark_uncertain(reservation, type(exc).__name__)
        raise ApiHalted(
            "The API call did not return a usable usage record. "
            "Execution is halted and the attempt was not assumed to be free."
        ) from exc
    usage_obj = getattr(response, "usage", None)
    if usage_obj is None:
        budget.mark_uncertain(reservation, "missing_usage")
        raise ApiHalted("The response had no usage object. Execution is halted.")
    input_details = getattr(usage_obj, "input_tokens_details", None)
    output_details = getattr(usage_obj, "output_tokens_details", None)
    usage = {
        "input_tokens": int(usage_obj.input_tokens),
        "output_tokens": int(usage_obj.output_tokens),
        "cached_tokens": int(getattr(input_details, "cached_tokens", 0) or 0),
        "reasoning_tokens": int(getattr(output_details, "reasoning_tokens", 0) or 0),
    }
    actual = budget.commit(reservation, usage)
    text = getattr(response, "output_text", "") or ""
    status = getattr(response, "status", "")
    parsed = None
    parse_error = None
    if status == "completed" and text:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            parse_error = str(exc)
    raw = None
    if hasattr(response, "model_dump"):
        raw = response.model_dump(mode="json")
        if isinstance(raw, dict):
            raw.pop("api_key", None)
    return {
        "status": status,
        "text": text,
        "proposal": parsed,
        "parse_error": parse_error,
        "usage": usage,
        "estimated_cost_usd": reservation.estimated_cost_usd,
        "actual_cost_usd": actual,
        "response_id": getattr(response, "id", None),
        "model_revision": getattr(response, "model", None),
        "raw": raw,
    }


def run_self_tests(tmp_dir: Path) -> list[str]:
    """Mock the budget, the edit gate and the disabled API switch. No network."""
    notes = []
    ledger = tmp_dir / "api_usage.csv"
    manager = BudgetManager(ledger, max_budget_usd=5.0)
    reservation = manager.authorize("gpt-6.2-sol", 1_000, "mock-1", "unit")
    actual = manager.commit(
        reservation,
        {"input_tokens": 1000, "cached_tokens": 200, "output_tokens": 500, "reasoning_tokens": 300},
    )
    expected = actual_request_cost("gpt-6.2-sol", 1000, 200, 500)
    if abs(actual - expected) > 1e-9:
        raise AssertionError("committed cost does not match the usage formula")
    # Reasoning tokens must not be added on top of output tokens.
    doubled = actual_request_cost("gpt-6.2-sol", 1000, 200, 500 + 300)
    if doubled <= actual:
        raise AssertionError("test setup failed to distinguish double counting")
    reloaded = BudgetManager(ledger, max_budget_usd=5.0)
    if abs(reloaded.spent - actual) > 1e-9:
        raise AssertionError("reloaded ledger spend does not match")
    tiny = BudgetManager(tmp_dir / "tiny.csv", max_budget_usd=0.01)
    try:
        tiny.authorize("gpt-6.2-astra", 20_000, "too-big", "unit")
        raise AssertionError("over-budget call was authorized")
    except BudgetExceeded:
        notes.append("over-budget call rejected")
    uncertain = BudgetManager(tmp_dir / "uncertain.csv", max_budget_usd=5.0)
    reservation = uncertain.authorize("gpt-6.2-astra", 100, "mock-uncertain", "unit")
    uncertain.mark_uncertain(reservation, "timeout")
    try:
        uncertain.authorize("gpt-6.2-sol", 100, "after-halt", "unit")
        raise AssertionError("halted budget authorized another call")
    except ApiHalted:
        notes.append("uncertain call halts later calls")
    broken = tmp_dir / "broken.csv"
    broken.write_text(
        ",".join(LEDGER_COLUMNS) + "\n"
        "t,c,gpt-6.2-sol,unit,10,0.1,10,0,10,0,1.0,completed,0.5\n",
        encoding="utf-8",
    )
    try:
        BudgetManager(broken, max_budget_usd=5.0)
        raise AssertionError("corrupt ledger was accepted")
    except LedgerCorrupt:
        notes.append("corrupt cumulative total is rejected")
    try:
        execute_edit("hello", "gpt-6.2-sol", manager, "should-not-run", "unit", api_enabled=False, start=tmp_dir)
        raise AssertionError("disabled API path fell through")
    except ApiDisabled:
        notes.append("api_enabled false blocks the client")
    env = tmp_dir / ".env"
    env.write_text("OPENAI_API_KEY=test-key-not-real\n", encoding="utf-8")
    previous = os.environ.get("OPENAI_API_KEY")
    os.environ.pop("OPENAI_API_KEY", None)
    try:
        loaded = load_api_key(tmp_dir)
    finally:
        if previous is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = previous
    if loaded != "test-key-not-real":
        raise AssertionError("env loader did not return the test key")
    notes.append("env loader reads a local key without printing it")

    state = {
        "fc1.weight": torch.zeros(4, 3),
        "fc1.bias": torch.zeros(4),
        "fc2.weight": torch.eye(4),
        "fc2.bias": torch.zeros(4),
        "out.weight": torch.ones(2, 4),
        "out.bias": torch.zeros(2),
    }
    ok, errors = validate_and_apply(
        state,
        {
            "hypothesis": "test",
            "explanation": "test",
            "edits": [
                {
                    "tensor": "fc2.weight",
                    "operation": "add",
                    "indices": [[0, 1]],
                    "values": [0.25],
                    "u_indices": [],
                    "u_values": [],
                    "v_indices": [],
                    "v_values": [],
                }
            ],
        },
    )
    if errors or ok is None or abs(float(ok["fc2.weight"][0, 1]) - 0.25) > 1e-6:
        raise AssertionError(f"valid edit failed: {errors}")
    rejected, errors = validate_and_apply(
        state,
        {
            "hypothesis": "bad",
            "explanation": "bad",
            "edits": [
                {
                    "tensor": "not.a.tensor",
                    "operation": "add",
                    "indices": [[0, 0]],
                    "values": [0.1],
                    "u_indices": [],
                    "u_values": [],
                    "v_indices": [],
                    "v_values": [],
                }
            ],
        },
    )
    if rejected is not None or not errors:
        raise AssertionError("invalid tensor name was applied")
    notes.append("edit validator accepts a sparse add and rejects an unknown tensor")
    text = serialize_state(state, 3)
    parsed = parse_serialized_state(text, 3)
    for name in TENSOR_NAMES:
        if torch.max(torch.abs(parsed[name] - state[name])).item() > 1e-6:
            raise AssertionError(f"serialization roundtrip failed for {name}")
    notes.append("weight serialization roundtrips")
    return notes
