"""Phase 1 orchestration. Trains, corrupts, gates and prices prompts. Does not call the API."""

from __future__ import annotations

import json
import math
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import config
from .api_client import (
    EDIT_SCHEMA,
    ApiDisabled,
    ApiHalted,
    BudgetExceeded,
    BudgetManager,
    build_prompt,
    count_tokens,
    estimate_request_cost,
    execute_edit,
    load_api_key,
    parse_serialized_state,
    run_self_tests,
    serialize_state,
    validate_and_apply,
)
from .baselines import random_hidden_edits, repair_gradient_budget, repair_last_layer, repair_output_bias
from .corruption import generate_candidates, train_masks
from .evaluation import (
    permutation_max_abs_diff,
    permute_hidden_neurons,
    plot_cost_estimates,
    plot_heatmaps,
    plot_phase2_scatter,
    plot_repair_bars,
    plot_training_curve,
    state_delta,
    summarize,
)
from .grammar import FEATURE_NAMES, build_dataset, dataset_summary, matrix_from_frame
from .local_checks import run_local_validation, sha256_bytes, sha256_text
from .model import (
    architecture_record,
    copy_model,
    load_state,
    meets_learning_target,
    parameter_count,
    quantize_state,
    select_and_train,
)

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]


def _utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if math.isnan(number) or math.isinf(number) else number
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def _public(summary: dict) -> dict:
    return {key: value for key, value in summary.items() if key not in {"cells", "scored"}}


def _save_model(path: Path, model, extra: dict) -> None:
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "architecture": architecture_record(model),
            "extra": extra,
        },
        path,
    )


def _bundles(frame: pd.DataFrame) -> dict:
    train = frame[frame["split"].isin(["train", "inlexicon"])].reset_index(drop=True)
    dev = frame[frame["split"].isin(["dev_diag", "dev_gate"])].reset_index(drop=True)
    liten = frame[frame["split"] == "inlexicon"].reset_index(drop=True)
    if set(train["split"]) & {"dev_diag", "dev_gate", "test"}:
        raise AssertionError("training bundle contains held-out rows")
    x_train, y_train = matrix_from_frame(train)
    x_dev, y_dev = matrix_from_frame(dev)
    x_liten, y_liten = matrix_from_frame(liten)
    return {
        "x_train": x_train,
        "y_train": y_train,
        "class_train": train["lexical_class"].to_numpy(),
        "x_dev": x_dev,
        "y_dev": y_dev,
        "class_dev": dev["lexical_class"].to_numpy(),
        "x_liten": x_liten,
        "y_liten": y_liten,
    }


def _mask_frame(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Masks for protocol-1 neuron scores.

    other_neuter is the definite singular neuter cell. Its label does not depend
    on gender, so a lesion that also moves this cell is not selective.
    """
    uter_sg_indef = (frame["gender"] == "uter") & (frame["number"] == "sg") & (frame["definiteness"] == "indef")
    neuter_sg_def = (frame["gender"] == "neuter") & (frame["number"] == "sg") & (frame["definiteness"] == "def")
    return train_masks(
        frame["is_target"].to_numpy(),
        ((frame["lexical_class"] == "geminating") & uter_sg_indef).to_numpy(),
        (neuter_sg_def & frame["lexical_class"].ne("invariant")).to_numpy(),
        (uter_sg_indef & frame["lexical_class"].ne("invariant")).to_numpy(),
    )


def _filter_candidate(metrics: dict, liten_acc: float) -> tuple[bool, str]:
    reasons = []
    if metrics["nontarget_acc"] < config.CORRUPTION["nontarget_micro_min"]:
        reasons.append("nontarget_micro")
    if metrics["nontarget_macro_acc"] < config.CORRUPTION["nontarget_macro_min"]:
        reasons.append("nontarget_macro")
    if liten_acc < config.CORRUPTION["liten_min"]:
        reasons.append("liten")
    low, high = config.CORRUPTION["prefer_target_accuracy"]
    target = metrics["target_acc"]
    if low <= target <= high:
        band = "in_window"
    elif target < low and config.CORRUPTION["allow_atomic_target_failure"]:
        band = "atomic"
    else:
        reasons.append("target_range")
        band = "reject"
    return not reasons, band if not reasons else ",".join(reasons)


def _rank_key(metrics: dict, l2: float, band: str) -> tuple:
    target = metrics["target_acc"]
    low, high = config.CORRUPTION["prefer_target_accuracy"]
    band_order = {"in_window": 0, "atomic": 1}.get(band, 2)
    if low <= target <= high:
        target_penalty = abs(target - 0.40)
    elif target < low:
        target_penalty = low - target
    else:
        target_penalty = target - high
    preserve = (1.0 - metrics["nontarget_acc"]) + (1.0 - metrics["nontarget_macro_acc"])
    return (band_order, target_penalty + preserve, l2)


def _solves(metrics: dict) -> bool:
    return (
        metrics["target_acc"] >= config.BASELINES["solved_target_min"]
        and metrics["nontarget_acc"] >= config.BASELINES["solved_nontarget_micro_min"]
    )


def _fit_baselines(model, x_diag: np.ndarray, y_diag: np.ndarray):
    bias = repair_output_bias(model, x_diag, y_diag)
    last = repair_last_layer(model, x_diag, y_diag)
    gradient, locations = repair_gradient_budget(model, x_diag, y_diag)
    return {"bias": bias, "last_layer": last, "gradient": gradient, "gradient_locations": locations}


def _choose_decimals(model, dev: pd.DataFrame, liten: pd.DataFrame) -> tuple[int, dict]:
    notes = []
    for decimals in config.QUANTIZE_DECIMALS_TRY:
        quantized = copy_model(model)
        load_state(quantized, quantize_state(model.state_dict(), decimals))
        dev_metrics = summarize(dev, quantized, "dev")
        liten_metrics = summarize(liten, quantized, "inlexicon")
        ok = dev_metrics["example_acc"] >= config.TRAINING["min_dev_example_acc"] and liten_metrics["example_acc"] >= config.TRAINING["min_liten_acc"]
        notes.append({"decimals": decimals, "dev_example_acc": dev_metrics["example_acc"], "liten_acc": liten_metrics["example_acc"], "accepted": ok})
        if ok:
            load_state(model, quantize_state(model.state_dict(), decimals))
            return decimals, {"trials": notes}
    raise RuntimeError(f"quantization destroyed the learned grammar: {notes}")


def run_phase1(experiment_dir: Path | None = None, overwrite: bool = False) -> dict:
    _utf8_stdio()
    experiment_dir = Path(experiment_dir) if experiment_dir else EXPERIMENT_DIR
    results = experiment_dir / "results"
    figures = results / "figures"
    checkpoints = results / "checkpoints"
    for path in (results, figures, checkpoints):
        path.mkdir(parents=True, exist_ok=True)
    summary_path = results / "phase1_summary.json"
    if not overwrite and summary_path.exists() and (checkpoints / "corrupted.pt").exists():
        print("Phase 1 artifacts are frozen. Loading the saved summary and not retraining.")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    started = time.perf_counter()
    print("Building the Swedish adjective dataset...")
    frame = build_dataset(config.SEED)
    summary = dataset_summary(frame)
    printable = frame.drop(columns=["features"])
    printable.to_csv(results / "dataset.csv", index=False)
    _write_json(results / "dataset_summary.json", summary)
    print(
        f"  {summary['n_stems']} stems, {summary['n_rows']} rows, "
        f"{summary['n_features']} features, {summary['hand_checked']} hand-checked forms."
    )

    print("Training the MLP...")
    bundles = _bundles(frame)
    trained, search = select_and_train(bundles, config.SEED)
    if not meets_learning_target(trained):
        raise RuntimeError(f"selected model missed the learning target: {search}")
    print(
        f"  hidden={trained.hidden}, parameters={trained.n_parameters}, "
        f"epochs={trained.epochs_ran}, dev accuracy={trained.dev_example_acc:.3f}, "
        f"liten={trained.liten_acc:.3f}, seconds={trained.seconds:.1f}"
    )
    plot_training_curve(trained.history, figures / "training_curve.png")

    dev_all = frame[frame["split"].isin(["dev_diag", "dev_gate"])].reset_index(drop=True)
    liten = frame[frame["split"] == "inlexicon"].reset_index(drop=True)
    decimals, quant_notes = _choose_decimals(trained.model, dev_all, liten)
    print(f"  serializing weights at {decimals} decimal places.")

    test = frame[frame["split"] == "test"].reset_index(drop=True)
    original_test = summarize(test, trained.model, "test")
    if original_test["example_acc"] < 0.95 or original_test["target_acc"] < 0.95:
        raise RuntimeError(
            "The quantized network does not reach 95% held-out accuracy "
            f"(example={original_test['example_acc']:.3f}, target={original_test['target_acc']:.3f}). "
            "Fix the dataset or architecture before corrupting it."
        )
    print(f"  held-out test example accuracy={original_test['example_acc']:.3f}, target cell={original_test['target_acc']:.3f}")
    _save_model(
        checkpoints / "original.pt",
        trained.model,
        {"decimals": decimals, "seed": config.SEED, "hidden_search": search},
    )

    print("Searching hidden-layer corruptions...")
    train_rows = frame[frame["split"] == "train"].reset_index(drop=True)
    x_train, _ = matrix_from_frame(train_rows)
    stem_start = FEATURE_NAMES.index("stem_0")
    candidates = generate_candidates(
        trained.model,
        x_train,
        _mask_frame(train_rows),
        slice(stem_start, stem_start + config.N_STEM_BITS),
    )
    gate_frame = frame[frame["split"] == "dev_gate"].reset_index(drop=True)
    filter_frame = frame[frame["split"].isin(["dev_gate", "inlexicon"])].reset_index(drop=True)
    probe = copy_model(trained.model)
    original_state = {key: value.detach().cpu().clone() for key, value in trained.model.state_dict().items()}
    scored_rows = []
    eligible = []
    for candidate in candidates:
        quantized = quantize_state(candidate["state"], decimals)
        load_state(probe, quantized)
        filt = summarize(filter_frame, probe, "filter")
        liten_acc = float(filt["scored"].loc[filt["scored"]["split"] == "inlexicon", "correct"].mean())
        gate_view = summarize(gate_frame, probe, "dev_gate")
        delta = state_delta(original_state, quantized)
        passed, band = _filter_candidate(gate_view, liten_acc)
        record = {
            "candidate_index": candidate["candidate_index"],
            "family": candidate["family"],
            "detail": candidate["detail"],
            "filter_passed": passed,
            "band": band,
            "liten_acc": liten_acc,
            "l2": delta["l2"],
            "n_changed": delta["n_changed"],
            **{f"gate_{key}": value for key, value in _public(gate_view).items() if key != "split"},
        }
        scored_rows.append(record)
        if passed:
            eligible.append({**record, "state": quantized, "rank": _rank_key(gate_view, delta["l2"], band)})
    pd.DataFrame(
        [
            {
                "candidate_index": row["candidate_index"],
                "family": row["family"],
                "detail": json.dumps(_jsonable(row["detail"])),
                "filter_passed": row["filter_passed"],
                "band": row["band"],
                "liten_acc": row["liten_acc"],
                "l2": row["l2"],
                "n_changed": row["n_changed"],
                "target_acc": row["gate_target_acc"],
                "nontarget_acc": row["gate_nontarget_acc"],
                "nontarget_macro_acc": row["gate_nontarget_macro_acc"],
                "example_acc": row["gate_example_acc"],
            }
            for row in scored_rows
        ]
    ).to_csv(results / "corruption_candidates.csv", index=False)
    print(f"  {len(candidates)} candidates, {len(eligible)} passed the dev-gate accuracy filter.")
    eligible.sort(key=lambda row: row["rank"])

    diag = frame[frame["split"].isin(["dev_diag", "inlexicon"])].reset_index(drop=True)
    x_diag, y_diag = matrix_from_frame(diag)
    gate_log = []
    winner = None
    fitted = None
    for record in eligible[: config.BASELINES["max_gate_evaluations"]]:
        load_state(probe, record["state"])
        corrupted_gate = summarize(gate_frame, probe, "dev_gate")
        repairs = _fit_baselines(probe, x_diag, y_diag)
        bias_m = summarize(gate_frame, repairs["bias"], "dev_gate")
        last_m = summarize(gate_frame, repairs["last_layer"], "dev_gate")
        grad_m = summarize(gate_frame, repairs["gradient"], "dev_gate")
        gain = grad_m["target_acc"] - corrupted_gate["target_acc"]
        bias_solves = _solves(bias_m)
        last_solves = _solves(last_m)
        grad_ok = gain >= config.BASELINES["gradient_min_gain"] and grad_m["nontarget_acc"] >= config.BASELINES["gradient_nontarget_min"]
        passed = (not bias_solves) and (not last_solves) and grad_ok
        entry = {
            "candidate_index": record["candidate_index"],
            "family": record["family"],
            "detail": record["detail"],
            "band": record["band"],
            "corrupted_target": corrupted_gate["target_acc"],
            "corrupted_nontarget": corrupted_gate["nontarget_acc"],
            "bias_target": bias_m["target_acc"],
            "bias_nontarget": bias_m["nontarget_acc"],
            "bias_solves": bias_solves,
            "last_target": last_m["target_acc"],
            "last_nontarget": last_m["nontarget_acc"],
            "last_solves": last_solves,
            "gradient_target": grad_m["target_acc"],
            "gradient_nontarget": grad_m["nontarget_acc"],
            "gradient_gain": gain,
            "gradient_ok": grad_ok,
            "passed": passed,
        }
        gate_log.append(entry)
        print(
            f"  gate #{record['candidate_index']} {record['family']}: "
            f"target {corrupted_gate['target_acc']:.2f} -> bias {bias_m['target_acc']:.2f}, "
            f"last {last_m['target_acc']:.2f}, grad {grad_m['target_acc']:.2f} "
            f"({'keep' if passed else 'reject'})"
        )
        if passed and winner is None:
            winner = record
            fitted = {
                "bias": repairs["bias"],
                "last_layer": repairs["last_layer"],
                "gradient": repairs["gradient"],
                "gradient_locations": repairs["gradient_locations"],
                "corrupted_gate": _public(corrupted_gate),
            }
            break

    _write_json(results / "corruption_gate.json", gate_log)
    if winner is None:
        _write_json(
            results / "phase1_summary.json",
            {
                "status": "no_corruption_passed_the_difficulty_gate",
                "n_candidates": len(candidates),
                "n_filter_passed": len(eligible),
                "n_gated": len(gate_log),
                "protocol_version": config.PROTOCOL_VERSION,
                "amendments": config.AMENDMENTS,
            },
        )
        raise RuntimeError(
            "No corruption was selective, nontrivial and gradient-repairable under the pre-registered gate. "
            "See results/corruption_candidates.csv and results/corruption_gate.json."
        )

    load_state(probe, winner["state"])
    corrupted_state = {key: value.detach().cpu().clone() for key, value in probe.state_dict().items()}
    _save_model(
        checkpoints / "corrupted.pt",
        probe,
        {"decimals": decimals, "candidate_index": winner["candidate_index"], "family": winner["family"], "detail": winner["detail"]},
    )
    _save_model(checkpoints / "baseline_bias.pt", fitted["bias"], {"fit_on": "dev_diag+inlexicon"})
    _save_model(checkpoints / "baseline_last_layer.pt", fitted["last_layer"], {"fit_on": "dev_diag+inlexicon"})
    _save_model(checkpoints / "baseline_gradient.pt", fitted["gradient"], {"fit_on": "dev_diag+inlexicon", "locations": fitted["gradient_locations"]})

    print("Scoring held-out test stems once...")
    methods = {
        "original": trained.model,
        "corrupted": probe,
        "output_bias": fitted["bias"],
        "last_layer": fitted["last_layer"],
        "gradient": fitted["gradient"],
    }
    # Reload original quantized weights into a fresh module. `trained.model` is that checkpoint;
    # probe currently holds the corruption. Bias/last/gradient are separate copies.
    test_rows = []
    test_cells = {}
    for name, module in methods.items():
        scored = summarize(test, module, "test")
        test_cells[name] = scored["cells"]
        delta = state_delta(corrupted_state, {k: v.detach().cpu() for k, v in module.state_dict().items()})
        if name == "original":
            delta = state_delta(corrupted_state, corrupted_state)
            edited = 0
        elif name == "corrupted":
            edited = 0
            delta = {"n_changed": 0, "l2": 0.0, "mean_abs_changed": 0.0}
        else:
            edited = delta["n_changed"]
        test_rows.append(
            {
                "method": name,
                "target_acc": scored["target_acc"],
                "nontarget_acc": scored["nontarget_acc"],
                "nontarget_macro_acc": scored["nontarget_macro_acc"],
                "example_acc": scored["example_acc"],
                "macro_condition_acc": scored["macro_condition_acc"],
                "n_changed": edited,
                "l2": 0.0 if name in {"original", "corrupted"} else delta["l2"],
                "api_cost_usd": 0.0,
                "target_predictions": scored["target_predictions"],
                "by_class": scored["by_class"],
            }
        )
    random_models = random_hidden_edits(probe)
    random_metrics = []
    for seed_offset, module in enumerate(random_models):
        scored = summarize(test, module, "test")
        delta = state_delta(corrupted_state, {k: v.detach().cpu() for k, v in module.state_dict().items()})
        random_metrics.append(
            {
                "seed": config.SEED + 1000 + seed_offset,
                "target_acc": scored["target_acc"],
                "nontarget_acc": scored["nontarget_acc"],
                "example_acc": scored["example_acc"],
                "n_changed": delta["n_changed"],
                "l2": delta["l2"],
            }
        )
    pd.DataFrame(random_metrics).to_csv(results / "random_edits.csv", index=False)
    test_rows.append(
        {
            "method": "random_mean",
            "target_acc": float(np.mean([row["target_acc"] for row in random_metrics])),
            "nontarget_acc": float(np.mean([row["nontarget_acc"] for row in random_metrics])),
            "nontarget_macro_acc": None,
            "example_acc": float(np.mean([row["example_acc"] for row in random_metrics])),
            "macro_condition_acc": None,
            "n_changed": float(np.mean([row["n_changed"] for row in random_metrics])),
            "l2": float(np.mean([row["l2"] for row in random_metrics])),
            "api_cost_usd": 0.0,
            "target_predictions": {},
            "by_class": {},
        }
    )
    pd.DataFrame(
        [
            {
                "method": row["method"],
                "target_acc": row["target_acc"],
                "nontarget_acc": row["nontarget_acc"],
                "example_acc": row["example_acc"],
                "n_changed": row["n_changed"],
                "l2": row["l2"],
                "api_cost_usd": row["api_cost_usd"],
            }
            for row in test_rows
        ]
    ).to_csv(results / "metrics.csv", index=False)
    test_cells["original"].to_csv(results / "test_cells_original.csv", index=False)
    test_cells["corrupted"].to_csv(results / "test_cells_corrupted.csv", index=False)
    plot_heatmaps(test_cells["original"], test_cells["corrupted"], figures / "accuracy_heatmap_test.png")
    plot_repair_bars(
        [
            {"method": row["method"], "target_acc": row["target_acc"], "nontarget_acc": row["nontarget_acc"]}
            for row in test_rows
        ],
        figures / "baseline_comparison.png",
    )

    print("Checking a function-preserving hidden-neuron permutation...")
    permuted, perms = permute_hidden_neurons(probe, seed=config.SEED + 7)
    x_all, _ = matrix_from_frame(frame)
    max_diff = permutation_max_abs_diff(probe, permuted, x_all)
    if max_diff > 1e-4:
        raise RuntimeError(f"permutation changed logits by {max_diff}, which is above floating-point tolerance")
    _save_model(checkpoints / "corrupted_permuted.pt", permuted, {"permutation": perms, "max_abs_logit_diff": max_diff})
    print(f"  max absolute logit difference={max_diff:.3e}")

    print("Measuring prompt tokens and reserved API cost...")
    diag_view = summarize(diag, probe, "diagnostic")
    architecture = architecture_record(probe)
    prompt = build_prompt(corrupted_state, architecture, diag_view["scored"], decimals, feedback=None)
    roundtrip = parse_serialized_state(serialize_state(corrupted_state, decimals), decimals)
    roundtrip_error = max(float(torch.max(torch.abs(roundtrip[name] - corrupted_state[name]))) for name in roundtrip)
    mistakes = []
    wrong = diag_view["scored"][~diag_view["scored"]["correct"]]
    for row in wrong.head(8).itertuples(index=False):
        mistakes.append(f"{row.phrase}: true {row.exponent}, predicted {row.pred_exponent}")
    feedback_prompt = build_prompt(
        corrupted_state,
        architecture,
        diag_view["scored"],
        decimals,
        feedback={
            "target_acc": diag_view["target_acc"],
            "nontarget_acc": diag_view["nontarget_acc"],
            "nontarget_macro_acc": diag_view["nontarget_macro_acc"],
            "mistakes": mistakes,
            "errors": [],
            "hypothesis": "Placeholder hypothesis used only to measure the feedback prompt. " * 12,
        },
    )
    (results / "prompt_one_shot.txt").write_text(prompt, encoding="utf-8")
    one_shot_tokens = count_tokens(prompt)
    feedback_tokens = count_tokens(feedback_prompt) + 2000
    safety = config.API["input_token_safety_factor"]
    call_plan = [
        ("sol_one_shot", "gpt-6.2-sol", one_shot_tokens),
        ("astra_one_shot", "gpt-6.2-astra", one_shot_tokens),
        ("sol_feedback", "gpt-6.2-sol", feedback_tokens),
        ("astra_feedback", "gpt-6.2-astra", feedback_tokens),
        ("sol_permuted", "gpt-6.2-sol", one_shot_tokens),
        ("astra_permuted", "gpt-6.2-astra", one_shot_tokens),
    ]
    cost_rows = []
    for call, model_name, raw_tokens in call_plan:
        reserved = math.ceil(raw_tokens * safety)
        cost_rows.append(
            {
                "call": call,
                "model": model_name,
                "raw_input_tokens": raw_tokens,
                "reserved_input_tokens": reserved,
                "max_output_tokens": config.API["max_output_tokens"],
                "worst_case_usd": estimate_request_cost(model_name, reserved),
            }
        )
    plot_cost_estimates(cost_rows, figures / "api_cost_reservation.png")
    primary = sum(row["worst_case_usd"] for row in cost_rows[:2])
    everything = sum(row["worst_case_usd"] for row in cost_rows)

    print("Testing the budget manager with mocked usage...")
    with tempfile.TemporaryDirectory() as tmp:
        self_test_notes = run_self_tests(Path(tmp))
    ledger = results / "api_usage.csv"
    if not ledger.exists():
        BudgetManager(ledger)
    edits_path = results / "edits.jsonl"
    if not edits_path.exists():
        edits_path.write_text("", encoding="utf-8")

    dev_gate_corrupted = summarize(gate_frame, probe, "dev_gate")
    report = {
        "status": "phase1_complete_api_not_called",
        "protocol_version": config.PROTOCOL_VERSION,
        "amendments": config.AMENDMENTS,
        "seed": config.SEED,
        "seconds": time.perf_counter() - started,
        "dataset": summary,
        "architecture": architecture,
        "hidden_search": search,
        "training_seconds": trained.seconds,
        "epochs": trained.epochs_ran,
        "quantize_decimals": decimals,
        "quantize_trials": quant_notes,
        "serialization_roundtrip_max_abs": roundtrip_error,
        "original_test": _public(original_test),
        "corruption": {
            "n_candidates": len(candidates),
            "n_filter_passed": len(eligible),
            "n_gated": len(gate_log),
            "selected_index": winner["candidate_index"],
            "family": winner["family"],
            "detail": winner["detail"],
            "band": winner["band"],
            "dev_gate": _public(dev_gate_corrupted),
            "liten_acc": winner["liten_acc"],
            "n_changed": winner["n_changed"],
            "l2": winner["l2"],
        },
        "gate": gate_log,
        "test_methods": test_rows,
        "random_edits": random_metrics,
        "permutation_max_abs_logit_diff": max_diff,
        "diagnostic_rows": int(len(diag)),
        "one_shot_input_tokens": one_shot_tokens,
        "feedback_input_tokens_with_buffer": feedback_tokens,
        "planned_calls": cost_rows,
        "worst_case_primary_pair_usd": primary,
        "worst_case_all_six_usd": everything,
        "budget_usd": config.API["max_budget_usd"],
        "budget_sufficient_for_six": everything <= config.API["max_budget_usd"],
        "budget_self_test": self_test_notes,
        "api_enabled": False,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "n_parameters": parameter_count(probe),
        },
    }
    _write_json(results / "phase1_summary.json", report)
    print(
        f"Phase 1 finished in {report['seconds']:.1f}s. "
        f"One-shot prompt={one_shot_tokens} tokens. "
        f"Worst-case primary pair=${primary:.2f}, all six=${everything:.2f}. "
        "No API call was made."
    )
    return report


def _phase2_path(results: Path, model_name: str) -> Path:
    return results / "api_responses" / f"one_shot_{model_name.replace('.', '_')}.json"


def _repair_category(valid: bool, before_target: float, after_target: float, after_nontarget: float) -> str:
    if not valid:
        return "no_useful_repair"
    if after_target >= 1.0 - 1e-9 and after_nontarget >= 1.0 - 1e-9:
        return "full_repair"
    if after_target > before_target + 1e-12 and after_nontarget >= 0.98:
        return "selective_partial_repair"
    if after_target > before_target + 1e-12:
        return "repair_with_regressions"
    return "no_useful_repair"


def _diagnosis_flags(proposal: dict | None) -> dict:
    diagnosis = {} if proposal is None else proposal.get("diagnosis") or {}
    parts = [
        str(diagnosis.get("suspected_component", "")),
        str(diagnosis.get("suspected_failure_mechanism", "")),
        str(diagnosis.get("supporting_evidence", "")),
        "" if proposal is None else str(proposal.get("hypothesis", "")),
        "" if proposal is None else str(proposal.get("explanation", "")),
    ]
    text = "\n".join(parts)
    lowered = text.lower()
    mentions_gender = "uter" in lowered and "neuter" in lowered
    mentions_same = any(
        token in lowered
        for token in (
            "identical column",
            "columns are identical",
            "columns are equal",
            "duplicate",
            "copied",
            "same column",
            "same weights",
            "equal columns",
            "weight vector is the same",
            "indistinguishable",
            "no difference between",
        )
    )
    mentions_first = any(token in lowered for token in ("fc1", "first layer", "input weight", "input column", "feature 0", "feature 1"))
    return {
        "mentions_gender": mentions_gender,
        "mentions_sameness": mentions_same,
        "mentions_first_layer": mentions_first,
        "heuristic_identifies_duplicate_columns": bool(mentions_gender and mentions_same and mentions_first),
    }


def _form_examples(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    keys = ["stem", "gender", "number", "definiteness"]
    columns = keys + ["phrase", "exponent", "pred_exponent", "correct", "is_target"]
    merged = before[columns].merge(after[keys + ["pred_exponent", "correct"]], on=keys, suffixes=("_before", "_after"))
    repaired = merged[(~merged["correct_before"]) & (merged["correct_after"])]
    damaged = merged[(merged["correct_before"]) & (~merged["correct_after"])]

    def pack(frame: pd.DataFrame) -> list[dict]:
        rows = []
        for _, row in frame.head(2).iterrows():
            rows.append(
                {
                    "phrase": row["phrase"],
                    "true_exponent": row["exponent"],
                    "before": row["pred_exponent_before"],
                    "after": row["pred_exponent_after"],
                    "target": bool(row["is_target"]),
                }
            )
        return rows

    return {"repaired": pack(repaired), "damaged": pack(damaged), "n_repaired": int(len(repaired)), "n_damaged": int(len(damaged))}


def _evaluate_proposal(checkpoint: dict, corrupted_state: dict, proposal, parse_error: str | None, test, diagnostic) -> dict:
    applied, errors = (None, [parse_error or "no proposal"])
    if proposal is not None:
        applied, errors = validate_and_apply(corrupted_state, proposal)
    probe = copy_model_from_checkpoint(checkpoint)
    before_test = summarize(test, probe, "test")
    before_diag = summarize(diagnostic, probe, "diagnostic")
    if applied is not None:
        load_state(probe, applied)
    after_test = summarize(test, probe, "test")
    after_diag = summarize(diagnostic, probe, "diagnostic")
    edited_state = {key: value.detach().cpu() for key, value in probe.state_dict().items()}
    delta = state_delta(corrupted_state, edited_state)
    max_abs = 0.0
    if applied is not None:
        for key, before in corrupted_state.items():
            max_abs = max(max_abs, float((applied[key] - before).abs().max()))
    flags = _diagnosis_flags(proposal)
    return {
        "valid": applied is not None,
        "errors": errors,
        "diagnosis": None if proposal is None else proposal.get("diagnosis"),
        "hypothesis": None if proposal is None else proposal.get("hypothesis"),
        "explanation": None if proposal is None else proposal.get("explanation"),
        "edits": None if proposal is None else proposal.get("edits"),
        "diagnosis_flags": flags,
        "n_changed": int(delta["n_changed"]) if applied is not None else 0,
        "l2": float(delta["l2"]) if applied is not None else 0.0,
        "max_abs": max_abs,
        "per_tensor": delta["per_tensor"] if applied is not None else {},
        "test_before": _public(before_test),
        "test": _public(after_test),
        "diagnostic_before": _public(before_diag),
        "diagnostic": _public(after_diag),
        "examples": _form_examples(before_test["scored"], after_test["scored"]),
        "category": _repair_category(
            applied is not None,
            before_test["target_acc"],
            after_test["target_acc"],
            after_test["nontarget_acc"],
        ),
        "applied_state": applied,
    }


def _comparison_rows(results: Path, outcomes: list[dict], diagnostic_found: bool) -> list[dict]:
    metrics = pd.read_csv(results / "metrics.csv")
    labels = {
        "corrupted": "Corrupted baseline",
        "random_mean": "Random edits",
        "output_bias": "Output-bias repair",
        "last_layer": "Last-layer repair",
        "gradient": "Gradient repair",
    }
    rows = []
    for method, label in labels.items():
        source = metrics[metrics["method"] == method].iloc[0]
        rows.append(
            {
                "method": label,
                "diagnosis": "—",
                "target_acc": float(source["target_acc"]),
                "nontarget_acc": float(source["nontarget_acc"]),
                "n_changed": int(source["n_changed"]),
                "api_cost_usd": 0.0,
                "color_key": "random" if method == "random_mean" else method,
            }
        )
    rows.insert(
        1,
        {
            "method": "Duplicate-column diagnostic",
            "diagnosis": "exact fc1 gender-column copy" if diagnostic_found else "gender columns not exact duplicates",
            "target_acc": None,
            "nontarget_acc": None,
            "n_changed": 0,
            "api_cost_usd": 0.0,
            "color_key": "diagnostic",
        },
    )
    for outcome in outcomes:
        flags = outcome.get("diagnosis_flags") or {}
        short = outcome.get("diagnosis_judgement")
        if not short:
            short = "duplicate gender columns" if flags.get("heuristic_identifies_duplicate_columns") else "other or incomplete account"
        if outcome.get("status") == "stopped":
            short = "no proposal"
        rows.append(
            {
                "method": outcome["model"],
                "diagnosis": short,
                "target_acc": None if outcome.get("test") is None else outcome["test"]["target_acc"],
                "nontarget_acc": None if outcome.get("test") is None else outcome["test"]["nontarget_acc"],
                "n_changed": int(outcome.get("n_changed") or 0),
                "api_cost_usd": outcome.get("actual_cost_usd"),
                "color_key": "sol" if "sol" in outcome["model"] else "astra",
                "category": outcome.get("category"),
            }
        )
    return rows


def _write_phase2(results: Path, outcomes: list[dict], budget: BudgetManager, status: str) -> dict:
    diagnostic = json.loads((results / "column_diagnostic.json").read_text(encoding="utf-8"))
    found = bool(diagnostic["corrupted"]["gender_columns_exact_duplicate"])
    rows = _comparison_rows(results, outcomes, found)
    plot_rows = []
    for row in rows:
        if row["target_acc"] is None:
            continue
        plot_rows.append({**row, "label": row["method"].replace(" repair", "").replace(" baseline", "")})
    plot_phase2_scatter(plot_rows, results / "figures" / "phase2_comparison.png")
    pd.DataFrame(rows).drop(columns=["color_key"], errors="ignore").to_csv(results / "phase2_comparison.csv", index=False)
    public_outcomes = []
    for outcome in outcomes:
        public = {key: value for key, value in outcome.items() if key != "applied_state"}
        public_outcomes.append(public)
    payload = {
        "status": status,
        "calls": len(outcomes),
        "spent_usd": budget.spent,
        "remaining_usd": budget.remaining(),
        "outcomes": public_outcomes,
        "comparison": rows,
        "feedback_enabled": False,
        "permutation_api_enabled": False,
    }
    _write_json(results / "phase2_summary.json", payload)
    return payload


def run_phase2(experiment_dir: Path | None = None, authorized: bool = False) -> dict:
    """One Sol call, then one Astra call. Saved responses are reloaded, not rebilled.

    Feedback and the permuted checkpoint stay off the API. An invalid proposal is
    scored as a failed attempt. The corrupted checkpoint file is only read.
    """
    experiment_dir = Path(experiment_dir) if experiment_dir else EXPERIMENT_DIR
    results = experiment_dir / "results"
    if config.API["run_feedback"] or config.API["run_permutation"]:
        raise RuntimeError("Feedback and permutation API calls are not authorized in this round.")
    manifest_path = results / "freeze_manifest.json"
    prompt_path = results / "prompt_one_shot_phase2.txt"
    if not manifest_path.exists() or not prompt_path.exists():
        if not authorized and not config.API["api_enabled"]:
            return {"status": "api_disabled", "calls": 0, "message": "API execution is disabled. No request was sent."}
        raise RuntimeError("Run local validation before an API call so the prompt and checkpoint are frozen.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prompt = prompt_path.read_text(encoding="utf-8")
    if sha256_text(prompt) != manifest["prompt_sha256"]:
        raise RuntimeError("The frozen prompt no longer matches its hash.")
    if sha256_text(json.dumps(EDIT_SCHEMA, sort_keys=True)) != manifest["schema_sha256"]:
        raise RuntimeError("The edit schema changed after the prompt was frozen.")
    corrupted_path = results / "checkpoints" / "corrupted.pt"
    if sha256_bytes(corrupted_path.read_bytes()) != manifest["corrupted_pt_sha256"]:
        raise RuntimeError("The corrupted checkpoint changed after it was frozen.")
    checkpoint = torch.load(corrupted_path, map_location="cpu", weights_only=False)
    model = copy_model_from_checkpoint(checkpoint)
    corrupted_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    frame = build_dataset(config.SEED)
    diagnostic = frame[frame["split"].isin(["dev_diag", "inlexicon"])].reset_index(drop=True)
    test = frame[frame["split"] == "test"].reset_index(drop=True)
    budget = BudgetManager(results / "api_usage.csv")
    (results / "api_responses").mkdir(parents=True, exist_ok=True)
    outcomes = []
    missing = []
    for model_name in config.API["models"]:
        path = _phase2_path(results, model_name)
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("evaluation") is None:
                evaluation = _evaluate_proposal(
                    checkpoint,
                    corrupted_state,
                    saved["api"].get("proposal"),
                    saved["api"].get("parse_error"),
                    test,
                    diagnostic,
                )
                evaluation.pop("applied_state", None)
                evaluation.update(saved.get("evaluation_stub") or {})
                saved["evaluation"] = evaluation
                _write_json(path, saved)
            outcomes.append(saved["evaluation"])
            print(f"{model_name}: loaded saved response {saved.get('response_id')}. No new API call.")
        else:
            missing.append(model_name)
    if not missing:
        after = sha256_bytes(corrupted_path.read_bytes())
        payload = _write_phase2(results, outcomes, budget, "finished")
        payload["checkpoint_unchanged"] = after == manifest["corrupted_pt_sha256"]
        _write_json(results / "phase2_summary.json", payload)
        return payload
    if not authorized and not config.API["api_enabled"]:
        if outcomes:
            return _write_phase2(results, outcomes, budget, "saved_partial_api_disabled")
        return {"status": "api_disabled", "calls": 0, "message": "API execution is disabled. No request was sent."}

    key = load_api_key(experiment_dir)
    print(f"API key loaded from the environment file ({len(key)} characters). The key is not printed.")
    raw_tokens = count_tokens(prompt)
    reserved = math.ceil(raw_tokens * config.API["input_token_safety_factor"])
    estimates = {name: estimate_request_cost(name, reserved) for name in missing}
    print(
        f"Frozen prompt tokens={raw_tokens}. Reserved input tokens={reserved}. "
        f"Remaining budget=${budget.remaining():.4f}."
    )
    for name, estimate in estimates.items():
        print(f"  worst case {name}: ${estimate:.4f}")
        if estimate > budget.remaining() + 1e-12:
            raise BudgetExceeded(f"{name} worst case ${estimate:.4f} exceeds remaining ${budget.remaining():.4f}.")
    if sum(estimates.values()) > budget.remaining() + 1e-12:
        raise BudgetExceeded("The two remaining one-shot calls do not both fit in the remaining budget.")

    status = "finished"
    for model_name in missing:
        call_id = f"one_shot-{model_name}"
        print(f"Calling {model_name} once. Reasoning={config.API['reasoning_effort']}, max_output_tokens={config.API['max_output_tokens']}.")
        try:
            response = execute_edit(
                prompt,
                model_name,
                budget,
                call_id=call_id,
                experiment="one_shot",
                api_enabled=True,
                start=experiment_dir,
            )
        except (ApiHalted, BudgetExceeded, ApiDisabled) as exc:
            print(f"{model_name} stopped before a usable response: {type(exc).__name__}. No retry.")
            status = "stopped"
            break
        stub = {
            "model": model_name,
            "experiment": "one_shot",
            "status": response.get("status"),
            "response_id": response.get("response_id"),
            "model_revision": response.get("model_revision"),
            "usage": response.get("usage"),
            "estimated_cost_usd": response.get("estimated_cost_usd"),
            "actual_cost_usd": response.get("actual_cost_usd"),
            "prompt_sha256": manifest["prompt_sha256"],
        }
        record = {
            "call_id": call_id,
            "response_id": response.get("response_id"),
            "model_revision": response.get("model_revision"),
            "prompt_sha256": manifest["prompt_sha256"],
            "api": {
                "status": response.get("status"),
                "text": response.get("text"),
                "proposal": response.get("proposal"),
                "parse_error": response.get("parse_error"),
                "usage": response.get("usage"),
                "estimated_cost_usd": response.get("estimated_cost_usd"),
                "actual_cost_usd": response.get("actual_cost_usd"),
                "raw": response.get("raw"),
            },
            "evaluation_stub": stub,
            "evaluation": None,
        }
        response_path = _phase2_path(results, model_name)
        _write_json(response_path, record)
        evaluation = _evaluate_proposal(
            checkpoint,
            corrupted_state,
            response.get("proposal"),
            response.get("parse_error"),
            test,
            diagnostic,
        )
        applied_state = evaluation.pop("applied_state")
        evaluation.update(stub)
        record["evaluation"] = evaluation
        _write_json(response_path, record)
        if applied_state is not None:
            edited = copy_model_from_checkpoint(checkpoint)
            load_state(edited, applied_state)
            _save_model(results / "checkpoints" / f"one_shot_{model_name.replace('.', '_')}.pt", edited, {"source": call_id})
        with (results / "edits.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(evaluation)) + "\n")
        outcomes.append(evaluation)
        test_metrics = evaluation["test"]
        print(
            f"{model_name}: valid={evaluation['valid']} category={evaluation['category']} "
            f"target={test_metrics['target_acc']:.3f} non-target={test_metrics['nontarget_acc']:.3f} "
            f"changed={evaluation['n_changed']} cost=${evaluation['actual_cost_usd']:.6f} "
            f"spent=${budget.spent:.6f} remaining=${budget.remaining():.6f}"
        )
        _write_phase2(results, outcomes, budget, "in_progress")
    after = sha256_bytes(corrupted_path.read_bytes())
    payload = _write_phase2(results, outcomes, budget, status)
    payload["checkpoint_unchanged"] = after == manifest["corrupted_pt_sha256"]
    payload["prompt_tokens"] = count_tokens(prompt)
    _write_json(results / "phase2_summary.json", payload)
    return payload


def copy_model_from_checkpoint(checkpoint: dict):
    from .model import GrammarMLP

    arch = checkpoint["architecture"]
    model = GrammarMLP(arch["in_dim"], arch["hidden"], arch["out_dim"])
    load_state(model, checkpoint["state_dict"])
    model.eval()
    return model


if __name__ == "__main__":
    run_phase1()
