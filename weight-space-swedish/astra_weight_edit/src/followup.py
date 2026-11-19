"""Follow-up: constrained gradient, permutation transfer, and two authorized calls.

Phase 1 and Phase 2 artifacts are read, not replaced. New outputs go under
results/followup and new response files.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from . import config
from .api_client import (
    ApiDisabled,
    ApiHalted,
    BudgetExceeded,
    BudgetManager,
    build_prompt,
    count_tokens,
    estimate_request_cost,
    execute_edit,
    load_api_key,
    validate_and_apply,
)
from .baselines import repair_gradient_constrained
from .column_diagnostic import diagnose_network
from .evaluation import permutation_max_abs_diff, permute_hidden_neurons, state_delta, summarize
from .grammar import EXPONENTS, FEATURE_NAMES, build_dataset, matrix_from_frame
from .local_checks import sha256_bytes, sha256_text
from .model import GrammarMLP, load_state
from .pipeline import (
    _evaluate_proposal,
    _jsonable,
    _save_model,
    _write_json,
    copy_model_from_checkpoint,
)

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]


def _load_checkpoint(path: Path) -> tuple[GrammarMLP, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = copy_model_from_checkpoint(checkpoint)
    return model, checkpoint


def _state(model: GrammarMLP) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _fraction(summary: dict, key: str) -> str:
    count_key = "target_n" if key == "target_acc" else "nontarget_n"
    correct = int(round(summary[key] * summary[count_key]))
    return f"{correct}/{summary[count_key]}"


def _constraint_report(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> dict:
    delta = state_delta(before, after)
    max_abs = 0.0
    max_value = 0.0
    for key in before:
        change = (after[key] - before[key]).abs()
        if change.numel():
            max_abs = max(max_abs, float(change.max()))
        if after[key].numel():
            max_value = max(max_value, float(after[key].abs().max()))
    limits = config.EDIT_LIMITS
    return {
        "n_changed": int(delta["n_changed"]),
        "l2": float(delta["l2"]),
        "max_abs_delta": max_abs,
        "max_abs_value": max_value,
        "within_parameter_budget": int(delta["n_changed"]) <= limits["max_edit_params"],
        "within_l2": float(delta["l2"]) <= limits["max_frobenius"] + 1e-6,
        "within_abs_delta": max_abs <= limits["max_abs_delta"] + 1e-6,
        "within_abs_value": max_value <= limits["max_abs_value"] + 1e-6,
        "satisfies_api_constraints": (
            int(delta["n_changed"]) <= limits["max_edit_params"]
            and float(delta["l2"]) <= limits["max_frobenius"] + 1e-6
            and max_abs <= limits["max_abs_delta"] + 1e-6
            and max_value <= limits["max_abs_value"] + 1e-6
        ),
    }


def _scores(model: GrammarMLP, test, diagnostic) -> dict:
    heldout = summarize(test, model, "test")
    dev = summarize(diagnostic, model, "diagnostic")
    return {
        "test_target": _fraction(heldout, "target_acc"),
        "test_nontarget": _fraction(heldout, "nontarget_acc"),
        "test_target_acc": heldout["target_acc"],
        "test_nontarget_acc": heldout["nontarget_acc"],
        "diagnostic_target": _fraction(dev, "target_acc"),
        "diagnostic_nontarget": _fraction(dev, "nontarget_acc"),
    }


def _full_repair(scores: dict) -> bool:
    return scores["test_target"] == "15/15" and scores["test_nontarget"] == "121/121"


def _scale_update(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor], max_l2: float) -> tuple[dict[str, torch.Tensor], float]:
    square = 0.0
    deltas = {}
    for key in before:
        delta = after[key] - before[key]
        deltas[key] = delta
        square += float((delta * delta).sum())
    norm = square**0.5
    scale = 1.0 if norm <= max_l2 + 1e-12 or norm == 0.0 else max_l2 / norm
    scaled = {key: before[key] + deltas[key] * scale for key in before}
    return scaled, scale


def _invert(perm: list[int]) -> list[int]:
    inverse = [0] * len(perm)
    for new_index, old_index in enumerate(perm):
        inverse[old_index] = new_index
    return inverse


def transfer_proposal(proposal: dict, permutation: dict) -> dict:
    """Move hidden-unit indices. Input features and output classes stay put."""
    inverse_h1 = _invert(permutation["h1"])
    inverse_h2 = _invert(permutation["h2"])
    transferred = json.loads(json.dumps(proposal))
    for edit in transferred.get("edits") or []:
        name = edit.get("tensor")
        indices = []
        for index in edit.get("indices") or []:
            if name == "fc1.weight":
                indices.append([inverse_h1[index[0]], index[1]])
            elif name == "fc1.bias":
                indices.append([inverse_h1[index[0]]])
            elif name == "fc2.weight":
                indices.append([inverse_h2[index[0]], inverse_h1[index[1]]])
            elif name == "fc2.bias":
                indices.append([inverse_h2[index[0]]])
            elif name == "out.weight":
                indices.append([index[0], inverse_h2[index[1]]])
            elif name == "out.bias":
                indices.append(list(index))
            else:
                raise RuntimeError(f"cannot transfer tensor {name}")
        edit["indices"] = indices
    return transferred


def run_local_followup(experiment_dir: Path | None = None) -> dict:
    experiment_dir = Path(experiment_dir) if experiment_dir else EXPERIMENT_DIR
    results = experiment_dir / "results"
    follow = results / "followup"
    follow.mkdir(parents=True, exist_ok=True)
    checkpoints = results / "checkpoints"
    corrupted_path = checkpoints / "corrupted.pt"
    permuted_path = checkpoints / "corrupted_permuted.pt"
    before_hashes = {
        "corrupted": sha256_bytes(corrupted_path.read_bytes()),
        "permuted": sha256_bytes(permuted_path.read_bytes()),
        "gradient": sha256_bytes((checkpoints / "baseline_gradient.pt").read_bytes()),
    }
    corrupted, corrupted_ckpt = _load_checkpoint(corrupted_path)
    gradient, _ = _load_checkpoint(checkpoints / "baseline_gradient.pt")
    permuted, permuted_ckpt = _load_checkpoint(permuted_path)
    frame = build_dataset(config.SEED)
    test = frame[frame["split"] == "test"].reset_index(drop=True)
    diagnostic = frame[frame["split"].isin(["dev_diag", "inlexicon"])].reset_index(drop=True)
    features, _ = matrix_from_frame(frame)
    x_diag, y_diag = matrix_from_frame(diagnostic)

    corrupted_state = _state(corrupted)
    gradient_state = _state(gradient)
    unconstrained = _scores(gradient, test, diagnostic)
    unconstrained_limits = _constraint_report(corrupted_state, gradient_state)
    scaled_state, scale = _scale_update(corrupted_state, gradient_state, config.EDIT_LIMITS["max_frobenius"])
    scaled_model = copy_model_from_checkpoint(corrupted_ckpt)
    load_state(scaled_model, scaled_state)
    scaled_scores = _scores(scaled_model, test, diagnostic)
    scaled_limits = _constraint_report(corrupted_state, scaled_state)
    _save_model(checkpoints / "baseline_gradient_scaled.pt", scaled_model, {"scale": scale, "reference": "uniform scaling of the saved 40-parameter repair"})
    projected = None
    if not _full_repair(scaled_scores):
        projected_model, locations, info = repair_gradient_constrained(corrupted, x_diag, y_diag)
        projected_state = _state(projected_model)
        projected = {
            "scores": _scores(projected_model, test, diagnostic),
            "constraints": _constraint_report(corrupted_state, projected_state),
            "optimization": info,
            "n_coordinates": len(locations),
            "coordinates_on_fc1": sum(1 for row in locations if row["tensor"] == "fc1.weight"),
            "full_repair": _full_repair(_scores(projected_model, test, diagnostic)),
        }
        projected["full_repair"] = _full_repair(projected["scores"])
        _save_model(
            checkpoints / "baseline_gradient_constrained.pt",
            projected_model,
            {"method": "projected Adam on the 40 largest initial diagnostic gradients", "info": info},
        )
    chosen = "scaled" if _full_repair(scaled_scores) else "projected"
    gradient_report = {
        "note": "The held-out test was already inspected in Phase 1. This is a constraint correction, not a new blind test.",
        "unconstrained": {"scores": unconstrained, "constraints": unconstrained_limits},
        "scaled": {"scale": scale, "scores": scaled_scores, "constraints": scaled_limits, "full_repair": _full_repair(scaled_scores)},
        "projected": projected,
        "reported_method": chosen,
    }

    fresh, fresh_perm = permute_hidden_neurons(corrupted, config.SEED + 7)
    saved_gap = permutation_max_abs_diff(corrupted, permuted, features)
    fresh_gap = permutation_max_abs_diff(corrupted, fresh, features)
    weight_gap = max(float((fresh.state_dict()[key] - permuted.state_dict()[key]).abs().max()) for key in fresh.state_dict())
    saved_scores = _scores(permuted, test, diagnostic)
    corrupted_scores = _scores(corrupted, test, diagnostic)
    predictions_match = saved_scores["test_target"] == corrupted_scores["test_target"] and saved_scores["test_nontarget"] == corrupted_scores["test_nontarget"]
    gender = diagnose_network(_state(permuted))
    stored_perm = permuted_ckpt.get("extra", {}).get("permutation")
    permutation = stored_perm or fresh_perm
    if stored_perm is not None and (stored_perm.get("h1") != fresh_perm["h1"] or stored_perm.get("h2") != fresh_perm["h2"]):
        raise RuntimeError("The permutation stored in the checkpoint does not match seed 49.")
    permutation_report = {
        "seed": config.SEED + 7,
        "checkpoint_sha256": before_hashes["permuted"],
        "saved_max_abs_logit_diff_all_768": saved_gap,
        "regenerated_max_abs_logit_diff": fresh_gap,
        "saved_vs_regenerated_max_abs_weight": weight_gap,
        "predictions_match_corrupted": predictions_match,
        "corrupted_scores": corrupted_scores,
        "permuted_scores": saved_scores,
        "input_features_unchanged": list(FEATURE_NAMES),
        "output_classes_unchanged": list(EXPONENTS),
        "gender_columns_still_exact_duplicate": gender["gender_columns_exact_duplicate"],
        "permutation_source": "checkpoint extra" if permuted_ckpt.get("extra", {}).get("permutation") else "regenerated from seed",
    }
    if saved_gap > 1e-4 or weight_gap > 1e-5 or not predictions_match or not gender["gender_columns_exact_duplicate"]:
        raise RuntimeError(f"saved permutation failed verification: {permutation_report}")

    astra = json.loads((results / "api_responses" / "one_shot_gpt-6_2-astra.json").read_text(encoding="utf-8"))
    proposal = astra["api"]["proposal"]
    transferred = transfer_proposal(proposal, permutation)
    applied, errors = validate_and_apply(_state(permuted), transferred)
    if applied is None:
        raise RuntimeError(f"transferred Astra edit was rejected: {errors}")
    transferred_model = copy_model_from_checkpoint(permuted_ckpt)
    load_state(transferred_model, applied)
    original_applied, original_errors = validate_and_apply(corrupted_state, proposal)
    if original_applied is None:
        raise RuntimeError(f"original Astra edit no longer validates: {original_errors}")
    edited_original = copy_model_from_checkpoint(corrupted_ckpt)
    load_state(edited_original, original_applied)
    permuted_after_edit, _ = permute_hidden_neurons(edited_original, config.SEED + 7)
    function_gap = permutation_max_abs_diff(permuted_after_edit, transferred_model, features)
    transfer_scores = _scores(transferred_model, test, diagnostic)
    transfer_limits = _constraint_report(_state(permuted), _state(transferred_model))
    if function_gap > 1e-4 or transfer_scores["test_target"] != "11/15" or transfer_scores["test_nontarget"] != "121/121":
        raise RuntimeError(
            f"transferred edit did not reproduce Astra's repair: gap={function_gap} scores={transfer_scores} errors={errors}"
        )
    _save_model(checkpoints / "astra_transferred_permuted.pt", transferred_model, {"source": "index transfer of the saved Astra edit"})
    transfer_report = {
        "validation_errors": errors,
        "function_gap_vs_permute_then_edit": function_gap,
        "scores": transfer_scores,
        "constraints": transfer_limits,
        "n_edit_operations": len(transferred["edits"]),
    }
    report = {
        "gradient": gradient_report,
        "permutation": permutation_report,
        "astra_transfer": transfer_report,
        "checkpoint_hashes_unchanged": {
            name: sha256_bytes((checkpoints / file_name).read_bytes()) == digest
            for name, digest, file_name in (
                ("corrupted", before_hashes["corrupted"], "corrupted.pt"),
                ("permuted", before_hashes["permuted"], "corrupted_permuted.pt"),
                ("gradient", before_hashes["gradient"], "baseline_gradient.pt"),
            )
        },
    }
    _write_json(follow / "local_followup.json", report)
    _write_json(follow / "permutation.json", permutation)
    return report


def _response_path(results: Path, name: str) -> Path:
    return results / "api_responses" / name


def _call_one(
    results: Path,
    prompt: str,
    prompt_hash: str,
    model_name: str,
    call_id: str,
    file_name: str,
    checkpoint: dict,
    base_state: dict[str, torch.Tensor],
    test,
    diagnostic,
    max_output_tokens: int,
    budget: BudgetManager,
    experiment_dir: Path,
) -> dict:
    path = _response_path(results, file_name)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("evaluation") is None:
            evaluation = _evaluate_proposal(checkpoint, base_state, saved["api"].get("proposal"), saved["api"].get("parse_error"), test, diagnostic)
            evaluation.pop("applied_state", None)
            evaluation.update(saved.get("evaluation_stub") or {})
            saved["evaluation"] = evaluation
            _write_json(path, saved)
        print(f"{model_name} {call_id}: loaded saved response. No new API call.")
        return saved["evaluation"]
    key = load_api_key(experiment_dir)
    print(f"API key loaded ({len(key)} characters). The key is not printed.")
    raw_tokens = count_tokens(prompt)
    reserved = math.ceil(raw_tokens * config.API["input_token_safety_factor"])
    estimate = estimate_request_cost(model_name, reserved, max_output_tokens)
    print(
        f"{call_id}: prompt_tokens={raw_tokens} reserved_input={reserved} "
        f"output_cap={max_output_tokens} worst_case=${estimate:.4f} remaining=${budget.remaining():.4f}"
    )
    if estimate > budget.remaining() + 1e-12:
        raise BudgetExceeded(f"{call_id} worst case ${estimate:.4f} exceeds remaining ${budget.remaining():.4f}.")
    print(f"Calling {model_name} once. Reasoning={config.API['reasoning_effort']}, max_output_tokens={max_output_tokens}.")
    try:
        response = execute_edit(
            prompt,
            model_name,
            budget,
            call_id=call_id,
            experiment=call_id,
            api_enabled=True,
            start=experiment_dir,
            max_output_tokens=max_output_tokens,
            timeout_seconds=900 if max_output_tokens > config.API["max_output_tokens"] else None,
        )
    except (ApiHalted, BudgetExceeded, ApiDisabled) as exc:
        print(f"{call_id} stopped: {type(exc).__name__}. No retry.")
        raise
    stub = {
        "model": model_name,
        "call_id": call_id,
        "status": response.get("status"),
        "response_id": response.get("response_id"),
        "model_revision": response.get("model_revision"),
        "usage": response.get("usage"),
        "estimated_cost_usd": response.get("estimated_cost_usd"),
        "actual_cost_usd": response.get("actual_cost_usd"),
        "prompt_sha256": prompt_hash,
        "max_output_tokens": max_output_tokens,
    }
    record = {
        "call_id": call_id,
        "response_id": response.get("response_id"),
        "model_revision": response.get("model_revision"),
        "prompt_sha256": prompt_hash,
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
    _write_json(path, record)
    evaluation = _evaluate_proposal(checkpoint, base_state, response.get("proposal"), response.get("parse_error"), test, diagnostic)
    applied_state = evaluation.pop("applied_state")
    evaluation.update(stub)
    record["evaluation"] = evaluation
    _write_json(path, record)
    if applied_state is not None:
        edited = copy_model_from_checkpoint(checkpoint)
        load_state(edited, applied_state)
        _save_model(results / "checkpoints" / f"{call_id.replace('-', '_').replace('.', '_')}.pt", edited, {"source": call_id})
    with (results / "edits.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(evaluation)) + "\n")
    test_metrics = evaluation["test"]
    print(
        f"{call_id}: valid={evaluation['valid']} category={evaluation['category']} "
        f"target={test_metrics['target_acc']:.3f} non-target={test_metrics['nontarget_acc']:.3f} "
        f"changed={evaluation['n_changed']} l2={evaluation['l2']:.3f} "
        f"cost=${evaluation['actual_cost_usd']:.6f} remaining=${budget.remaining():.6f}"
    )
    return evaluation


def run_api_followup(experiment_dir: Path | None = None, authorized: bool = False) -> dict:
    """One Astra call on the permuted network, then one Sol call with a 12,000-token cap."""
    experiment_dir = Path(experiment_dir) if experiment_dir else EXPERIMENT_DIR
    results = experiment_dir / "results"
    if not authorized and not config.API["api_enabled"]:
        saved = []
        for name in ("permuted_gpt-6_2-astra.json", "extended_gpt-6_2-sol.json"):
            path = _response_path(results, name)
            if path.exists():
                saved.append(json.loads(path.read_text(encoding="utf-8")).get("evaluation"))
        if len(saved) == 2:
            return {"status": "loaded", "calls": 2, "outcomes": saved}
        return {"status": "api_disabled", "calls": len(saved), "message": "API execution is disabled. No request was sent."}
    local = json.loads((results / "followup" / "local_followup.json").read_text(encoding="utf-8"))
    if local["astra_transfer"]["scores"]["test_target"] != "11/15":
        raise RuntimeError("Refusing the permutation call until the transferred edit reproduces 11/15.")
    frame = build_dataset(config.SEED)
    test = frame[frame["split"] == "test"].reset_index(drop=True)
    diagnostic = frame[frame["split"].isin(["dev_diag", "inlexicon"])].reset_index(drop=True)
    permuted, permuted_ckpt = _load_checkpoint(results / "checkpoints" / "corrupted_permuted.pt")
    corrupted, corrupted_ckpt = _load_checkpoint(results / "checkpoints" / "corrupted.pt")
    permuted_state = _state(permuted)
    corrupted_state = _state(corrupted)
    diag_view = summarize(diagnostic, permuted, "diagnostic")
    permuted_prompt = build_prompt(permuted_state, permuted_ckpt["architecture"], diag_view["scored"], 3)
    prompt_path = results / "followup" / "prompt_permuted.txt"
    prompt_path.write_text(permuted_prompt, encoding="utf-8")
    original_prompt = (results / "prompt_one_shot_phase2.txt").read_text(encoding="utf-8")
    manifest = json.loads((results / "freeze_manifest.json").read_text(encoding="utf-8"))
    if sha256_text(original_prompt) != manifest["prompt_sha256"]:
        raise RuntimeError("The original frozen prompt no longer matches its hash.")
    if "permutation" in permuted_prompt.lower() or "unit 18" in permuted_prompt:
        raise RuntimeError("The permuted prompt contains material from the earlier edit.")
    budget = BudgetManager(results / "api_usage.csv")
    astra_estimate = estimate_request_cost("gpt-6.2-astra", math.ceil(count_tokens(permuted_prompt) * 1.2), 6000)
    sol_estimate = estimate_request_cost("gpt-6.2-sol", math.ceil(count_tokens(original_prompt) * 1.2), 12000)
    if astra_estimate + sol_estimate > budget.remaining() + 1e-12:
        raise BudgetExceeded(
            f"Worst case for the two authorized calls is ${astra_estimate + sol_estimate:.4f}, "
            f"above remaining ${budget.remaining():.4f}."
        )
    outcomes = []
    try:
        outcomes.append(
            _call_one(
                results,
                permuted_prompt,
                sha256_text(permuted_prompt),
                "gpt-6.2-astra",
                "permuted-one-shot-gpt-6.2-astra",
                "permuted_gpt-6_2-astra.json",
                permuted_ckpt,
                permuted_state,
                test,
                diagnostic,
                6000,
                budget,
                experiment_dir,
            )
        )
        outcomes.append(
            _call_one(
                results,
                original_prompt,
                manifest["prompt_sha256"],
                "gpt-6.2-sol",
                "extended-one-shot-gpt-6.2-sol",
                "extended_gpt-6_2-sol.json",
                corrupted_ckpt,
                corrupted_state,
                test,
                diagnostic,
                12000,
                budget,
                experiment_dir,
            )
        )
        status = "finished"
    except (ApiHalted, BudgetExceeded, ApiDisabled):
        status = "stopped"
    payload = {"status": status, "calls": len(outcomes), "spent_usd": budget.spent, "remaining_usd": budget.remaining(), "outcomes": outcomes}
    _write_json(results / "followup" / "api_followup.json", payload)
    return payload
