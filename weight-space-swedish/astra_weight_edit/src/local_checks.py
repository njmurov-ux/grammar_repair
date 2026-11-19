"""Phase 2 checks that do not call the API.

They recount the last layer, run the duplicate-column diagnostic, re-score the
frozen checkpoints, and hash the prompt that the two models will both receive.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from . import config
from .api_client import EDIT_SCHEMA, build_prompt, parse_serialized_state, serialize_state
from .column_diagnostic import diagnose_network
from .evaluation import permutation_max_abs_diff, state_delta, summarize
from .grammar import build_dataset, matrix_from_frame
from .model import GrammarMLP, load_state


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(payload: str) -> str:
    return sha256_bytes(payload.encode("utf-8"))


def _load_checkpoint(path: Path) -> tuple[GrammarMLP, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    architecture = checkpoint["architecture"]
    model = GrammarMLP(architecture["in_dim"], architecture["hidden"], architecture["out_dim"])
    load_state(model, checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def _counts(summary: dict) -> str:
    return f"{int(round(summary['target_acc'] * summary['target_n']))}/{summary['target_n']} and {int(round(summary['nontarget_acc'] * summary['nontarget_n']))}/{summary['nontarget_n']}"


def _row_digest(frame) -> str:
    columns = ["split", "stem", "lexical_class", "gender", "number", "definiteness", "exponent", "phrase", "label"]
    payload = frame[columns].sort_values(columns).to_csv(index=False)
    return sha256_text(payload)


def last_layer_audit(corrupted: GrammarMLP, repaired: GrammarMLP) -> dict:
    weight = corrupted.out.weight.detach()
    bias = corrupted.out.bias.detach()
    delta_weight = (repaired.out.weight.detach() - weight).abs()
    delta_bias = (repaired.out.bias.detach() - bias).abs()
    weight_changed = delta_weight > 1e-8
    bias_changed = delta_bias > 1e-8
    other = state_delta(
        {key: value.detach() for key, value in corrupted.state_dict().items()},
        {key: value.detach() for key, value in repaired.state_dict().items()},
    )
    return {
        "out_weight_shape": list(weight.shape),
        "out_bias_shape": list(bias.shape),
        "total_parameters": int(weight.numel() + bias.numel()),
        "eligible_parameters": int(weight.numel() + bias.numel()),
        "weight_parameters": int(weight.numel()),
        "bias_parameters": int(bias.numel()),
        "weight_parameters_changed": int(weight_changed.sum()),
        "bias_parameters_changed": int(bias_changed.sum()),
        "parameters_changed": int(weight_changed.sum() + bias_changed.sum()),
        "parameters_unchanged": int((~weight_changed).sum() + (~bias_changed).sum()),
        "unchanged_input_columns": [int(index) for index in range(weight.shape[1]) if bool((~weight_changed[:, index]).all())],
        "unchanged_by_output_row": [int((~weight_changed[row]).sum()) for row in range(weight.shape[0])],
        "evaluation_count_matches_changed": other["n_changed"] == int(weight_changed.sum() + bias_changed.sum()),
        "evaluation_n_changed": other["n_changed"],
        "hidden_parameters_changed": {
            key: value for key, value in other["per_tensor"].items() if not key.startswith("out.")
        },
        "explanation": (
            "out.weight is stored as [9, 48], so the layer has 432 weights and 9 biases, 441 parameters. "
            "All 441 were eligible for the last-layer Adam fit. state_delta counts an entry as edited only "
            "when its absolute change exceeds 1e-8. 369 weights and all 9 biases moved, which is the reported 378. "
            "The other 63 entries are seven input columns, indices 7, 12, 15, 16, 21, 25 and 36, left unchanged "
            "for every output class. No hidden-layer parameter was eligible or changed."
        ),
    }


def run_local_validation(experiment_dir: Path) -> dict:
    results = experiment_dir / "results"
    checkpoints = results / "checkpoints"
    corrupted_path = checkpoints / "corrupted.pt"
    before_hash = sha256_bytes(corrupted_path.read_bytes())
    original, _ = _load_checkpoint(checkpoints / "original.pt")
    corrupted, corrupted_ckpt = _load_checkpoint(corrupted_path)
    gradient, _ = _load_checkpoint(checkpoints / "baseline_gradient.pt")
    last_layer, _ = _load_checkpoint(checkpoints / "baseline_last_layer.pt")
    permuted, _ = _load_checkpoint(checkpoints / "corrupted_permuted.pt")

    frame = build_dataset(config.SEED)
    test = frame[frame["split"] == "test"].reset_index(drop=True)
    diagnostic = frame[frame["split"].isin(["dev_diag", "inlexicon"])].reset_index(drop=True)
    scores = {
        "original": summarize(test, original, "test"),
        "corrupted": summarize(test, corrupted, "test"),
        "gradient": summarize(test, gradient, "test"),
        "last_layer": summarize(test, last_layer, "test"),
    }
    expected = {
        "original": (15, 121, 136),
        "corrupted": (0, 121, 121),
        "gradient": (15, 121, 136),
    }
    mismatches = []
    for name, (target, nontarget, overall) in expected.items():
        summary = scores[name]
        got = (
            int(round(summary["target_acc"] * summary["target_n"])),
            int(round(summary["nontarget_acc"] * summary["nontarget_n"])),
            int(round(summary["example_acc"] * summary["n"])),
        )
        if got != (target, nontarget, overall):
            mismatches.append({"method": name, "got": got, "expected": [target, nontarget, overall]})

    state = {key: value.detach().cpu().clone() for key, value in corrupted.state_dict().items()}
    decimals = int(json.loads((results / "phase1_summary.json").read_text(encoding="utf-8"))["quantize_decimals"])
    serialized = serialize_state(state, decimals)
    parsed = parse_serialized_state(serialized, decimals)
    roundtrip = max(float((parsed[key] - state[key]).abs().max()) for key in state)
    probe = GrammarMLP(corrupted.fc1.in_features, corrupted.fc1.out_features, corrupted.out.out_features)
    load_state(probe, parsed)
    parsed_score = summarize(test, probe, "test")
    parsed_matches = (
        parsed_score["target_acc"] == scores["corrupted"]["target_acc"]
        and parsed_score["nontarget_acc"] == scores["corrupted"]["nontarget_acc"]
    )

    features, _ = matrix_from_frame(frame)
    permutation_gap = permutation_max_abs_diff(corrupted, permuted, features)
    column_report = {
        "original": diagnose_network(original.state_dict()),
        "corrupted": diagnose_network(state),
    }
    architecture = corrupted_ckpt["architecture"]
    diagnostic_view = summarize(diagnostic, corrupted, "diagnostic")
    prompt = build_prompt(state, architecture, diagnostic_view["scored"], decimals)
    prompt_path = results / "prompt_one_shot_phase2.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    leaked = [phrase for phrase in ("gender_column_copy", "copied onto", "duplicate column") if phrase in prompt]

    gradient_delta = state_delta(state, {key: value.detach() for key, value in gradient.state_dict().items()})
    audit = last_layer_audit(corrupted, last_layer)
    manifest = {
        "corrupted_pt_sha256": before_hash,
        "original_pt_sha256": sha256_bytes((checkpoints / "original.pt").read_bytes()),
        "permuted_pt_sha256": sha256_bytes((checkpoints / "corrupted_permuted.pt").read_bytes()),
        "dataset_csv_sha256": sha256_bytes((results / "dataset.csv").read_bytes()),
        "diagnostic_rows_sha256": _row_digest(diagnostic),
        "test_rows_sha256": _row_digest(test),
        "prompt_sha256": sha256_text(prompt),
        "schema_sha256": sha256_text(json.dumps(EDIT_SCHEMA, sort_keys=True)),
        "edit_limits": dict(config.EDIT_LIMITS),
        "models": list(config.API["models"]),
        "reasoning_effort": config.API["reasoning_effort"],
        "max_output_tokens": config.API["max_output_tokens"],
        "seed": config.SEED,
        "quantize_decimals": decimals,
    }
    after_hash = sha256_bytes(corrupted_path.read_bytes())
    report = {
        "last_layer": audit,
        "accuracies": {name: _counts(summary) for name, summary in scores.items()},
        "accuracy_mismatches": mismatches,
        "serialization_max_abs_error": roundtrip,
        "parsed_predictions_match_corrupted": parsed_matches,
        "permutation_max_abs_logit_diff": permutation_gap,
        "column_diagnostic_detects_gender_copy": column_report["corrupted"]["gender_columns_exact_duplicate"],
        "column_diagnostic_false_exact_pairs": column_report["corrupted"]["other_exact_duplicate_pairs"],
        "original_gender_columns_exact": column_report["original"]["gender_columns_exact_duplicate"],
        "prompt_leak_phrases": leaked,
        "gradient_edit_l2": gradient_delta["l2"],
        "gradient_edit_exceeds_frobenius_cap": gradient_delta["l2"] > config.EDIT_LIMITS["max_frobenius"],
        "checkpoint_unchanged": before_hash == after_hash,
        "freeze": manifest,
    }
    (results / "column_diagnostic.json").write_text(json.dumps(column_report, indent=2), encoding="utf-8")
    (results / "local_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (results / "freeze_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if mismatches or roundtrip != 0.0 or not parsed_matches or leaked or before_hash != after_hash:
        raise RuntimeError(f"local validation failed: {report}")
    if permutation_gap > 1e-4:
        raise RuntimeError(f"permuted checkpoint is not functionally equivalent: {permutation_gap}")
    return report
