"""Fixed experiment settings.

Protocol constants live here so corruption search, edit budgets and API limits
are explicit. If a setting changes after a pilot run, bump PROTOCOL_VERSION and
append a dated entry to AMENDMENTS rather than quietly replacing the record.
"""

from __future__ import annotations

SEED = 42

# Input layout. Stem codes are idiosyncratic distractors; the label never depends on them.
N_STEM_BITS = 16

HIDDEN_CANDIDATES = (32, 40, 48, 64)
N_HIDDEN_LAYERS = 2
MAX_PARAMETERS = 10_000

TRAINING = {
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "max_epochs": 500,
    "patience": 80,
    "min_dev_example_acc": 0.95,
    "min_dev_class_acc": 0.95,
    "min_liten_acc": 1.0,
}

# Defect after protocol amendment 2: every non-invariant neuter singular
# indefinite form collapses onto the uter base. That is the cell in which
# gender actually changes the Swedish exponent (NEUT_T, NEUT_TT or NEUT_ET).
TARGET_GENDER = "neuter"
TARGET_NUMBER = "sg"
TARGET_DEFINITENESS = "indef"

# Stem allocation, summed against the lexicon in grammar.py.
# dev_diag is the only labeled repair set given to baselines and, later, the LLMs.
# dev_gate is held out from repair fitting and used to accept a corruption.
# test is not used for architecture, corruption or baseline selection.
SPLIT_SIZES = {
    "regular": {"train": 12, "dev_diag": 4, "dev_gate": 4, "test": 4},
    "ends_ig": {"train": 7, "dev_diag": 3, "dev_gate": 2, "test": 3},
    "geminating": {"train": 10, "dev_diag": 4, "dev_gate": 4, "test": 4},
    "syncopating": {"train": 6, "dev_diag": 2, "dev_gate": 2, "test": 2},
    "ends_en": {"train": 6, "dev_diag": 2, "dev_gate": 2, "test": 2},
    "invariant": {"train": 4, "dev_diag": 2, "dev_gate": 2, "test": 2},
    "liten": {"inlexicon": 1},
}

# Pre-registered corruption search. Scored on dev only.
PROTOCOL_VERSION = 2
AMENDMENTS: list[dict] = [
    {
        "version": 2,
        "reason": (
            "Every selective protocol-1 lesion that preserved non-target behavior "
            "was undone by refitting the full last layer on the diagnostic set. "
            "The grammatical label was still linearly decodable from the second hidden "
            "layer, so those lesions were readout problems. Protocol 2 adds a "
            "gender-column copy in fc1: the input weight for gender_neuter is replaced "
            "by the weight for gender_uter. A neuter phrase then has the same hidden "
            "state as its uter counterpart. Plural and definite exponents already ignore "
            "gender, so those cells stay correct, while singular indefinite neuter forms "
            "are predicted as the uter base. A later linear layer cannot assign two labels "
            "to one hidden state. The target set is this strong-neuter cell across lexical "
            "classes, not the geminating subclass alone."
        ),
    }
]
CORRUPTION = {
    "version": PROTOCOL_VERSION,
    "activation_split": "train",
    "score_split": ("dev_diag", "dev_gate", "inlexicon"),
    "prefer_target_accuracy": (0.20, 0.60),
    # A single grammatical cell is deterministic given shared class features, so
    # accuracy on that cell often jumps from 1 to 0. An atomic miss is acceptable
    # when the rest of the network is intact and a budgeted gradient edit can
    # restore the cell. Cherry-picking on test accuracy is not allowed.
    "allow_atomic_target_failure": True,
    "nontarget_micro_min": 0.90,
    "nontarget_macro_min": 0.90,
    "liten_min": 0.75,
    "max_abs_weight_delta": 5.0,
    "column_scale": {
        "m": (1, 2, 4, 8),
        "factors": (0.0, 0.5, -1.0),
        "scores": ("contrast_penalized", "target_vs_uter"),
    },
    "column_noise": {
        "m": (2, 4),
        "stds": (0.5, 1.0, 2.0),
        "seeds": (0, 1, 2),
    },
    "direction_dampen": {
        "alphas": (0.5, 1.0, 1.5, 2.0),
        "kinds": ("target_minus_uter", "target_minus_uter_orth_gender"),
    },
    "stem_threshold": {
        "scales": (0.75, 1.5, 3.0, 6.0),
        "seeds": (0, 1, 2, 3),
    },
}

# Repair budgets. Last-layer and bias baselines are intentionally NOT limited to
# this budget: they are the trivial methods the defect should resist.
# Gradient repair and the LLM interface share MAX_EDIT_PARAMS.
MAX_EDIT_PARAMS = 40
BASELINES = {
    "bias_steps": 250,
    "bias_lr": 0.05,
    "last_layer_steps": 250,
    "last_layer_lr": 0.03,
    "gradient_steps": 250,
    "gradient_lr": 0.05,
    "random_seeds": 8,
    "random_std": 0.25,
    "random_tensors": ("fc1.weight", "fc2.weight"),
    # "Solves" means the trivial edit removes the defect on dev_gate.
    # With four gate stems, 0.90 requires every stem back; 0.50 is two stems.
    "solved_target_min": 0.90,
    "solved_nontarget_micro_min": 0.90,
    "gradient_min_gain": 0.50,
    "gradient_nontarget_min": 0.85,
    # Protocol 1 left many selective-but-easy lesions. Walk the whole filtered
    # list so the gender-column lesion is judged by the same gate.
    "max_gate_evaluations": 80,
}

QUANTIZE_DECIMALS_TRY = (3, 4)

# Official Standard short-context rates, per 1M tokens.
# Source: https://developers.openai.com/api/docs/pricing (checked 2026-11-18).
# Above 272K input tokens the published long-context multiplier applies.
API = {
    "api_enabled": False,
    "max_budget_usd": 5.00,
    "models": ["gpt-6.2-sol", "gpt-6.2-astra"],
    "reasoning_effort": "medium",
    "max_output_tokens": 6000,
    "run_one_shot": True,
    "run_feedback": False,
    "run_permutation": False,
    "seed": SEED,
    "input_token_safety_factor": 1.20,
    "tokenizer_encoding": "o200k_base",
    "long_context_threshold": 272_000,
    "timeout_seconds": 600,
    "prices_usd_per_million": {
        "gpt-6.2-sol": {"input": 2.0, "cached_input": 0.10, "output": 10.0},
        "gpt-6.2-astra": {"input": 8.0, "cached_input": 0.40, "output": 40.0},
    },
    "price_source": "https://developers.openai.com/api/docs/pricing",
    "price_checked": "2026-11-18",
}

TENSOR_NAMES = (
    "fc1.weight",
    "fc1.bias",
    "fc2.weight",
    "fc2.bias",
    "out.weight",
    "out.bias",
)

EDIT_LIMITS = {
    "max_edit_params": MAX_EDIT_PARAMS,
    "max_abs_delta": 3.0,
    "max_abs_value": 8.0,
    "max_frobenius": 6.0,
    "max_rank1_affected_entries": 800,
    "max_edit_ops": 8,
}
