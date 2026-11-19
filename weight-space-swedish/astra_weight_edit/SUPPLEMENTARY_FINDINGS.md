# Supplementary findings

This supplement records the verified figures behind `RESEARCH_NOTE.md`. It does not add experiments. The earlier laboratory narrative is kept in `LABORATORY_NOTE.md`. Row-level predictions are in `results/followup/full_census.csv`, and every incorrect row is in `results/followup/full_census_errors.csv`.

## Task and network

The classifier predicts one of nine exponents: `BASE`, `NEUT_T`, `NEUT_TT`, `NEUT_ET`, `WEAK_A`, `WEAK_DROP_E`, `WEAK_NA`, `LILLA`, `SMA`. A fixed rule maps exponent and stem to a surface form. Nouns (`bil`, `hus`) are display carriers and are not features.

Architecture: `Linear(33, 48)`, ReLU, `Linear(48, 48)`, ReLU, `Linear(48, 9)`. Parameter count: 4,425. Weights shown to the models, and the checkpoints that were scored, are quantized to three decimals. Seed 42. The intact network scores 768/768 on the enumerated inventory.

Splits, assigned within each lexical class: train, dev_diag, dev_gate, test, and inlexicon (*liten* only). The labeled diagnostic packet is dev_diag plus inlexicon (144 rows: 16 target, 128 non-target). The reserved test split has 136 rows (15 target, 121 non-target). The development gate has 128 rows and was used neither as diagnostic material nor to choose edited coordinates.

## Lesion

Eighty-five candidate corruptions were scored. Lesions that a last-layer refit on the diagnostic rows could undo were rejected. The frozen lesion copies `fc1.weight[:, gender_uter]` onto `fc1.weight[:, gender_neuter]` and re-quantizes. After the copy the two columns are identical. It produces 86 errors, one per non-invariant stem, all in neuter singular indefinite, all predicted `BASE`. The other 682 predictions are unchanged. Invariant adjectives stay 80/80.

## What the models received

Both calls used the OpenAI Responses API, reasoning effort medium, `store=false`, and a strict JSON schema (`diagnosis`, `hypothesis`, `edits`, `explanation`). Operations were `add`, `set`, and `rank1`. Caps: 40 scalars, L2 norm at most 6, absolute change at most 3, stored absolute value at most 8, at most 8 operations. Invalid JSON was a failed attempt and was not repaired by hand.

The prompt contained the architecture, feature names, class definitions, the corrupted tensors, and the labeled diagnostic rows. It did not contain the uncorrupted weights, the phrase "copied onto", the lesion family name, gradients, baseline solutions, or test labels. The permuted Astra call used the same template with the permuted tensors substituted. It did not include the permutation or the earlier Astra edit. The extended Sol call reused the original frozen prompt; the only change was `max_output_tokens=12000`.

## Principal inventory

Counts below are correct predictions out of 768, from `results/followup/full_census.csv`.

| Method | Correct / 768 | Parameters | L2 norm | Role |
|---|---:|---:|---:|---|
| Original network | 768 | 0 | 0 | Intact reference |
| Corrupted network | 682 | 0 | 0 | Lesion |
| Output-bias repair | 678 | 9 | 2.074 | Readout bias fit on the diagnostic rows |
| Last-layer repair | 682 | 378 of 441 | 12.503 | Full last-layer refit; the L2 cap does not apply |
| Unconstrained gradient | 759 | 40 | 6.081 | Reference only; exceeds the edit cap |
| Constrained gradient | 757 | 40 | 6.000 | Unconstrained update scaled by 0.98665 |
| Duplicate-column optimizer | 767 | 40 | 4.053 | Projected Adam on the detected duplicate columns |
| Astra, original | 746 | 11 | 5.910 | One-shot, 6,000-token cap |
| Astra edit, transferred | 746 | 11 | 5.910 | Index rewrite of the original edit; no API call |
| Astra, permuted | 746 | 23 | 5.940 | New one-shot on the permuted checkpoint |
| Sol, first attempt | 682 | 0 | 0 | No proposal returned |
| Sol, extended budget | 768 | 28 | 5.976 | One-shot, 12,000-token cap |

The unconstrained gradient update was scaled onto the L2 ball. No new optimization was run, because the scaled update already satisfied every API constraint. Its score, 757/768, is the gradient number used in the research note.

The column optimizer is not gradient-free. A deterministic check finds the identical input columns. Projected Adam then runs for 250 steps at learning rate 0.05, the same schedule as the gradient baseline, on the 40 largest initial diagnostic gradients inside those columns. Test labels were not used to select the step. The last step had the best diagnostic loss.

## Accuracy by class

| Class | Rows | Corrupted | Bias | Last layer | Scaled gradient | Column optimizer | Astra | Sol extended |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| regular | 192 | 168 | 168 | 168 | 192 | 192 | 192 | 192 |
| ends_ig | 120 | 105 | 105 | 105 | 120 | 120 | 120 | 120 |
| geminating | 176 | 154 | 150 | 154 | 169 | 176 | 154 | 176 |
| syncopating | 96 | 84 | 84 | 84 | 96 | 96 | 96 | 96 |
| ends_en | 96 | 84 | 84 | 84 | 92 | 96 | 96 | 96 |
| invariant | 80 | 80 | 80 | 80 | 80 | 80 | 80 | 80 |
| liten | 8 | 7 | 7 | 7 | 8 | 7 | 8 | 8 |

Astra's three checkpoints (original edit, algebraic transfer, and the new permuted edit) agree on all 768 predictions. The 22 errors are every geminating stem in neuter singular indefinite, predicted `NEUT_T` instead of `NEUT_TT`: 10 in training, 4 in dev_diag, 4 in dev_gate, and 4 in test. Astra repairs 64 of the 86 damaged rows and regresses on none.

## Accuracy by split

| Split | Rows | Corrupted | Scaled gradient | Column optimizer | Astra | Sol extended |
|---|---:|---:|---:|---:|---:|---:|
| train | 360 | 319 | 353 | 360 | 350 | 360 |
| dev_diag | 136 | 121 | 134 | 136 | 132 | 136 |
| dev_gate | 128 | 114 | 126 | 128 | 124 | 128 |
| test | 136 | 121 | 136 | 136 | 132 | 136 |
| inlexicon | 8 | 7 | 8 | 7 | 8 | 8 |

Sol matches the original network on every row, including test 136/136 and dev_gate 128/128. The scaled gradient is perfect on the test split and misses 11 rows elsewhere, all still inside the original 86 damaged cells: five geminating and four *-en* neuters shared with the unscaled repair, plus *ett fritt hus* (dev_gate) and *ett vitt hus* (train, stem *vit*) introduced by the scaling. One of the shared misses, *ett rått hus*, is predicted `WEAK_A`. The column optimizer's only error is *de små bilarna* (`SMA` predicted `WEAK_NA`) on the in-lexicon row. The corrupted network had that row correct. Output bias scores 678/768, with 10 regressions on previously correct uter forms. The last-layer refit repairs 40 damaged cells and breaks 40 previously correct ones, ending at 682/768.

## Permutation

Seed 49. Hidden-unit order is permuted in both layers, with incoming and outgoing weights permuted together. Input features and output classes stay put, so the duplicated gender columns remain columns 0 and 1. Maximum absolute logit difference versus the corrupted network, over all 768 rows: 2.384e-6. Predictions match. Regenerating the permutation matches the saved checkpoint exactly. Transferring Astra's 11 indices through the inverse permutation reproduces the repaired function with logit gap 0.

## API accounting

Prices rechecked on 18 November 2026, Standard short context: Sol $2 / $10 per million input/output tokens, Astra $8 / $40, cached input $0.10 and $0.40. Both prompts are about 23.7k tokens, below the long-context threshold. Cumulative spend is $0.786873 of a $5 cap. Remaining budget: $4.213127. Ledger: `results/api_usage.csv`.

| Call | Output cap | Input tokens | Cached input | Output tokens | Reasoning tokens | Billed cost |
|---|---:|---:|---:|---:|---:|---:|
| Sol, first | 6,000 | 23,671 | 0 | 6,000 | 6,000 | $0.107342 |
| Astra, original | 6,000 | 23,671 | 0 | 2,010 | 1,339 | $0.269768 |
| Astra, permuted | 6,000 | 23,675 | 0 | 3,474 | 2,704 | $0.328360 |
| Sol, extended | 12,000 | 23,671 | 23,668 | 7,903 | 7,017 | $0.081403 |

The extended Sol prompt is identical to the first, so 23,668 input tokens were cache hits. A cold request at the same token counts would have cost about $0.126. Response ids: Sol first `resp_094d235e495f8de1016afe1f1e3aa087d2b15d7b5e3c6fb488` (incomplete); Astra original `resp_085490cd450b3ac6016afe1f79266887d2b0c9df5cf0885665`; Astra permuted `resp_0d2949709feae29a016afe239a640087d28d211540c8f2ea82`; Sol extended `resp_0a57bb104088da1b016afe2405051087d28fe1f0c7f415440d`.

## Saved proposals

Astra's original edit is a gate on first-layer unit 18 and second-layer unit 7. It separates neuter singular indefinite from the base reading and then assigns the same boost to `NEUT_T`, `NEUT_TT`, and `NEUT_ET`, which is why every geminating neuter remains `NEUT_T`. The permuted call built a different gate, on the renumbered first-layer unit that corresponds to original unit 18, and on a different second-layer unit. The function matches anyway.

Sol's edit uses 28 scalars across the first layer, three second-layer units, and the readout. It separates `NEUT_T`, `NEUT_TT`, and `NEUT_ET`. The resulting network matches the original predictions on all 768 rows. The weights are not the original weights.
