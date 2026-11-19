# Repairing a Swedish adjective network from its weights

## Question

The public note on weight-space programming reports that a frontier model can read a small network's weights and edit them, sometimes without feedback from the edited model. This experiment asks a narrower question. Given a network that has actually learned a multi-rule Swedish adjective paradigm, can GPT-6.2 Astra and GPT-6.2 Sol propose a small weight edit that fixes one grammatical failure and leaves the other cells alone?

Phase 1 does not answer that question. It builds the network, installs a selective failure, and checks that the failure is worth spending API budget on. No model was asked to edit weights. The ledger has no charges.

## Task and model

The classifier predicts an inflection exponent for an adjective in a noun phrase. The exponent is then mapped to a surface form by a fixed function. The rules are not one paradigm copied across the lexicon. Regular and *-ig* adjectives take *stor, stort, stora*. Geminating adjectives take *ny, nytt, nya*, *röd, rött, röda*, or *vit, vitt, vita*. Syncopating adjectives take *vacker, vackert, vackra* and *gammal, gammalt, gamla* (*gammal* simplifies *mm* in the weak form). Adjectives in *-en* take *mogen, moget, mogna*. *bra* never changes. *liten* splits into *liten, litet, lilla, små*. Gender changes the exponent only in the singular indefinite; plural and definite forms neutralize it, except that *liten* keeps number and definiteness apart.

There are 96 stems and 768 phrase types (8 grammatical cells each). Stems are split within each class into training, a labeled diagnostic slice, a development gate, and a held-out test. *liten* has a single stem, so it is scored as an in-lexicon irregular rather than as evidence of stem generalization. Sixteen small stem-specific bits keep a held-out stem from being an exact copy of a training input. The label does not depend on those bits, and training includes a copy of each row with the bits removed so the grammar flags have to carry the decision. Ninety surface forms were checked by hand. Nouns are only carriers (*bil*, *hus*): a second noun requiring the same exponent is not a new test.

The network is a ReLU MLP with widths 33, 48, 48, and 9 (4,425 parameters). Widths 32 and 40 did not learn the development stems; 48 did. Training used seed 42, Adam, and class-balanced loss on CPU for 180 epochs (0.4 seconds). Development accuracy, *liten*, and every development class reached 100%. After rounding to three decimals, which is the precision later shown to a model, held-out test accuracy was still 100% on all 136 stem–cell rows, including all 15 strong-neuter rows. Parsing the serialized weights reproduced the checkpoint exactly.

## The lesion

The search scored 85 candidates. Fourteen kept non-target development accuracy at or above 90% while actually damaging the strong-neuter cell. The seven protocol-1 lesions gated before the column copy — unit scaling, noise, and a stem-bit nudge — were all undone by refitting the last layer on the diagnostic rows: the target cell returned to 100%. The label was still linearly readable from the second hidden layer, so those lesions were rejected as readout problems. The log is `results/corruption_gate.json`.

The lesion that survived is a copy, in the first layer, of the uter gender column onto the neuter gender column. A neuter phrase then has the same hidden state as the uter phrase with the same stem and the same number and definiteness. Where the exponent already ignores gender, the prediction stays correct. Where gender matters — neuter singular indefinite, for every class except the invariants — the network predicts the uter base. On the test stems that is 0/15 correct in the target cell and 121/121 correct elsewhere. Overall accuracy is still 89%, which is why the cell score, not the average, is the one that matters. The target rate is 0% rather than the hoped-for 20–60% band because the copy removes the distinction outright. Partial copies were the kind of lesion the last layer could finish.

## Local repairs

Every repair saw the same 144 diagnostic rows (development-diagnostic stems plus *liten*) and was scored once on the untouched test stems.

| Method | Target accuracy | Non-target accuracy | Edited parameters | API cost |
|---|---:|---:|---:|---:|
| Original network | 1.00 | 1.00 | 0 | $0 |
| Corrupted network | 0.00 | 1.00 | 0 | $0 |
| Random hidden edits (mean of 8) | 0.00 | 1.00 | 40 | $0 |
| Output-bias repair | 0.20 | 0.98 | 9 | $0 |
| Last-layer repair | 0.53 | 0.93 | 378 of 441 | $0 |
| Gradient repair, 40 weights | 1.00 | 1.00 | 40 | $0 |

Changing only the nine output biases lifts the target cell to 20% and slightly damages other cells. The last-layer fit was allowed to move every output parameter: `out.weight` is shaped `[9, 48]`, so that layer contains 432 weights plus 9 biases, 441 parameters in all. The evaluation counts an entry only when its absolute change exceeds 1e-8. Under that rule 369 weights and all 9 biases moved, which is the reported 378. The other 63 weights are seven hidden-unit columns, indices 7, 12, 15, 16, 21, 25 and 36, left unchanged for every output class. No earlier parameter was eligible. That fit reaches 8/15 on the target cell and 113/121 elsewhere. The shared hidden state is why a readout cannot finish the job: the uter and neuter singular of one stem are now the same vector and cannot take two labels. Forty random edits of comparable count, with standard deviation 0.25 in the hidden matrices, move neither score. A gradient step restricted to the 40 largest diagnostic gradients restores both scores to 15/15 and 121/121. Its total change has L2 norm 6.08, just above the Frobenius cap of 6 that an API edit must obey, so that particular numerical solution would have been rejected as a model proposal. A direct correction of the copied column is smaller (L2 1.46, largest entry 0.55) but touches 48 parameters, eight more than the budget. The defect is selective, it is not a bias tweak, and it is repairable inside 40 parameters.

A permutation of both hidden layers, with incoming and outgoing weights moved together, changes logits by at most 2.4e-6 across all 768 rows. The permuted checkpoint is saved. It was not sent to an API.

## What a later API round can and cannot show

The one-shot prompt is 23,236 tokens and round-trips at the printed precision. Reserving a 20% token margin and 6,000 output tokens, which include reasoning, the worst-case Standard price is $0.12 for Sol and $0.46 for Astra. The pair is $0.58. Adding one feedback round and one permuted-network round for each model raises the reservation to $1.76. The cap is $5. Estimates use the short-context rates published for `gpt-6.2-sol` ($2 / $10 per million) and `gpt-6.2-astra` ($8 / $40), checked on 18 November 2026, and they do not assume a cache discount. `api_enabled` is false. Mocked tests reject an over-budget call, halt after an uncertain charge, and refuse a ledger whose cumulative total does not add up.

The one-shot calls were made after the checks below. Feedback and the permuted checkpoint stayed off the API. None of the outcomes show that the same edit would improve a pretrained Swedish model. The classifier is small, the paradigm is synthetic, and the lesion is one column copy chosen because weaker lesions were linearly undoable.

## Phase 2

### Local checks

The corrupted checkpoint, the 144 diagnostic rows, the 136 test rows, the prompt, and the edit schema were hashed before either call. Re-scoring the saved checkpoints reproduced the Phase 1 facts: original 136/136, corrupted 0/15 and 121/121, gradient repair 15/15 and 121/121. Parsing the three-decimal serialization changed no weight and reproduced those corrupted predictions. The hidden-unit permutation still moves logits by at most 2.4e-6. The corrupted file hash was the same after both calls.

A gradient-free column diagnostic compared every pair of columns in `fc1.weight`, `fc2.weight`, and `out.weight`, on both the original and the corrupted network. In the corrupted first layer, `gender_uter` (column 0) and `gender_neuter` (column 1) are an exact copy: cosine 1, distance 0, equal in all 48 rows. They are the only exact duplicate pair in any of the three matrices. On the original network those two columns have cosine 0.19 and distance 1.46, and there is no exact duplicate. The next-strongest first-layer pair, syncopating against `ends_el`, has cosine 0.73 in both networks. Output columns 30 and 34 have cosine 0.998 in both networks as well, with distance 0.055, so that near-match was not introduced by the lesion. The diagnostic detects the copied columns. It does not propose or score a repair, so it is not a repair baseline.

The prompt tells both models the architecture, feature names, exponent names, the corrupted weights, the labeled diagnostic rows, and the 40-parameter edit rules. It asks for a separate diagnosis and an edit. It does not say that a column was copied, and it does not include this diagnostic. The edit operations are the ones validated in Phase 1: `add`, `set`, and `rank1`, at most 40 scalars.

### The two calls

Both requests used the Responses API, reasoning effort medium, a 6,000-token output cap, no tools, and the same frozen prompt of 23,335 counted tokens. The API billed 23,671 input tokens. That is 336 above the local count; the 20% reservation still covered it. Charges match the Standard short-context rates checked on 18 November 2026. The ledger now has two completed rows. Cumulative spend is $0.377110, and $4.622890 remains. Rerunning the notebook loads the saved responses.

Sol (`gpt-6.2-sol`, response `resp_094d235e495f8de1016afe1f1e3aa087d2b15d7b5e3c6fb488`) returned status `incomplete` because the output cap was reached. All 6,000 output tokens were reasoning tokens, and the response text was empty. There is no diagnosis and no edit. The attempt is a failure. Target accuracy stays 0/15 and non-target accuracy stays 121/121. The call cost $0.107342. It was not retried, and the cap was not raised.

Astra (`gpt-6.2-astra`, response `resp_085490cd450b3ac6016afe1f79266887d2b0c9df5cf0885665`) returned a completed proposal. The diagnosis says `fc1.weight[:,1]` (`gender_neuter`) is identical to `fc1.weight[:,0]` (`gender_uter`) in all 48 rows, and that a neuter singular indefinite phrase therefore has the same first-layer preactivation as its uter counterpart. That is the lesion. The keyword heuristic initially missed it because the wording was "identical to" and "match in all 48 rows"; the judgement here is from reading the saved text.

The edit does not restore the copied column. It spends 11 scalars, L2 norm 5.91, building a gate on first-layer unit 18 and second-layer unit 7. Neuter singular indefinite is meant to turn the gate on, and plural, definite, or invariant inputs to turn it off. The gate then lowers the base logit and raises the three strong-neuter logits together, leaving the existing classifier to rank those three. On the test stems this repairs 11 of the 15 target rows and damages none of the 121 others. The four misses are all geminating neuters predicted as `NEUT_T` rather than `NEUT_TT`: *ett rött hus*, *ett glatt hus*, *ett slätt hus*, *ett flatt hus*. Forms such as *ett långt hus* and *ett vackert hus* go from the uter base to `NEUT_T`, and *ett galet hus* goes to `NEUT_ET`. Astra's own explanation notes that the gate assumes the right strong logit is already ahead of the other two. The diagnostic-set target accuracy is 12/16, with non-target accuracy 128/128. The call used 1,339 reasoning tokens inside 2,010 output tokens and cost $0.269768.

| Method | Diagnosis | Target accuracy | Non-target accuracy | Edited parameters | API cost |
|---|---|---:|---:|---:|---:|
| Corrupted baseline | — | 0/15 | 121/121 | 0 | $0 |
| Duplicate-column diagnostic | exact `fc1` gender-column copy; no repair proposed | — | — | 0 | $0 |
| Random edits | — | 0/15 | 121/121 | 40 | $0 |
| Output-bias repair | — | 3/15 | 118/121 | 9 | $0 |
| Last-layer repair | — | 8/15 | 113/121 | 378 of 441 | $0 |
| Gradient repair | — | 15/15 | 121/121 | 40 | $0 |
| Sol one-shot | no proposal; output cap spent on reasoning | 0/15 | 121/121 | 0 | $0.107342 |
| Astra one-shot | duplicated `fc1` gender columns | 11/15 | 121/121 | 11 | $0.269768 |

Astra is a selective partial repair: the target cell improves and non-target accuracy stays at 100%. Sol is not a useful repair. The figure is `results/figures/phase2_comparison.png`.

### What this does and does not show

Astra shows that one frontier model can name this particular copied column from the printed weights and can install a small gate that fixes most of the failing cell without touching the others. It does not show that the model recovered the original computation. The successful numerical repair uses 40 weights spread across the first layer and the readout, with L2 6.08. Astra's gate is a different mechanism, and it still assigns the geminating neuter the regular ending.

Sol provides no evidence either way. An empty completion is not a failed diagnosis of the columns. One trial, ended by the output cap, is not a comparison of the two models' reasoning.

The numerical diagnostic finds the same columns with no language model and no API cost. Finding them is not the same as repairing them. At the end of this one-shot comparison, the gradient edit was the only method that restored 15/15 and 121/121.

The lesion is an artificial duplicate, selected from 85 candidates partly because the last layer could not undo it. A conspicuous copied column is not a naturally occurring grammatical failure, and this result does not transfer to other lesions or to a pretrained language model.

### Next experiment

The more informative extension is the saved hidden-unit permutation, not the feedback round. Astra's diagnosis is about input columns, which a hidden-unit reordering does not move, but the repair addresses units 18 and 7 by number. Asking Astra once, on the functionally equivalent permuted checkpoint, tests whether the column diagnosis still leads to a useful edit when those numbers change. A feedback round would mainly ask whether the four remaining geminating forms can be patched, which is a narrower question. Sol cannot be revised from an empty completion without a larger output cap, and this round does not authorize one. The permuted checkpoint and the feedback code path are still unused.

## Follow-up

That permutation test and a larger Sol allowance were run afterward. The write-up is `findings.md`. Scaling the 40-parameter gradient update to L2 norm 6.00 still scores 15/15 and 121/121, so no new constrained optimization was needed. Astra's 11-parameter edit transfers exactly onto the permuted network. A new Astra call on those weights again names the copied columns and again repairs 11/15 with no held-out regressions, at $0.328360. Sol, given a 12,000-token output cap on the original prompt, completes a 28-parameter edit that scores 15/15 and 121/121. The billed cost is $0.081403 because the identical prompt was cached. Cumulative spend is $0.786873, with $4.213127 remaining. Further paid calls on this column-copy lesion are not justified.
