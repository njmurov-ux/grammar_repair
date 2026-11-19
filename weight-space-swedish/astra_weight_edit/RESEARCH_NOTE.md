# Can Frontier LLMs Repair a Learned Swedish Grammar Network by Editing Its Weights?

## Research question and motivation

*Sparks of Weight-Space Programming* (Takehi et al., 2026) proposes that general-purpose LLMs can interpret and modify neural-network weights rather than merely write training code. Its strongest evidence for direct editing is a one-shot intervention in a failed control policy: changing 3.5% of its parameters raised success from 0% to 100% across 1,024 episodes. The reported improvement to Qwen3.5-0.8B on MATH-500 (7.9% to 48.1%) is also striking, but came from an agent with a Python environment and up to 50 forward-pass evaluations. That result demonstrates effective *feedback-guided search*, not necessarily an ability to infer a repair from weights alone. The authors also report that Astra could not identify sycophancy from weights without additional behavioral information. These contrasts leave open how reliably direct weight inspection can produce a selective repair, especially for language-related computations.

We therefore ask: **Can a frontier LLM repair a known grammatical failure in a network that has already learned Swedish adjective inflection, without gradients or feedback on its proposed edit?** The hypothesis predicts higher accuracy on the damaged constructions **without regressions** on unaffected ones. Invalid edits, persistent target errors, or regressions would challenge it; comparable numerical repairs would limit any claim of practical advantage. Swedish morphology provides a controlled, objectively scored proxy for a capability SE.LLMA ultimately needs in much larger language models.

## Method

We trained a 4,425-parameter ReLU MLP (33 → 48 → 48 → 9) to predict adjective-inflection classes from explicit grammatical and lexical features. Its 96 stems cover regular, geminating, syncopating, invariant, and irregular forms. Before intervention, it classified all 768 generated stem–grammar combinations correctly. These are structured features, not raw Swedish text.

We deliberately copied the first-layer common-gender input column onto the neuter column, eliminating a distinction needed for singular indefinite forms. The lesion, selected from 85 candidate corruptions, produced 86 errors while retaining 682 correct predictions. Both GPT-6.2 Astra and GPT-6.2 Sol received the damaged weights, architecture, feature descriptions, output classes, and 144 labeled diagnostic cases—not the original weights, corruption mechanism, gradients, or held-out labels. Each proposal was limited to 40 scalar changes, with update L2 norm at most 6, and was evaluated by local code without post-edit feedback. A 136-item test split was kept separate from repair data; the additional 128-item development gate was not shown to the LLMs. Controls included unchanged and random edits, output-bias and last-layer repairs, constrained gradient repair, and optimization restricted to columns flagged by a simple duplicate detector.

## Results

| Intervention | Correct / 768 | Parameters changed |
|---|---:|---:|
| Damaged network | 682 | 0 |
| Output-bias / last-layer repair | 678 / 682 | 9 / 378 |
| Astra, original / permuted network | 746 / 746 | 11 / 23 |
| Constrained gradient repair | 757 | 40 |
| Duplicate-column optimizer | 767 | 40 |
| **Sol, extended output budget** | **768** | **28** |

Sol identified the duplicated gender columns and built an alternative, selective computation. Its 28-weight edit corrected all 86 failures without changing any previously correct prediction, scoring 136/136 on the reserved test split and 768/768 across the full defined task. This was one completed proposal with no gradient access or feedback on its edit. Sol initially exhausted a 6,000-token reasoning/output limit without returning a proposal; its successful attempt was allowed 12,000 tokens and used 7,903. It therefore cannot be ranked against Astra under an equal reasoning budget.

Astra independently generated two partial repairs. The original edit corrected 64/86 damaged cases; a second edit after a function-preserving hidden-neuron permutation achieved the same 746/768 predictions. Both preserved unaffected cases. Their 22 residual errors were all geminating neuter forms—for example, selecting the ordinary *-t* ending instead of *-tt* in *ett rött hus*. The permutation test supports robustness to this one reparameterization, not general invariance.

Eight random 40-weight edits did not repair any of the 15 held-out target cases. The numerical controls qualify the result: a deterministic test found the identical input columns, so locating the anomaly did not require an LLM. Constrained gradient editing reached 757/768, and projected Adam on the detected columns reached 767/768, but the latter introduced a regression on *de små bilarna*. The full 768-case audit uncovered errors invisible in the 136-case test, including failures in the gradient repair. The audit includes training and diagnostic examples and must not be interpreted as 768 independent held-out tests.

## Interpretation and recommendation

This study supports a narrow but substantive conclusion: **frontier LLMs can synthesize selective weight-level repairs of a small learned grammatical computation without gradients or feedback on their edits**. Sol recovered the full enumerated input–output behavior without restoring the original weights, and Astra found similarly useful interventions under two equivalent parameterizations. It does **not** establish that LLM editing generally outperforms gradient optimization, that Sol is intrinsically better than Astra, or that the approach improves a pretrained Swedish language model. The hand-designed features, conspicuous artificial lesion, single selected network, and unequal reasoning budgets constrain generalization.

It could be valuable to investigate the approach further, but as an experimental complement to conventional methods, not an assumed replacement. The next controlled study should use several less obvious, distributed defects, stronger numerical baselines, matched inference budgets, and independent regression tests. For relevance to much larger Swedish LLMs, a promising *zoom-out/zoom-in* workflow would first localize a failure using task examples and coarse layer- or module-level diagnostics, then inspect selected attention heads, MLPs, or low-rank directions and propose sparse edits; finally it would zoom out again to evaluate Swedish capabilities and regressions. Critically, it should compare one-shot, weight-informed interventions against feedback-guided editing and conventional fine-tuning or steering under comparable budgets. Whether such hierarchical inspection scales to billion-parameter networks remains an open question.

## References and AI usage

Takehi, R., Teoh, J., Zhang, A., & Khattab, O. (2026). [*Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights*](https://rikiyat.github.io/blog/weight-space-programming/).

ChatGPT (GPT-6) assisted with experimental planning and editorial revision; Grok 4.7 in Cursor assisted with implementation, analysis, and the initial report. GPT-6.2 Astra and Sol proposed the evaluated weight edits. Local code validated and applied their numerical proposals; no model-generated program was executed. Reproduction code, seeds, model identifiers, prompts, and per-example results accompany the submission.
