# Weight-space repair of a synthetic Swedish adjective network

Can a frontier language model propose a small numerical edit that repairs one learned grammatical computation, without gradients and without feedback on the edit?

The network is a ReLU MLP (33 → 48 → 48 → 9, 4,425 parameters) trained on a synthetic Swedish adjective paradigm. It is not a Swedish language model. The submission note is `RESEARCH_NOTE.md`. Tables, the lesion, the prompt configuration, and the API ledger are in `SUPPLEMENTARY_FINDINGS.md`. The AI disclosure is also in `AI_USAGE.md`. The pre-submission laboratory write-up is kept as `LABORATORY_NOTE.md`.

## Result in one table

Correct predictions on all 768 enumerated phrase types:

| Method | Correct / 768 | Parameters modified |
|---|---:|---:|
| Original network | 768 | 0 |
| Corrupted network | 682 | 0 |
| Constrained gradient repair | 757 | 40 |
| Duplicate-column optimizer | 767 | 40 |
| Astra, either representation | 746 | 11 or 23 |
| Sol, extended output budget | 768 | 28 |

Sol's completed edit matches the original network on every enumerated phrase. Astra's two repairs miss the same 22 geminating neuter forms and no others. The admissible gradient baseline is the update scaled to L2 norm 6 (757/768). Details and the rows that each method still misses are in the supplement.

## Reproduce the scored results

From `astra_weight_edit`, with the project environment:

```text
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Open `experiment.ipynb` and run it from this directory. Phase 1 loads the saved network. Phase 2 and the follow-up load the saved API responses. A rerun does not call the API and does not retrain if `results/checkpoints/corrupted.pt` is present.

The API key, if present, is read from a repository-root `.env` (`OPENAI_API_KEY` or `OPEN_API_KEY`). The loader does not print it. `src/config.py` keeps `api_enabled` false. Do not enable it. The four authorized calls have already been made, and the cumulative spend is $0.786873 of the $5 cap (`results/api_usage.csv`).

Checkpoints, prompts, and response JSON under `results/` are the experimental record. Do not overwrite them.

## Layout

```text
astra_weight_edit/
├── RESEARCH_NOTE.md
├── RESEARCH_NOTE.pdf
├── SUPPLEMENTARY_FINDINGS.md
├── AI_USAGE.md
├── LABORATORY_NOTE.md
├── findings.md
├── experiment.ipynb
├── src/
└── results/
```
