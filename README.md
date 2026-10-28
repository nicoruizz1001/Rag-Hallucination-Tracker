# RAG Hallucination and Reliability Benchmark

A Retrieval-Augmented Generation (RAG) question-answering system used to measure how
retrieval method (BM25 vs. FAISS dense retrieval), number of retrieved passages (K),
and prompt instructions affect answer correctness and unsupported factual claims.

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Local LLM via [Ollama](https://ollama.com/):

```powershell
ollama pull llama3.2:3b
```

Run the tests:

```powershell
python -m pytest
```

## Layout

| Path | Purpose |
|------|---------|
| `src/` | Pipeline code: data prep, retrieval, generation, retrieval evaluation, claim evaluation, experiments, plots |
| `configs/` | Experiment configurations |
| `data/` | Downloaded datasets and indexes (git-ignored) |
| `results/` | Saved experiment outputs, tables, figures |
| `notebooks/` | Exploratory analysis |
| `report/` | Research write-up |
| [`report/annotation_guidelines.md`](report/annotation_guidelines.md) | Claim-support labels, annotation rules, synthetic examples, and human validation protocol |
| `tests/` | pytest suite |

## Claim-support evaluation

`src/claim_evaluation.py` evaluates saved generation JSONL in two stages: it extracts factual claims from each saved question and answer, then checks each claim against all passages recorded as supplied to the generator. It does not regenerate answers, use reference answers or SQuAD labels, or change retrieval evaluation. Question correctness, citation support, and evidence-conditioned answerability remain separately marked unassessed/unverified. The default is a one-record smoke run; choose a larger `--limit` explicitly.

Run against an existing generation file (PowerShell examples):

```powershell
python src/claim_evaluation.py --input results/generations/generations_20261026T204950624686Z.jsonl
python src/claim_evaluation.py --input results/generations/generations_20261026T204950624686Z.jsonl --limit 8
```

Run the synthetic evidence demo, which prints extracted claims and judgments:

```powershell
python src/claim_evaluation.py --synthetic-demo
```

Run three live synthetic support checks (supported, contradicted, and insufficient evidence), saving the input and outputs in a fresh run directory:

```powershell
python src/claim_evaluation.py --synthetic-examples
```

### Evaluator diagnostics and model comparison

The diagnostic set is synthetic: inputs are in `data/evaluator_diagnostics/cases.json`, and expected labels are kept separately in `expected_labels.json`. Expected labels are loaded only after all model calls. Both files are frozen: their content hashes are pinned in `src/evaluator_diagnostics.py`, and runs refuse an edited set unless `--allow-unfrozen` is given. Each case is judged twice, once with its relevant passage alone and once with unrelated distractors added. Every Ollama request is recorded and checked (prompt, schema, passages, model, options, token counts). This is a **development diagnostic** for finding failure modes, **not** an unbiased accuracy benchmark or independent validation.

Extraction and judgment are configured independently. `--model` sets both stages. `--extraction-model` / `--judgment-model` override one stage, and `--extraction-options` / `--judgment-options` take JSON Ollama option overrides. Leave the options unset when comparing models, so that every run uses the same prompts, schemas, evidence ordering and settings (`num_ctx` 8192, temperature 0, seed 42). Models must already be installed (`ollama list`). Nothing is downloaded.

```powershell
# Baseline: one model for both stages
python src/evaluator_diagnostics.py --model llama3.2:3b

# Candidate judge with extraction held fixed: reuse the baseline's saved claims, so only the judge varies
python src/evaluator_diagnostics.py --judgment-model <installed-candidate> --reuse-extraction results/evaluator_diagnostics/<baseline_run_dir>

# Candidate for both stages (extraction quality is reported separately; a better judge is not assumed to extract better)
python src/evaluator_diagnostics.py --model <installed-candidate>

# Comparison table across saved runs (no model calls); flags runs whose prompts/schemas/options/case set differ
python src/evaluator_diagnostics.py --compare results/evaluator_diagnostics/<run_a> results/evaluator_diagnostics/<run_b>

# Re-score a saved run without model calls
python src/evaluator_diagnostics.py --rescore results/evaluator_diagnostics/<run_dir>
```

#### Result: candidate judge `qwen2.5:7b-instruct` vs `llama3.2:3b` (2026-10-28)

Development diagnostic only, **not independent validation**: 14 synthetic cases written while investigating known failures; 36 labelled judgments. Only the judge changed. Both runs used the same saved `llama3.2:3b` claims (`--reuse-extraction`), prompts `claim_extraction_v5` / `claim_judgment_v9`, the frozen case and label files, the same passage ordering, and the same settings (temperature 0, seed 42, `num_ctx` 8192, `num_predict` 1536). The `--compare` compatibility check passed.

| Metric | llama3.2:3b | qwen2.5:7b-instruct |
|---|---|---|
| Correct / labelled judgments (failures count as wrong) | 22/36 | 34/36 |
| SUPPORTED / CONTRADICTED / NOT_ENOUGH_EVIDENCE | 18/26 · 0/6 · 4/4 | 26/26 · 5/6 · 3/4 |
| Correct: relevant passage only / with distractors | 14/18 · 8/18 | 18/18 · 16/18 |
| Label changes after adding distractors | 6/18 | 2/18 (both are failed judgments) |
| Valid judgments citing a distractor | 16/18 | 0/18 |
| Judgment failures / retries / calls | 1 · 8 · 44 | 2 · 3 · 39 |
| Judgment model time / run wall-clock | 89 s · 112 s | 264 s · 268 s |

Both candidate misses are **formatting failures, not wrong labels**. With a near-miss distractor present, the model wrote a runaway explanation until it hit `num_predict` (`done_reason=length`, unterminated JSON), on the first attempt and on the retry. Each of its 34 valid judgments had the expected label and cited the relevant passage. Remaining quality issues: some evidence excerpts quote a pronoun sentence without the antecedent sentence, and explanations sometimes overstate wording ("explicitly states that Rafael Ortiz…" when the passage says "Ortiz"). These results do not by themselves justify switching judges; the real-data sample and human validation still apply.

Outputs:
- Baseline run: `results/evaluator_diagnostics/evaluator_diag_20261028T003414829923Z_66fdf154564c`
- Candidate run: `results/evaluator_diagnostics/evaluator_diag_20261028T005839983616Z_7d12cdf673ea`
- Comparison table: `results/evaluator_diagnostics/comparison_20261028T010314208995Z.md`

Eight-record real-data check: the first 8 records of the saved 400-record generation file (2 questions × 4 configurations). It writes a new run directory and leaves the generation records unchanged. Do not raise `--limit` to run all 400 records without deciding to.

```powershell
python src/claim_evaluation.py --input results/generations/generations_20261026T205113112864Z.jsonl --limit 8 --model llama3.2:3b
python src/claim_evaluation.py --input results/generations/generations_20261026T205113112864Z.jsonl --limit 8 --extraction-model llama3.2:3b --judgment-model <installed-candidate>
```

Each invocation creates a unique run directory under `results/claim_evaluation/`, with `answers.jsonl`, `claims.jsonl`, and `manifest.json`. The manifest and stage records include evaluator model identity/settings, prompts, raw model responses, latency, counts, and failures. The evaluator defaults to local Ollama model `llama3.2:3b`.

Using the same model as generator and judge can produce correlated errors. Human validation, following [the annotation guidelines](report/annotation_guidelines.md), is required before reporting findings. The evaluator's schema and provenance checks validate output structure and excerpt references; they do not establish semantic correctness.

## Environment

- Python 3.11
- LLM: `llama3.2:3b` via Ollama (record any change — it invalidates direct comparisons)
- Exact package versions are pinned in `requirements.txt`.

On a fixed sample of 100 SQuAD 2.0 validation questions and a corpus of 500 unique passages, BM25 achieved Source Hit@1 of 81%, Source Hit@5 of 94%, and MRR@5 of 0.865. For the 48 answerable questions, Source Hit@5 was 95.8%. These metrics measure retrieval of the assigned source passage, not answer correctness or corpus-wide answerability.

BM25 achieved higher Source Hit@1 (81% versus 78%) and MRR@5 (0.865 versus 0.854), while dense retrieval achieved slightly higher Source Hit@5 (95% versus 94%). Both methods retrieved the assigned source within the top five for 46 of 48 answerable questions. Results apply to our fixed 100-question sample and 500-passage corpus.
