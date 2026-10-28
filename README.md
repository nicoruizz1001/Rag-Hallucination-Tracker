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
| Valid with-distractor judgments citing a distractor (failed = unassessed) | 16/17 (1 unassessed) | 0/16 (2 unassessed) |
| Judgment failures / retries / calls | 1 · 8 · 44 | 2 · 3 · 39 |
| Judgment model time / run wall-clock | 89 s · 112 s | 264 s · 268 s |

Both candidate misses are **formatting failures, not wrong labels**. With a near-miss distractor present, the model wrote a runaway explanation until it hit `num_predict` (`done_reason=length`, unterminated JSON), on the first attempt and on the retry. Each of its 34 valid judgments had the expected label and cited the relevant passage. Remaining quality issues: some evidence excerpts quote a pronoun sentence without the antecedent sentence, and explanations sometimes overstate wording ("explicitly states that Rafael Ortiz…" when the passage says "Ortiz"). These results do not by themselves justify switching judges; the real-data sample and human validation still apply.

Outputs:
- Baseline run: `results/evaluator_diagnostics/evaluator_diag_20261028T003414829923Z_66fdf154564c`
- Candidate run: `results/evaluator_diagnostics/evaluator_diag_20261028T005839983616Z_7d12cdf673ea`
- Comparison table: `results/evaluator_diagnostics/comparison_20261028T010314208995Z.md`. This table's citation row ("16/18", "0/18") wrongly counted failed judgments as citing nothing. The corrected figures are above and in the later table below.

#### Result: `qwen2.5:7b-instruct` for both extraction and judgment (2026-10-28)

Development diagnostic, not independent validation. Same frozen set, prompts (`claim_extraction_v5` / `claim_judgment_v9`), schemas and options; started with `--model qwen2.5:7b-instruct`. Run: `results/evaluator_diagnostics/evaluator_diag_20261028T010828943558Z_50ac1e1618aa`. Three-way table: `results/evaluator_diagnostics/comparison_20261028T011038946283Z.md`.

**Extraction** (reported separately from judgment):

| Extraction metric | llama3.2:3b | qwen2.5:7b-instruct |
|---|---|---|
| Required facts preserved | 16/18 | 11/18 |
| Response behaviour correct | 14/14 | 13/14 (a plain denial was labelled `partial_abstention`) |
| Unmatched claims (manual review) | 1/18 (name dropped; not invented) | 2/14 (both drop a name; neither invented) |
| Structural failures / retries | 0/14 · 2 retries | 0/14 · 0 retries |

Qwen's omissions: it dropped the unsupported added quantity and date entirely ("employed 300 workers", "in 1874"). Those are exactly the hallucinated parts the evaluator must surface. It also dropped relationships: "built by Corvane Optics", "Hana Ruiz = second director", and the "second bookshop = Inkwell & Oak" identity (it kept "her second bookshop" without the name). The relationship heuristic flagged the Ruiz case but missed Inkwell & Oak, because its name pattern stops at "&".

**Judgment on this run's own claims:** 24/24 correct (SUPPORTED 18/18, CONTRADICTED 6/6, NOT_ENOUGH_EVIDENCE 0/0). There were 0 failures, 0 label changes from distractors, and 0/14 valid with-distractor judgments cited a distractor. **This is not comparable with the 34/36 reused-claims result.** Qwen's extraction produced a different, smaller claim set (28 judgments instead of 36). The two NOT_ENOUGH_EVIDENCE facts it dropped were never judged, so the hardest class has no denominator here.

**Eight-record real-data check (Qwen both stages):** `results/claim_evaluation/claim_eval_20261028T011038332358Z_cd43056e9f62`. It completed with 0 failures, 0 retries and 0 `done_reason=length` completions, producing 7 claims (6 judged SUPPORTED; 1 no-retrieval claim not applicable).
- The Marin answer became one unsplit compound claim keeping "his first fort, Fort Presque Isle" and both locations. It was judged SUPPORTED with the relevant passage ("He first constructed Fort Presque Isle (near present-day Erie, Pennsylvania) on Lake Erie's south shore."). Because it was not split, the location parts are not labelled individually.
- The no-retrieval clarification response is `full_abstention` with no claims.
- **6/8 answers were misclassified as `partial_abstention`.** These are all answers beginning "According to [Doc 1], …"; none withholds anything. This would distort abstention metrics on the full benchmark.
- Citations: 0/6 valid judgments cite an irrelevant passage (manual inspection); 0 failed judgments.

Do not adopt Qwen for extraction on this evidence. As judge only, it remains a candidate pending human validation.

#### Result: provisional mixed evaluator, `llama3.2:3b` extraction + `qwen2.5:7b-instruct` judgment (2026-10-28)

Eight-record real-data check, run `results/claim_evaluation/claim_eval_20261028T012544872471Z_41b65c4e8c7c`. Development check only, not validation. Prompts `claim_extraction_v5` / `claim_judgment_v9`, default options. Status `completed_with_errors`; the CLI exits with code 1 for this status.

- **Abstention:** 1 `full_abstention` (the no-retrieval clarification response, no claims; the first extraction attempt wrongly listed claims and the logged retry fixed it) and 7 `no_abstention`, all correct.
- **Extraction:**
  - Marin answers split into 3 claims ("Marin built Fort Presque Isle", "…near present-day Erie, Pennsylvania", "…on the south shore of Lake Erie"). The **"first fort" relationship is lost** in all three configurations; the extraction warning flags it.
  - Static-friction answers became one claim that **drops the qualifier** "up to an upper limit determined by the characteristics of the contact…". No heuristic flags this.
  - No added facts.
- **Judgment:** 10/12 judged claims are SUPPORTED, each citing the relevant passage. The 2 others **failed**: "Fort Presque Isle was built near present-day Erie, Pennsylvania" in both dense configurations (identical passages). Both attempts looped in the explanation until `num_predict` (`done_reason=length`, unterminated JSON). They are recorded as errors, not repaired, with label `null`.
- **Citations:** 0/10 valid judgments cite an irrelevant passage (manual check); 2 failed judgments are **unassessed**.
- **Compound claims:** the Qwen-extraction run (`claim_eval_20261028T011038332358Z_cd43056e9f62`) kept the Marin sentence as one compound claim. A compound claim can be supported and is not by itself an extraction failure. But one label then covers several facts, so a mixed-support answer would be scored entirely by its least-supported part, and per-fact errors would be hidden.

Answer-level `review_reason` previously showed only the judgment failure and hid extraction warnings; it now appends both (fixed after this run; the saved run is unchanged).

**Citation metric change.** Comparison tables now use `valid_judgments_denominator_v2`: distractor-citation rates divide by valid judgments, and failed judgments are reported as unassessed. Older saved files use the previous denominator, in which failed judgments implicitly counted as citing nothing. They are kept unchanged: the `report.md`/`manifest.json` of every diagnostic run up to `evaluator_diag_20261028T010828943558Z_50ac1e1618aa`, and `comparison_20261028T003747027926Z.md` / `comparison_20261028T010314208995Z.md`. Re-running `--compare` on any saved runs recomputes them with the corrected definition.

### Step 20: human review workflow (not yet completed)

`src/human_review.py` prepares blinded human review. No human labels exist yet, and **human validation is not complete**.

1. **Sample** (done, 2026-10-28): `results/human_review/review_20261028T012916Z_seed42/`. The sample was drawn from the saved generation records only, before any automatic judgment of the sampled answers existed.
   - **Size:** 36 answers, 6 per (RAG configuration × SQuAD answerability) stratum across `bm25_basic`, `dense_basic` and `dense_abstain`. Each question appears once. `no_retrieval` is excluded because it has no supplied passages.
   - **Exclusions:** questions already inspected during debugging, found by scanning every `results/claim_evaluation/*/answers.jsonl` and other generation files. This found the two development-check questions (`5733ea04…`, `5ad26a5f…`). Any other question you inspected by hand should be added with `--exclude-question-id`, and the sample redrawn before evaluation.
   - **Record:** the exact procedure, exclusions and stratum counts are in `coordinator/sample_manifest.json`.
2. **Evaluate the sample** (done, 2026-10-28): 36 answers, not the full benchmark. Run `coordinator/evaluation/claim_eval_20261028T015900914560Z_0dd1e4b7fb61`, status `completed_with_errors`.
   - **Frozen evaluator** (`coordinator/frozen_evaluator.json`):
     - Extraction: `llama3.2:3b` (digest `a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72`).
     - Judgment: `qwen2.5:7b-instruct` (digest `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`).
     - Prompts `claim_extraction_v5` / `claim_judgment_v9`, with their SHA-256 hashes; default options; one retry.
     - `export` refuses any evaluation run with a different configuration. Do not tune this evaluator during the validation round.
   - **Workflow totals only:** automatic labels for the held-out sample are deliberately not reported here before human review.
     - Answers: 36 processed, all present in the packet.
     - Claims: 55 extracted.
     - Extraction: unavailable for 5 answers (3 non-verbatim source excerpts, 1 negation check, 1 output limit). These are shown to reviewers with full answers, passages and a blank claim row.
     - Judgment: 19 of 55 claims failed, 18 of them from runaway output hitting `num_predict` (`done_reason=length`). Their labels are null and the failures are recorded in `coordinator/mapping.json`.
     - 17 answers have at least one failure.
     - This structural failure rate is far above the synthetic diagnostic's (2/36) and must be addressed before any 400-record run. That would be a new evaluator version, which needs a fresh held-out validation sample.
3. **Export** (done) the blinded packet to `reviewer/`:
   - `review_packet.md`: complete answers, full passages labelled Doc 1… in generator order, extracted claims, and an explicit EXTRACTION UNAVAILABLE status where needed. It sits alongside the instructions and blank `answers.csv` (36 rows) and `claims.csv` (60 rows: 55 extracted claims plus 5 blank rows for extraction-unavailable answers).
   - 504 blank manual fields in total.
   - Hidden from reviewers: automatic labels, explanations, evidence excerpts, review warnings, failure reasons, model/configuration names, and SQuAD answerability/reference answers.
   - `coordinator/mapping.json` holds source IDs, every automatic decision, extraction warnings, and judgment/extraction failure reasons with per-attempt `done_reason`. Reviewers must not open it.
   - The target is 50–100 claims. Sample size was fixed beforehand, so the count was never adjusted by resampling.
4. **Verify** (done; all checks passed; `coordinator/verification_20261028T022433735278Z.json`). The checks:
   - 36 answers, with item IDs and generation lines matching the frozen sample.
   - The sample input and the 400-record generation file are unchanged (SHA-256 `0caf7497…`).
   - Every item contains its full answer and every passage.
   - No automatic decisions or source metadata in reviewer files.
   - Only reviewer columns, and no human field filled.
5. **Review** (not started): reviewers fill support labels, supporting/contradicting evidence, missing claims, omitted qualifications and relationships, distorted-claim flags, abstention behaviour, and notes, following `report/annotation_guidelines.md`. They may add missing claims as `<item>-M1` rows.
6. **Compare:** writes `coordinator/agreement_<timestamp>.json`. It reports partial labelling as partial and never declares validation complete.

```powershell
python src/human_review.py sample --seed 42 --per-stratum 6        # already run; re-running makes a new directory
python src/claim_evaluation.py --input results/human_review/review_20261028T012916Z_seed42/coordinator/sample_input.jsonl --limit 36 --extraction-model llama3.2:3b --judgment-model qwen2.5:7b-instruct --output-root results/human_review/review_20261028T012916Z_seed42/coordinator/evaluation
python src/human_review.py export --review-dir results/human_review/review_20261028T012916Z_seed42 --evaluation-run results/human_review/review_20261028T012916Z_seed42/coordinator/evaluation/claim_eval_20261028T015900914560Z_0dd1e4b7fb61
python src/human_review.py verify --review-dir results/human_review/review_20261028T012916Z_seed42   # prints check results and totals only
python src/human_review.py compare --review-dir results/human_review/review_20261028T012916Z_seed42  # after reviewers fill the CSVs
```

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
