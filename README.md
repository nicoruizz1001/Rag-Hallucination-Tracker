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

Each invocation creates a unique run directory under `results/claim_evaluation/`, with `answers.jsonl`, `claims.jsonl`, and `manifest.json`. The manifest and stage records include evaluator model identity/settings, prompts, raw model responses, latency, counts, and failures. The evaluator defaults to local Ollama model `llama3.2:3b`.

Using the same model as generator and judge can produce correlated errors. Human validation, following [the annotation guidelines](report/annotation_guidelines.md), is required before reporting findings. The evaluator's schema and provenance checks validate output structure and excerpt references; they do not establish semantic correctness.

## Environment

- Python 3.11
- LLM: `llama3.2:3b` via Ollama (record any change — it invalidates direct comparisons)
- Exact package versions are pinned in `requirements.txt`.

On a fixed sample of 100 SQuAD 2.0 validation questions and a corpus of 500 unique passages, BM25 achieved Source Hit@1 of 81%, Source Hit@5 of 94%, and MRR@5 of 0.865. For the 48 answerable questions, Source Hit@5 was 95.8%. These metrics measure retrieval of the assigned source passage, not answer correctness or corpus-wide answerability.

BM25 achieved higher Source Hit@1 (81% versus 78%) and MRR@5 (0.865 versus 0.854), while dense retrieval achieved slightly higher Source Hit@5 (95% versus 94%). Both methods retrieved the assigned source within the top five for 46 of 48 answerable questions. Results apply to our fixed 100-question sample and 500-passage corpus.
