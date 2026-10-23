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
| `src/` | Pipeline code: data prep, retrieval, generation, evaluation, experiments, plots |
| `configs/` | Experiment configurations |
| `data/` | Downloaded datasets and indexes (git-ignored) |
| `results/` | Saved experiment outputs, tables, figures |
| `notebooks/` | Exploratory analysis |
| `report/` | Research write-up |
| `tests/` | pytest suite |

## Environment

- Python 3.11
- LLM: `llama3.2:3b` via Ollama (record any change — it invalidates direct comparisons)
- Exact package versions are pinned in `requirements.txt`.
