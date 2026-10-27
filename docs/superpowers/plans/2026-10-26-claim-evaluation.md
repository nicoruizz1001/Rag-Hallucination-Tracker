# Claim Evaluation Implementation Plan

> **For agentic workers:** Implement inline using test-first steps.

**Goal:** Add a separate, provenance-validated claim extraction and support judgment CLI over saved generation JSONL.

**Architecture:** Keep retrieval evaluation unchanged. Extend the Ollama wrapper with an optional structured JSON format and place extraction, judgment, validation, run persistence, and CLI code in `src/claim_evaluation.py`.

**Tech Stack:** Python, Ollama, pytest, JSONL.

**Spec:** User request for step 19; annotation protocol in `report/annotation_guidelines.md`.

## Global Constraints

- Never regenerate or modify generation inputs.
- Do not use reference answers, SQuAD labels, or outside knowledge.
- Retry malformed model output at most once and record both attempts.
- Keep all derived results under unique `results/claim_evaluation/<run_id>/` directories.
- Run only mocked tests; do not run the full benchmark.

## Review Focus

- Preserve backwards compatibility for existing generation calls.
- Reject fabricated claim/evidence provenance and unknown document IDs.
- Retain answer-level records for empty and pure-abstention answers.
- Keep no-retrieval labels not applicable and keep zero-claim faithfulness undefined.
- Flush completed answers and claims incrementally, including on validation errors.

### Task 1: Structured model calls and annotation stages

**Files:** Modify `src/generation.py`; create `src/claim_evaluation.py`; create `tests/test_claim_evaluation.py`.

- [ ] Add failing tests for extraction, judgments, output validation, one retry, abstentions, empty answers, and no-retrieval.
- [ ] Verify the new tests fail before implementation.
- [ ] Add optional structured JSON support to the Ollama wrapper and implement versioned extraction/judgment prompts plus schema/provenance validation.
- [ ] Verify mocked unit tests pass.

### Task 2: Saved-run processing, CLI, and documentation

**Files:** Modify `src/claim_evaluation.py`; modify `README.md`; extend `tests/test_claim_evaluation.py`.

- [ ] Add failing tests for incremental JSONL outputs, stable IDs, unique run directories, retry failure records, source immutability, and synthetic demo plumbing.
- [ ] Verify tests fail before implementation.
- [ ] Implement the default one-record smoke run, optional input limit, synthetic demo, unique output manifest, and README commands/limitations.
- [ ] Run focused tests, then the requested full pytest suite and diff checks; ensure retrieval evaluation behavior and saved generation outputs are unchanged.
