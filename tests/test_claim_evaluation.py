import hashlib
import json

import pytest

from src import claim_evaluation


def model_result(content, *, digest="judge-digest", latency=0.01):
    return {
        "generated_answer": content,
        "model": "llama3.2:3b",
        "model_digest": digest,
        "generation_settings": {"temperature": 0, "num_predict": 1024},
        "latency_seconds": latency,
        "done_reason": "stop",
    }


def extraction_response(claims, behavior="no_abstention"):
    return json.dumps({"response_behavior": behavior, "claims": claims})


def judgment_response(label, evidence=None, explanation="synthetic judgment"):
    return json.dumps({
        "label": label,
        "evidence": evidence or [],
        "explanation": explanation,
        "needs_review": False,
        "review_reason": "",
    })


def test_extract_claims_preserves_independent_source_excerpts(monkeypatch):
    answer = "Marin built Fort Presque Isle in 1753 using 500 workers."
    raw = extraction_response([
        {"claim": "Marin built Fort Presque Isle in 1753.",
         "source_excerpt": "Marin built Fort Presque Isle in 1753"},
        {"claim": "Marin used 500 workers.",
         "source_excerpt": "using 500 workers"},
    ])
    monkeypatch.setattr(
        claim_evaluation, "generate_answer", lambda *a, **kw: model_result(raw)
    )

    result = claim_evaluation.extract_claims("Who built the fort?", answer)

    assert [item["claim"] for item in result["claims"]] == [
        "Marin built Fort Presque Isle in 1753.", "Marin used 500 workers."
    ]
    assert all(item["source_excerpt"] in answer for item in result["claims"])
    assert result["response_behavior"] == "no_abstention"
    assert result["needs_review"] is False
    assert result["prompt_version"] == claim_evaluation.EXTRACTION_PROMPT_VERSION


def test_nonabstaining_answer_with_no_claims_is_flagged_for_review(monkeypatch):
    raw = extraction_response([])
    monkeypatch.setattr(
        claim_evaluation, "generate_answer", lambda *a, **kw: model_result(raw)
    )

    result = claim_evaluation.extract_claims("What happened?", "An ordinary answer.")

    assert result["claims"] == []
    assert result["needs_review"] is True
    assert "zero factual claims" in result["review_reason"]


def test_claim_source_excerpt_must_be_verbatim(monkeypatch):
    raw = extraction_response([{
        "claim": "Marin built the fort.", "source_excerpt": "Marin built another fort."
    }])
    monkeypatch.setattr(
        claim_evaluation, "generate_answer", lambda *a, **kw: model_result(raw)
    )

    with pytest.raises(claim_evaluation.StructuredOutputError) as error:
        claim_evaluation.extract_claims("Who built it?", "Marin built the fort.")

    assert len(error.value.attempts) == 2
    assert "not verbatim" in error.value.attempts[0]["validation_error"]


@pytest.mark.parametrize(
    "label,evidence,expected",
    [
        ("SUPPORTED", [{"document_id": "doc-1", "excerpt": "Marin built the fort in 1753."}], "SUPPORTED"),
        ("CONTRADICTED", [{"document_id": "doc-1", "excerpt": "Marin did not build the fort in 1753."}], "CONTRADICTED"),
        ("NOT_ENOUGH_EVIDENCE", [], "NOT_ENOUGH_EVIDENCE"),
    ],
)
def test_check_claim_returns_valid_judgment(monkeypatch, label, evidence, expected):
    raw = judgment_response(label, evidence)
    monkeypatch.setattr(
        claim_evaluation, "generate_answer", lambda *a, **kw: model_result(raw)
    )

    result = claim_evaluation.check_claim(
        "Marin built the fort in 1753.",
        [{"document_id": "doc-1", "text": "Marin built the fort in 1753." if label == "SUPPORTED" else "Marin did not build the fort in 1753."}],
    )

    assert result["label"] == expected
    assert result["needs_review"] is False
    assert result["prompt_version"] == claim_evaluation.JUDGMENT_PROMPT_VERSION


@pytest.mark.parametrize(
    "answer,behavior,claims",
    [
        ("Insufficient information to answer.", "full_abstention", []),
        ("I cannot give the population, but it has a lighthouse.", "partial_abstention", [{"claim": "The island has a lighthouse.", "source_excerpt": "it has a lighthouse"}]),
    ],
)
def test_extract_claims_classifies_abstentions_and_keeps_factual_claims(
    monkeypatch, answer, behavior, claims
):
    raw = extraction_response(claims, behavior)
    monkeypatch.setattr(
        claim_evaluation, "generate_answer", lambda *a, **kw: model_result(raw)
    )

    result = claim_evaluation.extract_claims("What is its population?", answer)

    assert result["response_behavior"] == behavior
    assert result["claims"] == claims


def test_empty_response_is_not_treated_as_abstention(monkeypatch):
    def unexpected_call(*args, **kwargs):
        pytest.fail("empty answers should not be sent to the model")

    monkeypatch.setattr(claim_evaluation, "generate_answer", unexpected_call)

    result = claim_evaluation.extract_claims("question", "")

    assert result["response_behavior"] == "empty_response"
    assert result["claims"] == []
    assert result["latency_seconds"] == 0


@pytest.mark.parametrize(
    "raw",
    [
        "not-json",
        json.dumps({"label": "MAYBE", "evidence": [], "explanation": "bad"}),
    ],
)
def test_invalid_structured_output_retries_once_then_retains_attempts(monkeypatch, raw):
    calls = []

    def fake_generate(*args, **kwargs):
        calls.append(kwargs)
        return model_result(raw)

    monkeypatch.setattr(claim_evaluation, "generate_answer", fake_generate)

    with pytest.raises(claim_evaluation.StructuredOutputError) as error:
        claim_evaluation.check_claim("claim", [])

    assert len(calls) == 2
    assert len(error.value.attempts) == 2
    assert all(attempt["raw_response"] == raw for attempt in error.value.attempts)
    assert error.value.attempts[1]["retry_of"] == 1


def test_model_call_failure_is_logged_without_retry(monkeypatch):
    calls = []

    def fail(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("Ollama unavailable")

    monkeypatch.setattr(claim_evaluation, "generate_answer", fail)

    with pytest.raises(claim_evaluation.StructuredOutputError) as error:
        claim_evaluation.check_claim("claim", [])

    assert len(calls) == 1
    assert len(error.value.attempts) == 1
    assert error.value.attempts[0]["raw_response"] is None


@pytest.mark.parametrize(
    "judgment",
    [
        {"label": "SUPPORTED", "evidence": [{"document_id": "unknown", "excerpt": "text"}], "explanation": "x"},
        {"label": "CONTRADICTED", "evidence": [{"document_id": "doc-1", "excerpt": "invented quote"}], "explanation": "x"},
    ],
)
def test_judgment_rejects_unknown_documents_and_fabricated_excerpts(monkeypatch, judgment):
    raw = json.dumps(judgment)
    monkeypatch.setattr(
        claim_evaluation, "generate_answer", lambda *a, **kw: model_result(raw)
    )

    with pytest.raises(claim_evaluation.StructuredOutputError):
        claim_evaluation.check_claim(
            "claim", [{"document_id": "doc-1", "text": "actual passage text"}]
        )


def test_no_retrieval_claim_support_is_not_applicable_and_input_is_unchanged(
    tmp_path, monkeypatch
):
    generated = "Marin built the fort in 1753."
    input_record = {
        "run_id": "source-run", "question_id": "q1", "question": "Who built it?",
        "configuration": "no_retrieval", "retrieval_method": "none",
        "retrieved_document_ids": [], "retrieved_passage_texts": [],
        "generated_answer": generated, "done_reason": "stop",
    }
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps(input_record) + "\n", encoding="utf-8")
    original = source.read_bytes()
    raw = extraction_response([{
        "claim": "Marin built the fort in 1753.",
        "source_excerpt": generated,
    }])
    monkeypatch.setattr(
        claim_evaluation, "generate_answer", lambda *a, **kw: model_result(raw)
    )

    run = claim_evaluation.evaluate_file(source, output_root=tmp_path / "derived", limit=1)

    answer = json.loads((run / "answers.jsonl").read_text(encoding="utf-8"))
    claim = json.loads((run / "claims.jsonl").read_text(encoding="utf-8"))
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert answer["evidence_support_status"] == "not_applicable"
    assert answer["faithfulness"] == "not_applicable"
    assert claim["label"] is None
    assert claim["evidence_support_status"] == "not_applicable"
    assert source.read_bytes() == original
    assert manifest["input_sha256"] == hashlib.sha256(original).hexdigest()
    assert answer["source_run_id"] == claim["source_run_id"] == "source-run"


def test_failed_retry_is_saved_as_error_with_raw_responses(tmp_path, monkeypatch):
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps({
        "run_id": "r", "question_id": "q", "question": "Q?",
        "configuration": "bm25_basic", "retrieved_document_ids": [],
        "retrieved_passage_texts": [], "generated_answer": "Answer.",
    }) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        claim_evaluation, "generate_answer", lambda *a, **kw: model_result("bad json")
    )

    run = claim_evaluation.evaluate_file(source, output_root=tmp_path / "derived")
    answer = json.loads((run / "answers.jsonl").read_text(encoding="utf-8"))
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))

    assert answer["evaluation_status"] == "error"
    assert answer["evidence_support_status"] == "error"
    assert len(answer["extraction"]["attempts"]) == 2
    assert manifest["failure_count"] == 1
    assert manifest["run_status"] == "completed_with_errors"


def test_empty_and_pure_abstention_have_answer_level_records_and_zero_claim_faithfulness(
    tmp_path, monkeypatch
):
    records = [
        {"run_id": "r", "question_id": "empty", "question": "Q", "configuration": "bm25_basic", "retrieved_document_ids": [], "retrieved_passage_texts": [], "generated_answer": ""},
        {"run_id": "r", "question_id": "abstain", "question": "Q", "configuration": "bm25_basic", "retrieved_document_ids": [], "retrieved_passage_texts": [], "generated_answer": "Insufficient information."},
    ]
    source = tmp_path / "input.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    raw = extraction_response([], "full_abstention")
    monkeypatch.setattr(
        claim_evaluation, "generate_answer", lambda *a, **kw: model_result(raw)
    )

    run = claim_evaluation.evaluate_file(source, output_root=tmp_path / "derived", limit=2)
    answers = [json.loads(line) for line in (run / "answers.jsonl").read_text(encoding="utf-8").splitlines()]
    claims = (run / "claims.jsonl").read_text(encoding="utf-8")

    assert [item["response_behavior"] for item in answers] == ["empty_response", "full_abstention"]
    assert all(item["faithfulness"] is None for item in answers)
    assert claims == ""


def test_unsupported_added_claim_and_demo_content_are_not_hardcoded(monkeypatch):
    calls = []
    extraction = extraction_response([
        {"claim": "Marin built Fort Presque Isle in 1753.", "source_excerpt": "Marin built Fort Presque Isle in 1753"},
        {"claim": "Marin used 500 workers.", "source_excerpt": "using 500 workers"},
    ])

    def fake_generate(system, user, **kwargs):
        calls.append((system, user, kwargs))
        if "claim extractor" in system.lower():
            return model_result(extraction)
        if '"claim": "marin built fort presque isle in 1753.' in user.lower():
            return model_result(judgment_response(
                "SUPPORTED", [{"document_id": "demo-doc", "excerpt": "Marin built Fort Presque Isle in 1753."}]
            ))
        return model_result(judgment_response("NOT_ENOUGH_EVIDENCE", []))

    monkeypatch.setattr(claim_evaluation, "generate_answer", fake_generate)

    result = claim_evaluation.run_synthetic_demo(output_root=None)

    assert [claim["label"] for claim in result["claims"]] == [
        "SUPPORTED", "NOT_ENOUGH_EVIDENCE"
    ]
    assert len(calls) == 3


def test_cli_synthetic_demo_prints_stage_results(monkeypatch, capsys):
    monkeypatch.setattr(claim_evaluation, "run_synthetic_demo", lambda **kw: {
        "answer": {"response_behavior": "no_abstention"},
        "claims": [{"claim": "A synthetic claim", "label": "SUPPORTED"}],
    })

    assert claim_evaluation.main(["--synthetic-demo"]) == 0
    output = capsys.readouterr().out
    assert "Extracted claims and judgments" in output
    assert "SUPPORTED" in output


def test_repeated_runs_use_unique_directories_and_stable_ids(tmp_path, monkeypatch):
    record = {
        "run_id": "source", "question_id": "q", "question": "Q",
        "configuration": "no_retrieval", "retrieved_document_ids": [],
        "retrieved_passage_texts": [], "generated_answer": "A fact.",
    }
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        claim_evaluation, "generate_answer",
        lambda *a, **kw: model_result(extraction_response([{
            "claim": "A fact.", "source_excerpt": "A fact."
        }])),
    )

    first = claim_evaluation.evaluate_file(source, output_root=tmp_path / "derived")
    second = claim_evaluation.evaluate_file(source, output_root=tmp_path / "derived")
    first_answer = json.loads((first / "answers.jsonl").read_text(encoding="utf-8"))
    second_answer = json.loads((second / "answers.jsonl").read_text(encoding="utf-8"))
    first_claim = json.loads((first / "claims.jsonl").read_text(encoding="utf-8"))
    second_claim = json.loads((second / "claims.jsonl").read_text(encoding="utf-8"))

    assert first != second
    assert first_answer["answer_id"] == second_answer["answer_id"]
    assert first_claim["claim_id"] == second_claim["claim_id"]
