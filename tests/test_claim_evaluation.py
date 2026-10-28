import hashlib
import json
from pathlib import Path

import pytest

from src import claim_evaluation


def model_result(content, *, digest="judge-digest", latency=0.01, done_reason="stop"):
    return {
        "generated_answer": content,
        "model": "llama3.2:3b",
        "model_digest": digest,
        "generation_settings": {"temperature": 0, "num_predict": 1024},
        "latency_seconds": latency,
        "done_reason": done_reason,
    }


def extraction_response(claims, behavior="no_abstention"):
    return json.dumps({"response_behavior": behavior, "claims": claims})


def judgment_response(label, evidence=None, explanation="Synthetic judgment."):
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
    assert claim_evaluation.EXTRACTION_PROMPT_VERSION == "claim_extraction_v5"
    assert result["prompt_version"] == "claim_extraction_v5"


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


def test_citation_only_excerpt_is_rejected_for_complete_claim(monkeypatch):
    raw = extraction_response([{
        "claim": "Marin built Fort Presque Isle near Erie.",
        "source_excerpt": "According to [Doc 1],",
    }])
    monkeypatch.setattr(
        claim_evaluation, "generate_answer", lambda *a, **kw: model_result(raw)
    )

    with pytest.raises(claim_evaluation.StructuredOutputError) as error:
        claim_evaluation.extract_claims(
            "Where did Marin build the fort?",
            "According to [Doc 1], Marin built Fort Presque Isle near Erie.",
        )

    assert len(error.value.attempts) == 2
    assert "citation-only" in error.value.attempts[0]["validation_error"]


def test_extraction_validation_does_not_drop_source_negation(monkeypatch):
    answer = "The ferry did not operate in winter 2022."
    raw = extraction_response([{
        "claim": "The ferry operated in winter 2022.", "source_excerpt": answer,
    }])
    monkeypatch.setattr(
        claim_evaluation, "generate_answer", lambda *a, **kw: model_result(raw)
    )

    with pytest.raises(claim_evaluation.StructuredOutputError) as error:
        claim_evaluation.extract_claims("When did the ferry operate?", answer)

    assert "negation" in error.value.attempts[0]["validation_error"]


def test_judgment_schema_bounds_quotes_explanations_and_output_budget(monkeypatch):
    calls = []
    raw = judgment_response("NOT_ENOUGH_EVIDENCE", [])

    def fake_generate(*args, **kwargs):
        calls.append(kwargs)
        return model_result(raw)

    monkeypatch.setattr(claim_evaluation, "generate_answer", fake_generate)
    result = claim_evaluation.check_claim("A claim.", [])

    evidence_schema = claim_evaluation.JUDGMENT_SCHEMA["properties"]["evidence"]
    item_schema = evidence_schema["items"]["properties"]
    assert evidence_schema["maxItems"] == 2
    # Grammar-level maxLength truncates strings mid-sentence, so limits live in validation.
    assert "maxLength" not in item_schema["excerpt"]
    assert "maxLength" not in claim_evaluation.JUDGMENT_SCHEMA["properties"]["explanation"]
    assert calls[0]["generation_settings"]["num_predict"] == 1536
    assert result["attempts"][0]["done_reason"] == "stop"


def test_length_done_reason_retries_even_valid_json_and_logs_both_attempts(monkeypatch):
    calls = []
    raw = judgment_response("NOT_ENOUGH_EVIDENCE", [])

    def fake_generate(*args, **kwargs):
        calls.append(kwargs)
        done_reason = "length" if len(calls) == 1 else "stop"
        return model_result(raw, done_reason=done_reason)

    monkeypatch.setattr(claim_evaluation, "generate_answer", fake_generate)
    result = claim_evaluation.check_claim("A claim.", [])

    assert len(calls) == 2
    assert [attempt["done_reason"] for attempt in result["attempts"]] == ["length", "stop"]
    assert "done_reason=length" in result["attempts"][0]["validation_error"]


def test_length_response_after_retry_is_failure_even_when_json_is_valid(monkeypatch):
    raw = judgment_response("NOT_ENOUGH_EVIDENCE", [])
    monkeypatch.setattr(
        claim_evaluation, "generate_answer",
        lambda *a, **kw: model_result(raw, done_reason="length"),
    )

    with pytest.raises(claim_evaluation.StructuredOutputError) as error:
        claim_evaluation.check_claim("A claim.", [])

    assert len(error.value.attempts) == 2
    assert all(attempt["raw_response"] == raw for attempt in error.value.attempts)
    assert all(attempt["done_reason"] == "length" for attempt in error.value.attempts)


def test_judgment_rejects_evidence_and_explanations_over_limits():
    passage = "x" * 500
    docs = [{"document_id": f"doc-{i}", "text": passage} for i in range(4)]
    too_many = {
        "label": "SUPPORTED",
        "evidence": [{"document_id": f"doc-{i}", "excerpt": "x"} for i in range(4)],
        "explanation": "Supported.", "needs_review": False, "review_reason": "",
    }
    with pytest.raises(ValueError, match="at most 2"):
        claim_evaluation._validate_judgment(too_many, docs)
    too_long = {**too_many, "evidence": [{"document_id": "doc-0", "excerpt": "x" * 401}]}
    with pytest.raises(ValueError, match="400 characters"):
        claim_evaluation._validate_judgment(too_long, docs)
    too_many_sentences = {**too_long, "evidence": [], "label": "NOT_ENOUGH_EVIDENCE",
                          "explanation": "One. Two. Three."}
    with pytest.raises(ValueError, match="2 short sentences"):
        claim_evaluation._validate_judgment(too_many_sentences, docs)


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
    assert claim_evaluation.JUDGMENT_PROMPT_VERSION == "claim_judgment_v9"
    assert result["prompt_version"] == "claim_judgment_v9"


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
    assert manifest["not_applicable_answer_count"] == 1
    assert manifest["not_applicable_claim_count"] == 1


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
    assert answer["needs_review"] is True
    assert answer["review_reason"]
    assert len(answer["extraction"]["attempts"]) == 2
    assert answer["extraction"]["attempts"][0]["raw_response"] == "bad json"
    assert answer["extraction"]["attempts"][0]["validation_error"]
    assert manifest["failure_count"] == 1
    assert manifest["run_status"] == "completed_with_errors"
    assert manifest["successful_answer_count"] == 0
    assert manifest["failed_answer_count"] == 1
    assert manifest["not_applicable_answer_count"] == 0
    assert manifest["failed_claim_count"] == 0


def test_failed_judgment_keeps_null_label_and_flags_claim_and_answer(tmp_path, monkeypatch):
    passage = "The harbor opened in 1901."
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps({
        "run_id": "r", "question_id": "q", "question": "When did the harbor open?",
        "configuration": "bm25_basic", "retrieved_document_ids": ["doc-1"],
        "retrieved_passage_texts": [passage], "generated_answer": passage,
    }) + "\n", encoding="utf-8")
    calls = []

    def fake_generate(system, user, **kwargs):
        calls.append(system)
        if "claim extractor" in system.lower():
            return model_result(extraction_response([{
                "claim": passage, "source_excerpt": passage,
            }]))
        return model_result('{"label":"UNSURE"}')

    monkeypatch.setattr(claim_evaluation, "generate_answer", fake_generate)

    run = claim_evaluation.evaluate_file(source, output_root=tmp_path / "runs")
    answer = json.loads((run / "answers.jsonl").read_text(encoding="utf-8"))
    claim = json.loads((run / "claims.jsonl").read_text(encoding="utf-8"))
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))

    assert answer["needs_review"] is True
    assert answer["review_reason"].startswith("Judgment failed:")
    assert claim["needs_review"] is True
    assert claim["review_reason"].startswith("Judgment failed:")
    assert claim["label"] is None
    assert len(claim["judgment"]["attempts"]) == 2
    assert all(item["raw_response"] == '{"label":"UNSURE"}' for item in claim["judgment"]["attempts"])
    assert manifest["failed_answer_count"] == 1
    assert manifest["failed_claim_count"] == 1


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


def test_synthetic_examples_are_persisted_and_decisions_come_from_model(
    tmp_path, monkeypatch
):
    evidence = "Marin built Fort Presque Isle in 1753."
    calls = []

    def fake_generate(system, user, **kwargs):
        calls.append((system, user))
        if "claim extractor" in system.lower():
            data = json.loads(user.split("\n", 1)[1])
            answer = data["generated_answer"]
            if "500 workers" in answer:
                claims = [
                    {"claim": "Marin used 500 workers.", "source_excerpt": answer},
                ]
            else:
                claims = [{"claim": answer, "source_excerpt": answer}]
            return model_result(extraction_response(claims))
        data = json.loads(user.split("\n", 1)[1])
        claim = data["claim"]
        if "did not build" in claim:
            return model_result(judgment_response(
                "CONTRADICTED", [{"document_id": "synthetic-doc", "excerpt": evidence}]
            ))
        if "500 workers" in claim:
            return model_result(judgment_response("NOT_ENOUGH_EVIDENCE", []))
        return model_result(judgment_response(
            "SUPPORTED", [{"document_id": "synthetic-doc", "excerpt": evidence}]
        ))

    monkeypatch.setattr(claim_evaluation, "generate_answer", fake_generate)

    run = claim_evaluation.run_synthetic_examples(output_root=tmp_path / "runs")
    claims = [json.loads(line) for line in (run / "claims.jsonl").read_text(encoding="utf-8").splitlines()]
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))

    assert [claim["label"] for claim in claims] == [
        "SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_EVIDENCE"
    ]
    assert manifest["synthetic_examples"] is True
    assert Path(manifest["input_path"]).is_file()
    assert manifest["successful_answer_count"] == 3
    assert manifest["successful_claim_count"] == 3
    assert len(calls) == 6


# ---------------------------------------------------------------------------
# Software regression tests for refusal handling and relationship preservation.
# These use mocked model responses: they check the evaluator's validation and
# flagging logic, NOT the quality of any live model. Live behaviour is measured
# separately by src/evaluator_diagnostics.py.
# ---------------------------------------------------------------------------


def sequenced_model(monkeypatch, responses):
    calls = []

    def fake_generate(*args, **kwargs):
        calls.append(kwargs)
        return model_result(responses[min(len(calls), len(responses)) - 1])

    monkeypatch.setattr(claim_evaluation, "generate_answer", fake_generate)
    return calls


REFUSAL = ("I don't have information on a person named Odalys building a lighthouse. "
           "Could you provide more context or clarify who Odalys is?")


@pytest.mark.parametrize("claim", [
    "I don't have information on a person named Odalys building a lighthouse.",
    "Could you provide more context or clarify who Odalys is?",
])
def test_self_knowledge_and_clarification_claims_are_rejected(monkeypatch, claim):
    bad = extraction_response(
        [{"claim": claim, "source_excerpt": claim}], "partial_abstention"
    )
    good = extraction_response([], "full_abstention")
    calls = sequenced_model(monkeypatch, [bad, good])

    result = claim_evaluation.extract_claims("Where did Odalys build it?", REFUSAL)

    assert len(calls) == 2
    assert "abstention" in result["attempts"][0]["validation_error"]
    assert result["response_behavior"] == "full_abstention"
    assert result["claims"] == []


def test_refusal_claim_fails_after_retry_instead_of_becoming_a_fact(monkeypatch):
    claim = "I don't have information on a person named Odalys building a lighthouse."
    bad = extraction_response(
        [{"claim": claim, "source_excerpt": claim}], "partial_abstention"
    )
    sequenced_model(monkeypatch, [bad])

    with pytest.raises(claim_evaluation.StructuredOutputError):
        claim_evaluation.extract_claims("Where did Odalys build it?", REFUSAL)


def test_documents_do_not_mention_statement_is_abstention():
    assert claim_evaluation._is_abstention_statement(
        "The provided documents do not mention when the canal closed."
    )
    assert not claim_evaluation._is_abstention_statement(
        "The canal did not close until 1950."
    )
    assert not claim_evaluation._is_abstention_statement(
        "Ines Ruiz opened Sol Dorado in Valencia."
    )


def test_partial_abstention_without_world_claims_is_rejected(monkeypatch):
    sequenced_model(monkeypatch, [extraction_response([], "partial_abstention")])

    with pytest.raises(claim_evaluation.StructuredOutputError) as error:
        claim_evaluation.extract_claims("Q?", REFUSAL)

    assert "partial_abstention requires" in error.value.attempts[0]["validation_error"]


def test_partial_abstention_keeps_world_facts_only(monkeypatch):
    answer = "I could not find the canal's opening year, but it is 42 kilometres long."
    claims = [{"claim": "The canal is 42 kilometres long.",
               "source_excerpt": "it is 42 kilometres long"}]
    sequenced_model(monkeypatch, [extraction_response(claims, "partial_abstention")])

    result = claim_evaluation.extract_claims("How long is the canal?", answer)

    assert result["claims"] == claims
    assert result["needs_review"] is False


def test_dropped_appositive_relationship_is_flagged_for_review(monkeypatch):
    answer = ("According to [Doc 1], Odalys Brandt built her first lighthouse, "
              "Gull Point Light, near Harwick on the north cape.")
    excerpt = answer.split(", ", 1)[1]
    claims = [
        {"claim": "Odalys Brandt built her first lighthouse", "source_excerpt": excerpt},
        {"claim": "Gull Point Light was built near Harwick", "source_excerpt": excerpt},
    ]
    sequenced_model(monkeypatch, [extraction_response(claims)])

    result = claim_evaluation.extract_claims("Where was her first lighthouse?", answer)

    assert result["needs_review"] is True
    assert any("first lighthouse" in w and "Gull Point Light" in w
               for w in result["extraction_warnings"])


def test_preserved_appositive_relationship_is_not_flagged(monkeypatch):
    answer = ("According to [Doc 1], Odalys Brandt built her first lighthouse, "
              "Gull Point Light, near Harwick on the north cape.")
    excerpt = answer.split(", ", 1)[1]
    claims = [
        {"claim": "Odalys Brandt's first lighthouse was Gull Point Light.",
         "source_excerpt": excerpt},
        {"claim": "Odalys Brandt built Gull Point Light near Harwick on the north cape.",
         "source_excerpt": excerpt},
    ]
    sequenced_model(monkeypatch, [extraction_response(claims)])

    result = claim_evaluation.extract_claims("Where was her first lighthouse?", answer)

    assert result["extraction_warnings"] == []
    assert result["needs_review"] is False


def test_name_then_role_appositive_is_checked(monkeypatch):
    answer = "Hana Ruiz, the station's second director, introduced night shifts in 1979."
    claims = [{"claim": "Hana Ruiz introduced night shifts in 1979.", "source_excerpt": answer}]
    sequenced_model(monkeypatch, [extraction_response(claims)])

    result = claim_evaluation.extract_claims("Who introduced night shifts?", answer)

    assert any("second director" in w for w in result["extraction_warnings"])


def test_number_missing_from_all_claims_is_flagged(monkeypatch):
    answer = "The railway opened in 1902 and employed 300 workers."
    claims = [{"claim": "The railway opened in 1902.",
               "source_excerpt": "The railway opened in 1902"}]
    sequenced_model(monkeypatch, [extraction_response(claims)])

    result = claim_evaluation.extract_claims("When did it open?", answer)

    assert result["extraction_warnings"] == [
        "Number '300' from the answer appears in no extracted claim"
    ]


def test_extraction_prompt_uses_generic_examples_not_benchmark_entities():
    prompt = claim_evaluation.EXTRACTION_SYSTEM_PROMPT
    for benchmark_term in ("Marin", "Presque", "Erie", "friction"):
        assert benchmark_term not in prompt
    assert "drops which company" in prompt
    assert "requests for clarification" in prompt


def test_complete_explanation_over_old_schema_limit_is_accepted():
    # The old schema maxLength (200) cut explanations mid-sentence at decode time.
    explanation = ("The passage names the bridge and states that it opened to traffic "
                   "in the same year as the claim, and no other passage gives a different "
                   "date for this bridge, so the complete claim is established by the passage.")
    assert 200 < len(explanation) <= claim_evaluation.JUDGMENT_EXPLANATION_MAX_CHARS
    judgment = {"label": "NOT_ENOUGH_EVIDENCE", "evidence": [],
                "explanation": explanation, "needs_review": False, "review_reason": ""}

    assert claim_evaluation._validate_judgment(judgment, []) is judgment


def test_prompt_reaching_context_window_is_an_attempt_error(monkeypatch):
    raw = judgment_response("NOT_ENOUGH_EVIDENCE", [])

    def fake_generate(*args, **kwargs):
        result = model_result(raw)
        result["generation_settings"] = {"num_ctx": 8192}
        result["prompt_eval_count"] = 8192
        return result

    monkeypatch.setattr(claim_evaluation, "generate_answer", fake_generate)
    with pytest.raises(claim_evaluation.StructuredOutputError) as error:
        claim_evaluation.check_claim("A claim.", [])

    assert "reached num_ctx" in error.value.attempts[0]["validation_error"]
    assert error.value.attempts[0]["prompt_eval_count"] == 8192


def test_cli_configures_extraction_and_judgment_models_independently(tmp_path, monkeypatch):
    record = {
        "run_id": "r", "question_id": "q", "question": "When did it open?",
        "configuration": "bm25_basic", "retrieval_method": "bm25",
        "retrieved_document_ids": ["d1"], "retrieved_passage_texts": ["It opened in 1931."],
        "generated_answer": "It opened in 1931.",
    }
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")
    seen = []

    def fake_generate(system, user, **kwargs):
        stage = "extraction" if "claim extractor" in system.lower() else "judgment"
        seen.append((stage, kwargs["model"], kwargs["generation_settings"]["num_ctx"]))
        if stage == "extraction":
            return model_result(extraction_response(
                [{"claim": "It opened in 1931.", "source_excerpt": "It opened in 1931."}]))
        return model_result(judgment_response(
            "SUPPORTED", [{"document_id": "d1", "excerpt": "It opened in 1931."}]))

    monkeypatch.setattr(claim_evaluation, "generate_answer", fake_generate)
    claim_evaluation.main([
        "--input", str(source), "--output-root", str(tmp_path / "out"),
        "--judgment-model", "judge:test", "--judgment-options", '{"num_ctx": 4096}',
    ])
    manifest = json.loads(next((tmp_path / "out").glob("*/manifest.json")).read_text(encoding="utf-8"))

    assert seen == [("extraction", "llama3.2:3b", 8192), ("judgment", "judge:test", 4096)]
    assert manifest["extraction_model"] == "llama3.2:3b"
    assert manifest["judgment_model"] == "judge:test"
    assert manifest["options"]["judgment"]["num_ctx"] == 4096
