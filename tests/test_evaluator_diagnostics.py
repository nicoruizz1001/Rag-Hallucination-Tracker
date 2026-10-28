"""Software tests for the diagnostic harness, using a mocked Ollama client.

These check plumbing (label separation, scoring, request inspection). They say
nothing about live evaluator quality; run src/evaluator_diagnostics.py for that.
"""

import json

import ollama

from src import claim_evaluation, evaluator_diagnostics as diag


class PromptLog(list):
    """Prompts sent to the fake client; ``models`` holds (model, is_extraction)."""
    models = None


def fake_ollama(monkeypatch, *, explanation="Synthetic.", cite_distractor=False):
    prompts = PromptLog()
    prompts.models = []

    def fake_chat(**kwargs):
        system, user = (m["content"] for m in kwargs["messages"])
        prompts.append(system + "\n" + user)
        prompts.models.append((kwargs["model"], "extractor" in system.lower()))
        data = json.loads(user.split("\n", 1)[1])
        if "claim extractor" in system.lower():
            answer = data["generated_answer"]
            content = {"response_behavior": "no_abstention",
                       "claims": [{"claim": answer, "source_excerpt": answer}]}
        else:
            docs = data["retrieved_documents"]
            doc = docs[0] if cite_distractor else docs[-1 if len(docs) == 1 else 1]
            content = {"label": "SUPPORTED" if len(docs) == 1 else "NOT_ENOUGH_EVIDENCE",
                       "evidence": [{"document_id": doc["document_id"],
                                     "excerpt": doc["text"][:20]}],
                       "explanation": explanation, "needs_review": False,
                       "review_reason": ""}
        return {"message": {"content": json.dumps(content)}, "done_reason": "stop",
                "prompt_eval_count": 500, "eval_count": 50}

    monkeypatch.setattr(ollama, "list", lambda: {"models": [
        {"model": "llama3.2:3b", "digest": "d"}, {"model": "judge:test", "digest": "j"}]})
    monkeypatch.setattr(ollama, "chat", fake_chat)
    return prompts


CASES = {
    "distractor_passages": {"other": "The Corran Bridge opened in 1928."},
    "cases": [{
        "case_id": "c1", "category": "explicitly_supported", "question": "When?",
        "generated_answer": "The Halvorsen Bridge opened in 1931.",
        "relevant_passages": ["The Halvorsen Bridge opened in 1931."],
        "distractors": ["other"],
    }],
}
EXPECTED = {"cases": {"c1": {"response_behavior": "no_abstention", "facts": [
    {"fact_id": "opened_1931", "must_contain": ["Halvorsen", "1931"], "label": "SUPPORTED"},
    {"fact_id": "missing_fact", "must_contain": ["engineer"], "label": "SUPPORTED"},
]}}}


def write_inputs(tmp_path):
    cases = tmp_path / "cases.json"
    expected = tmp_path / "expected.json"
    cases.write_text(json.dumps(CASES), encoding="utf-8")
    expected.write_text(json.dumps(EXPECTED), encoding="utf-8")
    return cases, expected


def test_expected_labels_never_reach_prompts(tmp_path, monkeypatch):
    prompts = fake_ollama(monkeypatch)
    cases, expected = write_inputs(tmp_path)

    diag.run_diagnostics(cases_path=cases, expected_path=expected, allow_unfrozen=True,
                         output_root=tmp_path / "out")

    assert prompts
    for prompt in prompts:
        assert "must_contain" not in prompt and "fact_id" not in prompt
        assert "explicitly_supported" not in prompt


def test_scoring_reports_omissions_label_changes_and_irrelevant_citations(tmp_path, monkeypatch):
    fake_ollama(monkeypatch, cite_distractor=True)
    cases, expected = write_inputs(tmp_path)

    run = diag.run_diagnostics(cases_path=cases, expected_path=expected, allow_unfrozen=True,
                               output_root=tmp_path / "out")
    scored = json.loads((run / "scored_cases.json").read_text(encoding="utf-8"))[0]

    assert scored["omissions"] == ["missing_fact"]
    assert scored["label_changes"] == [{
        "claim": "The Halvorsen Bridge opened in 1931.",
        "relevant_only": "SUPPORTED", "with_distractors": "NOT_ENOUGH_EVIDENCE",
    }]
    assert [c["condition"] for c in scored["irrelevant_citations"]] == ["with_distractors"]
    assert "NOT an unbiased accuracy benchmark" in (run / "report.md").read_text(encoding="utf-8")


def test_request_inspection_confirms_schema_prompt_and_passages(tmp_path, monkeypatch):
    fake_ollama(monkeypatch)
    cases, expected = write_inputs(tmp_path)

    run = diag.run_diagnostics(cases_path=cases, expected_path=expected, allow_unfrozen=True,
                               output_root=tmp_path / "out")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    requests = [json.loads(line) for line in
                (run / "requests.jsonl").read_text(encoding="utf-8").splitlines()]

    assert manifest["request_issues"] == []
    assert requests[0]["request"]["format"] == claim_evaluation.EXTRACTION_SCHEMA
    assert requests[1]["request"]["format"] == claim_evaluation.JUDGMENT_SCHEMA
    assert ollama.chat.__name__ == "fake_chat"  # recorder restored the client


def test_request_inspection_flags_mid_sentence_explanation(tmp_path, monkeypatch):
    recorder_records = [{
        "case_id": "c", "stage": "judgment", "condition": "relevant_only",
        "intended_documents": [],
        "request": {"model": "m", "stream": False, "options": {"num_ctx": 8192},
                    "format": claim_evaluation.JUDGMENT_SCHEMA,
                    "messages": [{"content": claim_evaluation.JUDGMENT_SYSTEM_PROMPT},
                                 {"content": "x\n" + json.dumps(
                                     {"claim": "c", "retrieved_documents": []})}]},
        "response_content": json.dumps({"explanation": "It confirms the date, but the"}),
        "done_reason": "stop", "prompt_eval_count": 100, "eval_count": 10,
    }]

    stats, issues = diag.check_requests(recorder_records)

    assert stats["mid_sentence_explanations"] == 1
    assert any("mid-sentence" in issue for issue in issues)


def test_request_inspection_parses_retry_prompts_with_appended_note():
    docs = [{"document_id": "d1", "text": "Passage."}]
    record = {
        "case_id": "c", "stage": "judgment", "condition": "relevant_only",
        "intended_documents": docs,
        "request": {"model": "m", "stream": False, "options": {"num_ctx": 8192},
                    "format": claim_evaluation.JUDGMENT_SCHEMA,
                    "messages": [{"content": claim_evaluation.JUDGMENT_SYSTEM_PROMPT},
                                 {"content": "x\n" + json.dumps(
                                     {"claim": "c", "retrieved_documents": docs})
                                  + "\n\nThe previous output failed validation."}]},
        "response_content": json.dumps({"explanation": "Fine."}),
        "done_reason": "stop", "prompt_eval_count": 100, "eval_count": 10,
    }

    _, issues = diag.check_requests([record])

    assert issues == []


def test_frozen_diagnostic_set_is_unchanged_and_edits_are_refused(tmp_path):
    frozen = diag.check_frozen(diag.CASES_PATH, diag.EXPECTED_PATH)
    assert frozen["matches_frozen_set"] is True

    edited = tmp_path / "expected.json"
    data = json.loads(diag.EXPECTED_PATH.read_text(encoding="utf-8"))
    data["cases"]["explicit_supported"]["facts"][0]["label"] = "CONTRADICTED"
    edited.write_text(json.dumps(data), encoding="utf-8")
    try:
        diag.check_frozen(diag.CASES_PATH, edited)
    except ValueError as error:
        assert "frozen" in str(error)
    else:
        raise AssertionError("an edited expected-label file must be refused")


def test_frozen_set_never_leaks_expected_labels_and_keeps_failed_cases(tmp_path, monkeypatch):
    # The mocked extractor echoes each answer as a claim, so refusal cases fail
    # validation; they must stay in the results rather than disappear.
    prompts = fake_ollama(monkeypatch)
    run = diag.run_diagnostics(output_root=tmp_path / "out")
    expected = json.loads(diag.EXPECTED_PATH.read_text(encoding="utf-8"))
    fact_ids = {f["fact_id"] for c in expected["cases"].values() for f in c["facts"]}
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    scored = json.loads((run / "scored_cases.json").read_text(encoding="utf-8"))

    for prompt in prompts:
        assert "must_contain" not in prompt
        assert not any(fact_id in prompt for fact_id in fact_ids)
    cases = json.loads(diag.CASES_PATH.read_text(encoding="utf-8"))["cases"]
    assert len(scored) == len(cases)
    assert manifest["summary"]["extraction_failures"]["count"] >= 1
    assert manifest["summary"]["behavior"]["total"] == len(cases)
    assert manifest["diagnostic_set"]["matches_frozen_set"] is True


def test_prompts_share_no_entities_or_years_with_diagnostic_cases():
    import re
    cases = diag.CASES_PATH.read_text(encoding="utf-8")
    payload = json.loads(cases)
    text = json.dumps([payload["cases"], payload["distractor_passages"]], ensure_ascii=False)
    common = {"The", "I", "A", "In", "It", "He", "She", "Could", "Six", "That", "Dr",
              "Who", "When", "Where", "Did", "How", "Its", "His", "Her", "During", "As"}
    tokens = set(re.findall(r"\b[A-Z][a-z]+\b|\b\d{3,}\b", text)) - common
    for prompt in (claim_evaluation.EXTRACTION_SYSTEM_PROMPT,
                   claim_evaluation.JUDGMENT_SYSTEM_PROMPT):
        assert sorted(t for t in tokens if re.search(rf"\b{t}\b", prompt)) == []
    for benchmark_term in ("Marin", "Presque", "Fairview", "north bank", "He constructed it"):
        assert benchmark_term not in claim_evaluation.JUDGMENT_SYSTEM_PROMPT


def test_extraction_and_judgment_models_are_configured_independently(tmp_path, monkeypatch):
    prompts = fake_ollama(monkeypatch)
    cases, expected = write_inputs(tmp_path)

    run = diag.run_diagnostics(cases_path=cases, expected_path=expected, allow_unfrozen=True,
                               output_root=tmp_path / "out", judgment_model="judge:test",
                               judgment_settings={"num_ctx": 4096})
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))

    assert {m for m, is_extraction in prompts.models if is_extraction} == {"llama3.2:3b"}
    assert {m for m, is_extraction in prompts.models if not is_extraction} == {"judge:test"}
    assert manifest["stages"]["judgment"]["options"]["num_ctx"] == 4096
    assert manifest["stages"]["extraction"]["options"]["num_ctx"] == 8192
    assert manifest["request_issues"] == []


def test_reused_extraction_holds_claims_fixed_and_comparison_reports_denominators(
    tmp_path, monkeypatch
):
    prompts = fake_ollama(monkeypatch)
    cases, expected = write_inputs(tmp_path)
    monkeypatch.setattr(diag, "FROZEN_CONTENT_SHA256", {
        "cases": diag.content_sha256(cases), "expected": diag.content_sha256(expected)})
    first = diag.run_diagnostics(cases_path=cases, expected_path=expected,
                                 output_root=tmp_path / "out")
    calls_before = len(prompts)
    second = diag.run_diagnostics(cases_path=cases, expected_path=expected,
                                  output_root=tmp_path / "out", judgment_model="judge:test",
                                  reuse_extraction=first)

    new_models = prompts.models[calls_before:]
    assert new_models and all(not is_extraction for _, is_extraction in new_models)
    first_claims = json.loads((first / "scored_cases.json").read_text(encoding="utf-8"))[0]["claims"]
    second_claims = json.loads((second / "scored_cases.json").read_text(encoding="utf-8"))[0]["claims"]
    assert first_claims == second_claims

    table = diag.compare([first, second], output_root=tmp_path / "out").read_text(encoding="utf-8")
    assert "Comparable configuration (frozen cases/labels, prompts, schemas, options, conditions): **yes**" in table
    assert "| **Correct labels / labeled judgments** | 1/2 (50%) | 1/2 (50%) |" in table
    assert "CONTRADICTED correct | 0/0 | 0/0 |" in table
    assert "NOT an unbiased accuracy benchmark" in table


def test_comparison_flags_runs_with_different_settings(tmp_path, monkeypatch):
    fake_ollama(monkeypatch)
    cases, expected = write_inputs(tmp_path)
    kwargs = dict(cases_path=cases, expected_path=expected, allow_unfrozen=True,
                  output_root=tmp_path / "out")
    first = diag.run_diagnostics(**kwargs)
    second = diag.run_diagnostics(**kwargs, judgment_settings={"num_ctx": 4096})

    table = diag.compare([first, second], output_root=tmp_path / "out").read_text(encoding="utf-8")

    assert "**NO** — differs in options" in table


def test_failed_judgments_are_unassessed_for_citations_not_counted_as_clean():
    claim = "The Halvorsen Bridge opened in 1931."
    judgment = lambda label, error=None: {"claim_index": 0, "claim": claim, "label": label,
                                          "evidence": [], "explanation": "", "attempts": [{}],
                                          "error": error, "needs_review": False}
    record = {
        "case_id": "c1", "category": "explicitly_supported", "generated_answer": claim,
        "distractor_ids": ["x"], "extraction_error": None,
        "extraction": {"claims": [{"claim": claim, "source_excerpt": claim}],
                       "response_behavior": "no_abstention", "attempts": [{}]},
        "judgments": {"relevant_only": [judgment("SUPPORTED")],
                      "with_distractors": [judgment(None, "done_reason=length")]},
    }
    expected = {"cases": {"c1": EXPECTED["cases"]["c1"]}}

    summary = diag.summarize(diag.score([record], expected), [record])

    citations = summary["irrelevant_citations"]
    assert citations["valid_with_distractor_judgments"] == 0
    assert citations["failed_with_distractor_judgments_unassessed"] == 1
    assert summary["labeled_judgments"] == {"correct": 1, "total": 2, "failed": 1}


def test_comparison_names_the_corrected_citation_metric(tmp_path, monkeypatch):
    fake_ollama(monkeypatch)
    cases, expected = write_inputs(tmp_path)
    run = diag.run_diagnostics(cases_path=cases, expected_path=expected, allow_unfrozen=True,
                               output_root=tmp_path / "out")

    table = diag.compare([run], output_root=tmp_path / "out").read_text(encoding="utf-8")
    report = (run / "report.md").read_text(encoding="utf-8")

    assert f"Citation metric: `{diag.CITATION_METRIC_VERSION}` (corrected)" in table
    assert "previous denominator" in table
    assert "Failed with-distractor judgments (citations unassessed)" in table
    assert diag.CITATION_METRIC_VERSION in report
