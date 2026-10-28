"""Software tests for the human-review workflow (mocked evaluator; no model)."""

import csv
import json
import re

import pytest

from src import claim_evaluation, human_review

CONFIGS = ("no_retrieval", "bm25_basic", "dense_basic", "dense_abstain")


def write_generations(path, questions=20):
    rows = []
    for q in range(questions):
        for config in CONFIGS:
            rows.append({
                "run_id": "gen", "question_id": f"q{q:02d}", "question": f"Question {q}?",
                "configuration": config,
                "retrieval_method": "none" if config == "no_retrieval" else "bm25",
                "is_answerable": q % 2 == 0,
                "retrieved_document_ids": [] if config == "no_retrieval" else ["d1", "d2"],
                "retrieved_passage_texts": [] if config == "no_retrieval" else
                [f"Passage one about topic {q}.", f"Passage two about topic {q}."],
                "generated_answer": f"According to [Doc 1], fact {q} holds.",
                "done_reason": "stop",
            })
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def inspected_source(tmp_path, question_ids):
    run = tmp_path / "claim_evaluation" / "run1"
    run.mkdir(parents=True)
    (run / "answers.jsonl").write_text(
        "".join(json.dumps({"question_id": q, "response_behavior": "no_abstention"}) + "\n"
                for q in question_ids), encoding="utf-8")
    return [tmp_path / "claim_evaluation"]


def draw(tmp_path, seed=42, per_stratum=2, name="out"):
    generations = tmp_path / "generations.jsonl"
    if not generations.exists():
        write_generations(generations)
    sources = inspected_source(tmp_path, ["q00", "q01"]) if not (tmp_path / "claim_evaluation").exists() \
        else [tmp_path / "claim_evaluation"]
    return human_review.sample(generations_path=generations, seed=seed, per_stratum=per_stratum,
                               output_root=tmp_path / name, inspected_sources=sources)


def manifest_of(review_dir):
    return json.loads((review_dir / "coordinator" / "sample_manifest.json").read_text(encoding="utf-8"))


def test_sampling_is_reproducible_stratified_and_excludes_inspected_questions(tmp_path):
    first = manifest_of(draw(tmp_path, name="a"))
    second = manifest_of(draw(tmp_path, name="b"))
    other_seed = manifest_of(draw(tmp_path, seed=7, name="c"))

    assert first["items"] == second["items"]
    assert first["items"] != other_seed["items"]
    qids = [item["question_id"] for item in first["items"]]
    assert len(qids) == len(set(qids)) == 12  # 3 configurations x 2 answerability x 2
    assert not {"q00", "q01"} & set(qids)
    assert set(first["excluded_question_ids"]) == {"q00", "q01"}
    assert {item["configuration"] for item in first["items"]} == set(human_review.RAG_CONFIGURATIONS)
    assert all(s["selected"] == 2 for s in first["strata"])
    assert "random.Random(42)" in first["procedure"]


def evaluate_sample(tmp_path, review_dir, fail_question=None):
    def fake_generate(system, user, **kwargs):
        data = json.loads(user.split("\n", 1)[1])
        if "claim extractor" in system.lower():
            answer = data["generated_answer"]
            claim = answer.split(", ", 1)[1]
            return {"generated_answer": json.dumps({"response_behavior": "no_abstention",
                                                    "claims": [{"claim": claim, "source_excerpt": claim}]}),
                    "model": kwargs["model"], "model_digest": "x", "latency_seconds": 0.0,
                    "generation_settings": kwargs["generation_settings"], "done_reason": "stop"}
        if fail_question and fail_question in data["claim"]:
            content = "{not json"
        else:
            doc = data["retrieved_documents"][0]
            content = json.dumps({"label": "SUPPORTED", "evidence": [
                {"document_id": doc["document_id"], "excerpt": "SECRET-EXCERPT"[:0] + doc["text"][:7]}],
                "explanation": "HIDDEN EXPLANATION TEXT.", "needs_review": True,
                "review_reason": "HIDDEN REVIEW FLAG"})
        return {"generated_answer": content, "model": kwargs["model"], "model_digest": "x",
                "latency_seconds": 0.0, "generation_settings": kwargs["generation_settings"],
                "done_reason": "stop"}

    import pytest as _pytest
    patch = _pytest.MonkeyPatch()
    patch.setattr(claim_evaluation, "generate_answer", fake_generate)
    try:
        return claim_evaluation.evaluate_file(
            review_dir / "coordinator" / "sample_input.jsonl",
            output_root=review_dir / "coordinator" / "evaluation", limit=1000,
            extraction_model="llama3.2:3b", judgment_model="qwen2.5:7b-instruct")
    finally:
        patch.undo()


def test_export_blinds_reviewer_files_and_keeps_failures(tmp_path):
    review_dir = draw(tmp_path)
    items = manifest_of(review_dir)["items"]
    failing = items[0]["question_id"].lstrip("q").lstrip("0") or "0"
    run = evaluate_sample(tmp_path, review_dir, fail_question=f"fact {failing} ")

    reviewer, claim_total = human_review.export(review_dir, run)

    text = "".join(p.read_text(encoding="utf-8-sig") for p in reviewer.iterdir())
    for hidden in ("HIDDEN EXPLANATION", "HIDDEN REVIEW FLAG", "qwen", "llama",
                   "bm25", "dense", "no_retrieval", "needs_review", "d1", "d2"):
        assert hidden not in text
    assert not re.search(r"q\d\d", text)  # no source question IDs
    assert "SUPPORTED, CONTRADICTED, or NOT_ENOUGH_EVIDENCE" in text  # allowed-value help only
    claims_rows = list(csv.DictReader((reviewer / "claims.csv").open(encoding="utf-8-sig")))
    assert claim_total == len(claims_rows) == len(items)
    assert all(row["human_support_label"] == "" for row in claims_rows)
    mapping = json.loads((review_dir / "coordinator" / "mapping.json").read_text(encoding="utf-8"))
    labels = [c["automatic_label"] for item in mapping["items"] for c in item["claims"]]
    assert labels.count(None) == 1 and labels.count("SUPPORTED") == len(items) - 1
    assert mapping["judgment_model"] == "qwen2.5:7b-instruct"
    assert {item["configuration"] for item in mapping["items"]} == set(human_review.RAG_CONFIGURATIONS)
    packet = (reviewer / "review_packet.md").read_text(encoding="utf-8")
    assert "Passage one about topic" in packet and "**Doc 2:**" in packet
    assert "HIDDEN EXPLANATION" not in packet and "SUPPORTED" not in packet


def test_export_refuses_an_evaluation_of_a_different_input(tmp_path):
    review_dir = draw(tmp_path)
    other = tmp_path / "other.jsonl"
    other.write_text((review_dir / "coordinator" / "sample_input.jsonl")
                     .read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    patch = pytest.MonkeyPatch()
    patch.setattr(claim_evaluation, "generate_answer", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("no model calls expected")))
    try:
        run = claim_evaluation.evaluate_file(other, output_root=tmp_path / "eval")
    finally:
        patch.undo()

    with pytest.raises(ValueError, match="not produced from this sample"):
        human_review.export(review_dir, run)


def fill(path, updates, key):
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    for row in rows:
        row.update(updates.get(row[key], {}))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_compare_reports_partial_agreement_and_rejects_invalid_labels(tmp_path):
    review_dir = draw(tmp_path)
    run = evaluate_sample(tmp_path, review_dir)
    reviewer, _ = human_review.export(review_dir, run)
    claim_ids = [r["claim_id"] for r in csv.DictReader((reviewer / "claims.csv").open(encoding="utf-8-sig"))]
    fill(reviewer / "claims.csv", {claim_ids[0]: {"human_support_label": "SUPPORTED"},
                                   claim_ids[1]: {"human_support_label": "NOT_ENOUGH_EVIDENCE",
                                                  "human_claim_distorted": "yes"}}, "claim_id")
    fill(reviewer / "answers.csv", {"R001": {"human_abstention_behavior": "no_abstention",
                                             "human_extraction_omissions": "Dropped a date."}},
         "item_id")

    _, report = human_review.compare(review_dir)

    assert report["status"] == "partial"
    assert report["support_agreement"] == {"agree": 1, "labelled": 2}
    assert report["confusion_human_by_automatic"]["NOT_ENOUGH_EVIDENCE"] == {"SUPPORTED": 1}
    assert report["claims_unlabelled"] == len(claim_ids) - 2
    assert report["abstention_agreement"] == 1 and report["items_with_reported_omissions"] == 1
    assert report["claims_marked_distorted"] == 1
    assert "not by itself complete human validation" in report["note"]

    fill(reviewer / "claims.csv", {claim_ids[2]: {"human_support_label": "TRUE"}}, "claim_id")
    with pytest.raises(ValueError, match="invalid support label"):
        human_review.compare(review_dir)
