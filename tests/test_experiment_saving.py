import hashlib
import json

import pandas as pd
import pytest

from src import experiments


REQUIRED_RECORD_FIELDS = {
    "run_id",
    "created_at_utc",
    "question_id",
    "question",
    "configuration",
    "retrieval_method",
    "k",
    "retrieved_document_ids",
    "retrieved_passage_texts",
    "generated_answer",
    "reference_answer",
    "is_answerable",
    "prompt_version",
    "system_prompt",
    "user_prompt",
    "model",
    "model_digest",
    "generation_settings",
    "latency_seconds",
    "done_reason",
}


class FakeRetriever:
    def __init__(self, method):
        self.method = method
        self.calls = []

    def search(self, question, top_k):
        self.calls.append((question, top_k))
        return [
            {
                "document_id": f"{self.method}-doc-{i}",
                "text": f"{self.method} passage {i}",
            }
            for i in range(1, top_k + 1)
        ]


def make_input_files(tmp_path, two_questions=False):
    questions_path = tmp_path / "questions.csv"
    documents_path = tmp_path / "documents.csv"
    rows = [
        {
            "question_id": "q-answerable",
            "question": "What color is the sky?",
            "gold_answer": '["blue", "Blue"]',
            "is_answerable": True,
            "gold_context_id": "gold-1",
        }
    ]
    if two_questions:
        rows.append(
            {
                "question_id": "q-unanswerable",
                "question": "What color is the moon?",
                "gold_answer": "[]",
                "is_answerable": False,
                "gold_context_id": "gold-2",
            }
        )
    pd.DataFrame(rows).to_csv(questions_path, index=False)
    pd.DataFrame(
        {"document_id": ["doc-1"], "text": ["The sky is blue."]}
    ).to_csv(documents_path, index=False)
    return questions_path, documents_path, pd.read_csv(questions_path)


def install_fake_generation(monkeypatch, *, fail_on_call=None, answer="  Réponse.\n"):
    calls = []

    def fake_generate(system_prompt, user_prompt, model, generation_settings):
        calls.append((system_prompt, user_prompt, model, generation_settings))
        if fail_on_call == len(calls):
            raise RuntimeError("synthetic generation failure")
        return {
            "generated_answer": answer,
            "model": model,
            "model_digest": "digest-full",
            "generation_settings": {
                "temperature": 0,
                "seed": 42,
                "num_predict": 256,
                "num_ctx": 8192,
            },
            "latency_seconds": 0.25,
            "done_reason": "stop",
        }

    monkeypatch.setattr(experiments, "generate_answer", fake_generate)
    return calls


def run_saved(tmp_path, monkeypatch, *, two_questions=False, output_path=None):
    questions_path, documents_path, loaded_questions = make_input_files(
        tmp_path, two_questions=two_questions
    )
    install_fake_generation(monkeypatch)
    records, jsonl_path, manifest_path = experiments.run_and_save_experiments(
        loaded_questions,
        bm25_retriever=FakeRetriever("bm25"),
        dense_retriever=FakeRetriever("dense"),
        questions_path=questions_path,
        documents_path=documents_path,
        output_path=output_path or (tmp_path / "run.jsonl"),
    )
    return records, jsonl_path, manifest_path, questions_path, documents_path


def test_jsonl_has_required_fields_four_unique_records_and_exact_answer_text(
    tmp_path, monkeypatch
):
    records, jsonl_path, _, _, _ = run_saved(tmp_path, monkeypatch)
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    loaded = [json.loads(line) for line in lines]

    assert len(records) == len(loaded) == 4
    assert all(set(REQUIRED_RECORD_FIELDS) <= set(record) for record in loaded)
    assert len({(r["question_id"], r["configuration"]) for r in loaded}) == 4
    assert len({r["run_id"] for r in loaded}) == 1
    assert all(r["generated_answer"] == "  Réponse.\n" for r in loaded)
    no_retrieval = next(r for r in loaded if r["configuration"] == "no_retrieval")
    assert no_retrieval["retrieval_method"] == "none"
    assert no_retrieval["k"] == 0
    assert no_retrieval["retrieved_document_ids"] == []
    assert no_retrieval["retrieved_passage_texts"] == []


def test_reference_answers_are_lists_including_empty_unanswerable_answers(
    tmp_path, monkeypatch
):
    _, jsonl_path, _, _, _ = run_saved(
        tmp_path, monkeypatch, two_questions=True
    )
    records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]

    answerable = [r for r in records if r["question_id"] == "q-answerable"]
    unanswerable = [r for r in records if r["question_id"] == "q-unanswerable"]
    assert all(r["reference_answer"] == ["blue", "Blue"] for r in answerable)
    assert all(r["is_answerable"] is True for r in answerable)
    assert all(r["reference_answer"] == [] for r in unanswerable)
    assert all(r["is_answerable"] is False for r in unanswerable)


def test_manifest_hashes_counts_and_status_reflect_success(tmp_path, monkeypatch):
    _, jsonl_path, manifest_path, questions_path, documents_path = run_saved(
        tmp_path, monkeypatch
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["run_status"] == "completed"
    assert manifest["completed_record_count"] == 4
    assert manifest["requested_configurations"] == list(experiments.CONFIGURATIONS)
    assert manifest["question_count"] == 1
    assert manifest["selected_question_ids"] == ["q-answerable"]
    assert manifest["questions_sha256"] == hashlib.sha256(
        questions_path.read_bytes()
    ).hexdigest()
    assert manifest["documents_sha256"] == hashlib.sha256(
        documents_path.read_bytes()
    ).hexdigest()
    assert manifest["model_digest"] == "digest-full"
    assert manifest["generation_settings"]["seed"] == 42
    assert manifest["started_at_utc"]
    assert manifest["ended_at_utc"]
    assert manifest["python_version"]
    assert manifest["package_versions"]
    assert jsonl_path.is_file()


def test_completed_records_survive_later_generation_failure_and_manifest_fails(
    tmp_path, monkeypatch
):
    questions_path, documents_path, questions = make_input_files(tmp_path)
    install_fake_generation(monkeypatch, fail_on_call=3)
    output_path = tmp_path / "partial.jsonl"

    with pytest.raises(RuntimeError, match="synthetic generation failure"):
        experiments.run_and_save_experiments(
            questions,
            bm25_retriever=FakeRetriever("bm25"),
            dense_retriever=FakeRetriever("dense"),
            questions_path=questions_path,
            documents_path=documents_path,
            output_path=output_path,
        )

    jsonl_records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(output_path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert len(jsonl_records) == 2
    assert manifest["run_status"] == "failed"
    assert manifest["completed_record_count"] == 2
    assert "synthetic generation failure" in manifest["error"]


def test_existing_jsonl_or_manifest_is_never_overwritten(tmp_path, monkeypatch):
    questions_path, documents_path, questions = make_input_files(tmp_path)
    install_fake_generation(monkeypatch)
    output_path = tmp_path / "existing.jsonl"
    output_path.write_text("keep this\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        experiments.run_and_save_experiments(
            questions,
            questions_path=questions_path,
            documents_path=documents_path,
            output_path=output_path,
        )

    assert output_path.read_text(encoding="utf-8") == "keep this\n"

    output_path.unlink()
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text("keep manifest\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        experiments.run_and_save_experiments(
            questions,
            questions_path=questions_path,
            documents_path=documents_path,
            output_path=output_path,
        )
    assert manifest_path.read_text(encoding="utf-8") == "keep manifest\n"
    assert not output_path.exists()


def test_run_rejects_misaligned_retrieved_ids_and_texts(tmp_path, monkeypatch):
    questions_path, documents_path, questions = make_input_files(tmp_path)
    install_fake_generation(monkeypatch)

    class MismatchedRetriever:
        def search(self, question, top_k):
            return [{"document_id": "doc-1", "text": "text"}, {"document_id": "doc-2"}]

    with pytest.raises(RuntimeError, match="document-ID and passage-text counts differ"):
        experiments.run_and_save_experiments(
            questions,
            bm25_retriever=MismatchedRetriever(),
            questions_path=questions_path,
            documents_path=documents_path,
            output_path=tmp_path / "mismatch.jsonl",
            config="bm25_basic",
        )
