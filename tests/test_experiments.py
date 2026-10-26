import pandas as pd
import pytest

from src import experiments, rag


GENERATION_SETTINGS = {
    "temperature": 0,
    "seed": 42,
    "num_predict": 256,
    "num_ctx": 8192,
}


class FakeRetriever:
    def __init__(self, name, documents):
        self.name = name
        self.documents = documents
        self.calls = []

    def search(self, question, top_k):
        self.calls.append((question, top_k))
        return self.documents[:top_k]


def question_frame():
    return pd.DataFrame(
        [
            {
                "question_id": "q-1",
                "question": "What color is the object?",
                "gold_answer": "secret reference answer",
                "is_answerable": True,
                "gold_context_id": "gold-doc",
            }
        ]
    )


def setup_generation(monkeypatch):
    calls = []

    def fake_generate(system_prompt, user_prompt, model, generation_settings):
        calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "model": model,
                "generation_settings": generation_settings,
            }
        )
        return {
            "generated_answer": f"answer-{len(calls)}",
            "model": model,
            "model_digest": "same-digest",
            "generation_settings": GENERATION_SETTINGS.copy(),
            "latency_seconds": 0.1,
            "done_reason": "stop",
        }

    monkeypatch.setattr(experiments, "generate_answer", fake_generate)
    return calls


def test_basic_retrieval_configurations_share_prompt_instructions(monkeypatch):
    calls = setup_generation(monkeypatch)
    bm25 = FakeRetriever("bm25", [{"document_id": "b1", "text": "blue"}])
    dense = FakeRetriever("dense", [{"document_id": "d1", "text": "blue"}])

    records = experiments.run_experiments(
        question_frame(), bm25, dense, config="all", limit=1
    )

    basic_system_prompts = [
        record["system_prompt"]
        for record in records
        if record["configuration"] in {"bm25_basic", "dense_basic"}
    ]
    assert basic_system_prompts == [rag.EVIDENCE_BASIC_SYSTEM_PROMPT] * 2
    assert basic_system_prompts[0] == basic_system_prompts[1]
    assert "Insufficient information" not in basic_system_prompts[0]
    assert calls[1]["system_prompt"] == calls[2]["system_prompt"]


def test_dense_configs_reuse_same_evidence_and_only_abstain_adds_instruction(
    monkeypatch,
):
    setup_generation(monkeypatch)
    documents = [
        {"document_id": f"d{number}", "text": f"passage {number}"}
        for number in range(1, 5)
    ]
    dense = FakeRetriever("dense", documents)

    records = experiments.run_experiments(
        question_frame(),
        FakeRetriever("bm25", []),
        dense,
        config="all",
        limit=1,
    )
    basic = next(r for r in records if r["configuration"] == "dense_basic")
    abstain = next(r for r in records if r["configuration"] == "dense_abstain")

    assert dense.calls == [("What color is the object?", 3)]
    assert basic["retrieved_document_ids"] == abstain["retrieved_document_ids"] == [
        "d1", "d2", "d3"
    ]
    assert basic["retrieved_passage_texts"] == abstain["retrieved_passage_texts"] == [
        "passage 1", "passage 2", "passage 3"
    ]
    assert basic["user_prompt"] == abstain["user_prompt"]
    assert basic["system_prompt"] == rag.EVIDENCE_BASIC_SYSTEM_PROMPT
    assert abstain["system_prompt"] == rag.EVIDENCE_ABSTAIN_SYSTEM_PROMPT
    assert "Insufficient information" not in basic["system_prompt"]
    assert abstain["system_prompt"].endswith(
        "If the evidence does not establish the answer, respond with exactly:\n"
        "Insufficient information"
    )


def test_no_retrieval_never_calls_a_retriever(monkeypatch):
    setup_generation(monkeypatch)
    unused_bm25 = FakeRetriever("bm25", [])
    unused_dense = FakeRetriever("dense", [])

    records = experiments.run_experiments(
        question_frame(), unused_bm25, unused_dense,
        config="no_retrieval", limit=1,
    )

    assert unused_bm25.calls == []
    assert unused_dense.calls == []
    assert records[0]["retrieved_document_ids"] == []
    assert records[0]["retrieved_passage_texts"] == []
    assert records[0]["prompt_version"] == "no_retrieval_v1"
    assert records[0]["system_prompt"] == "Answer the question clearly and concisely."


def test_configurations_use_same_question_model_settings_and_exclude_references(
    monkeypatch,
):
    calls = setup_generation(monkeypatch)
    settings = {"temperature": 0.25}
    records = experiments.run_experiments(
        question_frame(),
        FakeRetriever("bm25", [{"document_id": "b", "text": "blue"}]),
        FakeRetriever("dense", [{"document_id": "d", "text": "blue"}]),
        config="all",
        limit=1,
        model="chosen-model",
        generation_settings=settings,
    )

    assert len(records) == 4
    assert {call["model"] for call in calls} == {"chosen-model"}
    assert all(call["generation_settings"] is settings for call in calls)
    assert {record["question"] for record in records} == {
        "What color is the object?"
    }
    assert {record["model"] for record in records} == {"chosen-model"}
    assert all(record["generation_settings"] == GENERATION_SETTINGS for record in records)
    for record in records:
        for prompt in (record["system_prompt"], record["user_prompt"]):
            assert "secret reference answer" not in prompt
            assert "gold-doc" not in prompt
            assert "is_answerable" not in prompt
            assert "True" not in prompt


def test_inconsistent_model_metadata_stops_experiment(monkeypatch):
    responses = iter(("digest-1", "digest-2"))

    def fake_generate(system_prompt, user_prompt, model, generation_settings):
        return {
            "generated_answer": "answer",
            "model": model,
            "model_digest": next(responses),
            "generation_settings": GENERATION_SETTINGS.copy(),
            "latency_seconds": 0.1,
            "done_reason": "stop",
        }

    monkeypatch.setattr(experiments, "generate_answer", fake_generate)

    with pytest.raises(RuntimeError, match="[Mm]odel digest or generation settings differ"):
        experiments.run_experiments(
            question_frame(),
            FakeRetriever("bm25", []),
            FakeRetriever("dense", []),
            config="all",
            limit=1,
        )


def test_existing_rag_smoke_prompt_keeps_strict_abstention():
    system_prompt, user_prompt = rag.build_rag_prompt(
        "Question?", [{"document_id": "doc-1", "text": "Evidence text."}]
    )

    assert system_prompt == rag.EVIDENCE_ABSTAIN_SYSTEM_PROMPT
    assert "Insufficient information" in system_prompt
    assert "[Doc 1]\nEvidence text." in user_prompt
