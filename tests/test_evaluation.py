import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import evaluation


class FakeRetriever:
    def __init__(self, documents_path):
        self.documents = pd.read_csv(documents_path)

    def search(self, question, top_k=5):
        document_ids = ["doc-2", "doc-3", "doc-1", "doc-4", "doc-5"]
        return [
            {"document_id": document_id, "rank": rank}
            for rank, document_id in enumerate(document_ids[:top_k], start=1)
        ]


def test_evaluate_retriever_saves_ordered_retrieved_ids(tmp_path):
    documents_path = tmp_path / "documents.csv"
    pd.DataFrame(
        {"document_id": ["doc-1", "doc-2", "doc-3", "doc-4", "doc-5"]}
    ).to_csv(documents_path, index=False)
    questions = pd.DataFrame([
        {
            "question_id": "question-1",
            "question": "Which document is assigned?",
            "is_answerable": True,
            "gold_context_id": "doc-1",
        }
    ])

    details, _ = evaluation.evaluate_retriever(
        FakeRetriever(documents_path), questions, "fake"
    )

    assert details.loc[0, "retrieved_ids"] == [
        "doc-2",
        "doc-3",
        "doc-1",
        "doc-4",
        "doc-5",
    ]


def test_main_writes_k_1_3_5_retrieval_results(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    pd.DataFrame(
        {"document_id": ["doc-1", "doc-2", "doc-3", "doc-4", "doc-5"]}
    ).to_csv(data_dir / "documents.csv", index=False)
    pd.DataFrame([
        {
            "question_id": f"question-{number}",
            "question": f"Question {number}",
            "is_answerable": True,
            "gold_context_id": "doc-1",
        }
        for number in range(1, 101)
    ]).to_csv(data_dir / "questions.csv", index=False)

    monkeypatch.setattr(evaluation, "BM25Retriever", FakeRetriever)
    monkeypatch.setattr(evaluation, "DenseRetriever", FakeRetriever)
    monkeypatch.setattr(
        evaluation,
        "Path",
        lambda _: Path(tmp_path / "src" / "evaluation.py"),
    )

    evaluation.main()

    retrieval_results = pd.read_csv(tmp_path / "results" / "retrieval_results.csv")
    assert len(retrieval_results) == 600
    assert set(retrieval_results["k"]) == {1, 3, 5}

    bm25_first_question = retrieval_results[
        (retrieval_results["method"] == "bm25")
        & (retrieval_results["question_id"] == "question-1")
    ].set_index("k")

    assert json.loads(bm25_first_question.loc[1, "retrieved_ids"]) == ["doc-2"]
    assert json.loads(bm25_first_question.loc[3, "retrieved_ids"]) == [
        "doc-2",
        "doc-3",
        "doc-1",
    ]
    assert pd.isna(bm25_first_question.loc[1, "gold_rank"])
    assert bm25_first_question.loc[3, "gold_rank"] == 3
    assert not bool(bm25_first_question.loc[1, "gold_found"])
    assert bool(bm25_first_question.loc[3, "gold_found"])

    recall_table = pd.read_csv(
        tmp_path / "results" / "retrieval_recall_table.csv",
        index_col="method",
    )
    assert recall_table.loc["bm25", "Source Hit@1"] == 0.0
    assert recall_table.loc["bm25", "Source Hit@3"] == 1.0
    assert recall_table.loc["bm25", "Source Hit@5"] == 1.0
