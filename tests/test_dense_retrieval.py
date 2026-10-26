from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dense_retrieval


class FakeEmbeddingModel:
    instances = []

    def __init__(self, model_name, device, revision):
        self.model_name = model_name
        self.device = device
        self.revision = revision
        self.passage_batches = []
        FakeEmbeddingModel.instances.append(self)

    def encode(self, texts, **kwargs):
        if len(texts) > 1:
            self.passage_batches.append(list(texts))
        vectors = {
            "alpha passage": [1.0, 0.0, 0.0],
            "beta passage": [0.0, 1.0, 0.0],
            "gamma passage": [0.0, 0.0, 1.0],
            "changed passage": [-1.0, 0.0, 0.0],
            "alpha query": [1.0, 0.0, 0.0],
            "beta query": [0.2, 0.98, 0.0],
        }
        return np.asarray([vectors[text] for text in texts], dtype="float32")


@pytest.fixture(autouse=True)
def fake_embedding_model(monkeypatch):
    FakeEmbeddingModel.instances.clear()
    monkeypatch.setattr(dense_retrieval, "SentenceTransformer", FakeEmbeddingModel)


def write_documents(path, document_ids, texts):
    pd.DataFrame({"document_id": document_ids, "text": texts}).to_csv(
        path, index=False
    )


def test_matching_cache_skips_passage_encoding_and_still_encodes_queries(
    tmp_path, capsys
):
    documents_path = tmp_path / "documents.csv"
    write_documents(
        documents_path,
        ["doc-1", "doc-2"],
        ["alpha passage", "beta passage"],
    )

    first = dense_retrieval.DenseRetriever(documents_path)
    first_results = first.search("alpha query", top_k=2)
    first_output = capsys.readouterr().out
    second = dense_retrieval.DenseRetriever(documents_path)
    second_results = second.search("alpha query", top_k=2)
    second_output = capsys.readouterr().out

    assert len(FakeEmbeddingModel.instances) == 2
    assert len(FakeEmbeddingModel.instances[0].passage_batches) == 1
    assert FakeEmbeddingModel.instances[1].passage_batches == []
    assert first_results == second_results
    assert "Built and saved FAISS index" in first_output
    assert "Loaded FAISS index from cache" in second_output
    assert (tmp_path / "cache" / "passage_embeddings.npy").is_file()
    assert (tmp_path / "cache" / "index.faiss").is_file()
    assert (tmp_path / "cache" / "metadata.json").is_file()


@pytest.mark.parametrize(
    ("updated_ids", "updated_texts"),
    [
        (["doc-1", "doc-2"], ["changed passage", "beta passage"]),
        (["doc-2", "doc-1"], ["beta passage", "alpha passage"]),
    ],
)
def test_text_or_order_change_invalidates_cache(
    tmp_path, capsys, updated_ids, updated_texts
):
    documents_path = tmp_path / "documents.csv"
    write_documents(
        documents_path,
        ["doc-1", "doc-2"],
        ["alpha passage", "beta passage"],
    )
    dense_retrieval.DenseRetriever(documents_path)
    capsys.readouterr()

    write_documents(documents_path, updated_ids, updated_texts)
    updated = dense_retrieval.DenseRetriever(documents_path)

    assert FakeEmbeddingModel.instances[-1].passage_batches == [updated_texts]
    assert "Built and saved FAISS index" in capsys.readouterr().out
    assert len(updated.search("beta query", top_k=2)) == 2


def test_fresh_and_cached_retrieval_results_are_equivalent(tmp_path):
    documents_path = tmp_path / "documents.csv"
    write_documents(
        documents_path,
        ["doc-1", "doc-2", "doc-3"],
        ["alpha passage", "beta passage", "gamma passage"],
    )

    fresh = dense_retrieval.DenseRetriever(documents_path)
    fresh_results = fresh.search("beta query", top_k=3)
    cached = dense_retrieval.DenseRetriever(documents_path)

    assert cached.search("beta query", top_k=3) == fresh_results
    assert [result["document_id"] for result in fresh_results] == [
        "doc-2",
        "doc-1",
        "doc-3",
    ]
