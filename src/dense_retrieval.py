import hashlib
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


PASSAGE_BATCH_SIZE = 32
NORMALIZE_EMBEDDINGS = True
CACHE_VERSION = 1


class DenseRetriever:
    def __init__(
        self,
        documents_path,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_revision="main",
        cache_dir=None,
    ):
        documents_path = Path(documents_path)
        self.documents = pd.read_csv(documents_path)

        assert not self.documents.empty
        assert self.documents["document_id"].is_unique
        assert self.documents["text"].notna().all()

        self.model = SentenceTransformer(
            model_name, device="cpu", revision=model_revision
        )

        self.cache_dir = Path(cache_dir) if cache_dir else documents_path.parent / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_paths = {
            "embeddings": self.cache_dir / "passage_embeddings.npy",
            "index": self.cache_dir / "index.faiss",
            "metadata": self.cache_dir / "metadata.json",
        }
        expected_metadata = self._expected_metadata(model_name, model_revision)

        if self._load_cache(expected_metadata):
            print("Loaded FAISS index from cache")
        else:
            self._build_index()
            self._save_cache(expected_metadata)
            print("Built and saved FAISS index")

    def _expected_metadata(self, model_name, model_revision):
        ordered_documents = list(
            zip(
                self.documents["document_id"].tolist(),
                self.documents["text"].tolist(),
            )
        )
        serialized_documents = json.dumps(
            ordered_documents,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        fingerprint = hashlib.sha256(serialized_documents).hexdigest()
        return {
            "cache_version": CACHE_VERSION,
            "model": model_name,
            "revision": model_revision,
            "device": "cpu",
            "embedding_settings": {
                "batch_size": PASSAGE_BATCH_SIZE,
                "normalize_embeddings": NORMALIZE_EMBEDDINGS,
                "convert_to_numpy": True,
                "dtype": "float32",
                "index_metric": "inner_product",
            },
            "document_count": len(self.documents),
            "documents_fingerprint": fingerprint,
        }

    def _load_cache(self, expected_metadata):
        try:
            with self._cache_paths["metadata"].open(encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
            if metadata != expected_metadata:
                return False

            embeddings = np.load(self._cache_paths["embeddings"], allow_pickle=False)
            index = faiss.read_index(str(self._cache_paths["index"]))
            document_count = len(self.documents)
            if (
                embeddings.ndim != 2
                or embeddings.shape[0] != document_count
                or index.ntotal != document_count
                or index.d != embeddings.shape[1]
            ):
                return False

            self.embeddings = embeddings.astype("float32", copy=False)
            self.index = index
            return True
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, json.JSONDecodeError):
            return False

    def _build_index(self):
        # Convert each passage into a normalized vector.
        self.embeddings = self.model.encode(
            self.documents["text"].tolist(),
            batch_size=PASSAGE_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
        ).astype("float32")

        # Inner product on normalized vectors gives cosine similarity.
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def _save_cache(self, metadata):
        temporary_embeddings = self._cache_paths["embeddings"].with_suffix(".npy.tmp")
        with temporary_embeddings.open("wb") as embeddings_file:
            np.save(embeddings_file, self.embeddings)
        temporary_embeddings.replace(self._cache_paths["embeddings"])

        temporary_index = self._cache_paths["index"].with_suffix(".faiss.tmp")
        faiss.write_index(self.index, str(temporary_index))
        temporary_index.replace(self._cache_paths["index"])

        temporary_metadata = self._cache_paths["metadata"].with_suffix(".json.tmp")
        temporary_metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary_metadata.replace(self._cache_paths["metadata"])

    def search(self, question, top_k=5):
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        if not question.strip():
            raise ValueError("Question cannot be empty")

        query_embedding = self.model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
        ).astype("float32")

        k = min(top_k, len(self.documents))
        scores, indices = self.index.search(query_embedding, k)

        results = []

        for rank, (index, score) in enumerate(
            zip(indices[0], scores[0]),
            start=1,
        ):
            document = self.documents.iloc[int(index)]

            results.append({
                "rank": rank,
                "document_id": document["document_id"],
                "text": document["text"],
                "score": float(score),
            })

        return results


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"

    retriever = DenseRetriever(data_dir / "documents.csv")
    questions = pd.read_csv(data_dir / "questions.csv")

    example = questions.iloc[0]
    results = retriever.search(example["question"], top_k=5)

    assert len(results) == 5
    assert len({r["document_id"] for r in results}) == 5

    print("\nQuestion:", example["question"])

    for result in results:
        is_gold = (
            result["document_id"] == example["gold_context_id"]
        )

        print(f"\nRank: {result['rank']}")
        print(f"Cosine similarity: {result['score']:.4f}")
        print(f"Matches original source: {is_gold}")
        print("Passage:", result["text"])

    gold_found = any(
        result["document_id"] == example["gold_context_id"]
        for result in results
    )

    print("\nOriginal source found in top 5:", gold_found)


if __name__ == "__main__":
    main()
