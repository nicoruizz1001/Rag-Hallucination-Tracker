from pathlib import Path

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    def __init__(
        self,
        documents_path,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.documents = pd.read_csv(documents_path)

        assert not self.documents.empty
        assert self.documents["document_id"].is_unique
        assert self.documents["text"].notna().all()

        self.model = SentenceTransformer(model_name, device="cpu")

        # Convert each passage into a normalized vector.
        embeddings = self.model.encode(
            self.documents["text"].tolist(),
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        # Inner product on normalized vectors gives cosine similarity.
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, question, top_k=5):
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        if not question.strip():
            raise ValueError("Question cannot be empty")

        query_embedding = self.model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
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