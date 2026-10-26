"""retrieval module for the RAG reliability benchmark."""

import re
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi


def tokenize(text):
    """Lowercase text and split it into words."""
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    def __init__(self, documents_path):
        self.documents = pd.read_csv(documents_path)

        assert not self.documents.empty
        assert self.documents["document_id"].is_unique
        assert self.documents["text"].notna().all()

        tokenized_documents = [
            tokenize(text)
            for text in self.documents["text"]
        ]

        self.index = BM25Okapi(tokenized_documents)

    def search(self, question, top_k=5):
        """Return the top passages, their IDs, and BM25 scores."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        tokens = tokenize(question)

        if not tokens:
            raise ValueError("Question must contain at least one word")

        scores = self.index.get_scores(tokens)

        # Sort highest score first; use row order to break ties.
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: (-float(scores[i]), i),
        )[:top_k]

        results = []

        for rank, index in enumerate(top_indices, start=1):
            document = self.documents.iloc[index]

            results.append({
                "rank": rank,
                "document_id": document["document_id"],
                "text": document["text"],
                "score": float(scores[index]),
            })

        return results


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"

    retriever = BM25Retriever(data_dir / "documents.csv")
    questions = pd.read_csv(data_dir / "questions.csv")

    # Try the first question from your reproducible sample.
    example = questions.iloc[0]
    results = retriever.search(example["question"], top_k=5)

    assert len(results) == 5
    assert len({r["document_id"] for r in results}) == 5

    print("Question:", example["question"])
    print("Gold context ID:", example["gold_context_id"])

    for result in results:
        is_gold = (
            result["document_id"] == example["gold_context_id"]
        )

        print(f"\nRank: {result['rank']}")
        print(f"BM25 score: {result['score']:.4f}")
        print(f"Matches original source: {is_gold}")
        print("Passage:", result["text"])

    gold_found = any(
        result["document_id"] == example["gold_context_id"]
        for result in results
    )

    print("\nOriginal source found in top 5:", gold_found)


if __name__ == "__main__":
    main()