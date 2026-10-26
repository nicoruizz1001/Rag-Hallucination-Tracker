"""evaluation module for the RAG reliability benchmark."""
from pathlib import Path

import pandas as pd

from retrieval import BM25Retriever


def evaluate_bm25():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    questions = pd.read_csv(data_dir / "questions.csv")
    retriever = BM25Retriever(data_dir / "documents.csv")

    assert len(questions) == 100
    assert questions["question_id"].is_unique
    assert questions["gold_context_id"].isin(
        retriever.documents["document_id"]
    ).all()

    records = []

    for _, row in questions.iterrows():
        # Only the question is passed to the retriever.
        results = retriever.search(row["question"], top_k=5)

        gold_rank = next(
            (
                result["rank"]
                for result in results
                if result["document_id"] == row["gold_context_id"]
            ),
            None,
        )

        records.append({
            "question_id": row["question_id"],
            "question": row["question"],
            "is_answerable": row["is_answerable"],
            "gold_context_id": row["gold_context_id"],
            "gold_rank_at_5": gold_rank,
            "source_hit_at_1": int(gold_rank == 1),
            "source_hit_at_5": int(gold_rank is not None),
            "reciprocal_rank_at_5": (
                1.0 / gold_rank if gold_rank is not None else 0.0
            ),
        })

    details = pd.DataFrame(records)

    # Convert labels explicitly, including when CSV loads them as strings.
    labels = details["is_answerable"].astype(str).str.lower()
    assert labels.isin(["true", "false"]).all()
    details["is_answerable"] = labels.eq("true")

    groups = {
        "all": details,
        "answerable": details[details["is_answerable"]],
        "unanswerable": details[~details["is_answerable"]],
    }

    summary_records = []

    for name, group in groups.items():
        summary_records.append({
            "group": name,
            "question_count": len(group),
            "source_hit_at_1": group["source_hit_at_1"].mean(),
            "source_hit_at_5": group["source_hit_at_5"].mean(),
            "mrr_at_5": group["reciprocal_rank_at_5"].mean(),
        })

    summary = pd.DataFrame(summary_records)

    details.to_csv(
        results_dir / "bm25_retrieval_details.csv",
        index=False,
    )
    summary.to_csv(
        results_dir / "bm25_retrieval_summary.csv",
        index=False,
    )

    print("\nBM25 retrieval evaluation")

    for record in summary_records:
        print(f"\nGroup: {record['group']}")
        print(f"Questions: {record['question_count']}")
        print(f"Source Hit@1: {record['source_hit_at_1']:.1%}")
        print(f"Source Hit@5: {record['source_hit_at_5']:.1%}")
        print(f"MRR@5: {record['mrr_at_5']:.3f}")

    print("\nSaved results to:", results_dir)


if __name__ == "__main__":
    evaluate_bm25()