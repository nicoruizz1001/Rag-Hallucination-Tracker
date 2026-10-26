"""evaluation module for the RAG reliability benchmark."""
from pathlib import Path

import pandas as pd

from retrieval import BM25Retriever
from dense_retrieval import DenseRetriever


def evaluate_retriever(retriever, questions, method):
    assert questions["gold_context_id"].isin(
        retriever.documents["document_id"]
    ).all()

    records = []

    for _, row in questions.iterrows():
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
            "method": method,
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

    groups = {
        "all": details,
        "answerable": details[details["is_answerable"]],
        "unanswerable": details[~details["is_answerable"]],
    }

    summary_records = []

    for group_name, group in groups.items():
        summary_records.append({
            "method": method,
            "group": group_name,
            "question_count": len(group),
            "source_hit_at_1": group["source_hit_at_1"].mean(),
            "source_hit_at_5": group["source_hit_at_5"].mean(),
            "mrr_at_5": group["reciprocal_rank_at_5"].mean(),
        })

    return details, pd.DataFrame(summary_records)


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    questions = pd.read_csv(data_dir / "questions.csv")

    assert len(questions) == 100
    assert questions["question_id"].is_unique

    labels = questions["is_answerable"].astype(str).str.lower()
    assert labels.isin(["true", "false"]).all()
    questions["is_answerable"] = labels.eq("true")

    documents_path = data_dir / "documents.csv"

    retrievers = {
        "bm25": BM25Retriever(documents_path),
        "dense": DenseRetriever(documents_path),
    }

    summaries = []

    for method, retriever in retrievers.items():
        print(f"\nEvaluating {method}...")

        details, summary = evaluate_retriever(
            retriever,
            questions,
            method,
        )

        details.to_csv(
            results_dir / f"{method}_retrieval_details.csv",
            index=False,
        )

        summary.to_csv(
            results_dir / f"{method}_retrieval_summary.csv",
            index=False,
        )

        summaries.append(summary)

        for _, row in summary.iterrows():
            print(f"\nGroup: {row['group']}")
            print(f"Questions: {row['question_count']}")
            print(f"Source Hit@1: {row['source_hit_at_1']:.1%}")
            print(f"Source Hit@5: {row['source_hit_at_5']:.1%}")
            print(f"MRR@5: {row['mrr_at_5']:.3f}")

    comparison = pd.concat(summaries, ignore_index=True)

    comparison.to_csv(
        results_dir / "retrieval_comparison.csv",
        index=False,
    )

    print("\nOverall comparison:")
    overall = comparison[comparison["group"] == "all"]

    print(
        overall[
            ["method", "source_hit_at_1", "source_hit_at_5", "mrr_at_5"]
        ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

    print("\nSaved results to:", results_dir)


if __name__ == "__main__":
    main()