"""evaluation module for the RAG reliability benchmark."""
import json
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
            "retrieved_ids": [
                result["document_id"] for result in results
            ],
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

    # Build the detailed K=1, 3, 5 results required by step 13.
    retrieval_records = []

    for method in retrievers:
        details = pd.read_csv(
            results_dir / f"{method}_retrieval_details.csv",
            converters={"retrieved_ids": __import__("ast").literal_eval},
        )

        for _, row in details.iterrows():
            top_five_ids = row["retrieved_ids"]

            for k in (1, 3, 5):
                retrieved_ids = top_five_ids[:k]
                gold_id = row["gold_context_id"]

                gold_rank = (
                    retrieved_ids.index(gold_id) + 1
                    if gold_id in retrieved_ids
                    else None
                )

                retrieval_records.append({
                    "question_id": row["question_id"],
                    "method": method,
                    "k": k,
                    "retrieved_ids": json.dumps(retrieved_ids),
                    "gold_rank": gold_rank,
                    "gold_found": gold_rank is not None,
                })

    retrieval_results = pd.DataFrame(retrieval_records)

    assert len(retrieval_results) == 100 * 2 * 3

    retrieval_results.to_csv(
        results_dir / "retrieval_results.csv",
        index=False,
    )

    recall_table = (
        retrieval_results
        .groupby(["method", "k"])["gold_found"]
        .mean()
        .unstack("k")
    )

    recall_table.columns = [
        f"Source Hit@{k}" for k in recall_table.columns
    ]

    recall_table.to_csv(results_dir / "retrieval_recall_table.csv")

    print("\nSource retrieval comparison at K=1, 3, 5:")
    print(recall_table.to_string(float_format=lambda x: f"{x:.1%}"))

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
