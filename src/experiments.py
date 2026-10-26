"""Controlled baseline generation configurations."""

import argparse
from pathlib import Path

import pandas as pd

if __package__:
    from .dense_retrieval import DenseRetriever
    from .generation import generate_answer
    from .rag import (
        EVIDENCE_ABSTAIN_PROMPT_VERSION,
        EVIDENCE_BASIC_PROMPT_VERSION,
        NO_RETRIEVAL_PROMPT_VERSION,
        build_rag_prompt,
    )
    from .retrieval import BM25Retriever
else:
    from dense_retrieval import DenseRetriever
    from generation import generate_answer
    from rag import (
        EVIDENCE_ABSTAIN_PROMPT_VERSION,
        EVIDENCE_BASIC_PROMPT_VERSION,
        NO_RETRIEVAL_PROMPT_VERSION,
        build_rag_prompt,
    )
    from retrieval import BM25Retriever


DEFAULT_MODEL = "llama3.2:3b"
RETRIEVAL_K = 3
CONFIGURATIONS = {
    "no_retrieval": {
        "retriever": None,
        "top_k": 0,
        "prompt_version": NO_RETRIEVAL_PROMPT_VERSION,
    },
    "bm25_basic": {
        "retriever": "bm25",
        "top_k": RETRIEVAL_K,
        "prompt_version": EVIDENCE_BASIC_PROMPT_VERSION,
    },
    "dense_basic": {
        "retriever": "dense",
        "top_k": RETRIEVAL_K,
        "prompt_version": EVIDENCE_BASIC_PROMPT_VERSION,
    },
    "dense_abstain": {
        "retriever": "dense",
        "top_k": RETRIEVAL_K,
        "prompt_version": EVIDENCE_ABSTAIN_PROMPT_VERSION,
    },
}


def run_experiments(
    questions,
    bm25_retriever=None,
    dense_retriever=None,
    config="all",
    limit=None,
    model=DEFAULT_MODEL,
    generation_settings=None,
):
    """Run selected controlled configurations for questions in input order."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    if config == "all":
        configuration_names = list(CONFIGURATIONS)
    elif config in CONFIGURATIONS:
        configuration_names = [config]
    else:
        raise ValueError(
            f"Unknown configuration '{config}'. Choose one of: "
            f"{', '.join(CONFIGURATIONS)} or 'all'."
        )

    required_retrievers = {
        CONFIGURATIONS[name]["retriever"] for name in configuration_names
    }
    if "bm25" in required_retrievers and bm25_retriever is None:
        raise ValueError("A BM25 retriever is required for the selected configuration")
    if "dense" in required_retrievers and dense_retriever is None:
        raise ValueError("A dense retriever is required for the selected configuration")

    selected_questions = questions.head(limit) if limit is not None else questions
    records = []
    metadata_reference = None

    for _, row in selected_questions.iterrows():
        question = row["question"]
        retrieved_by_method = {}
        if "bm25" in required_retrievers:
            retrieved_by_method["bm25"] = bm25_retriever.search(
                question, top_k=RETRIEVAL_K
            )
        if "dense" in required_retrievers:
            # Reuse this exact ordered result for dense_basic and dense_abstain.
            retrieved_by_method["dense"] = dense_retriever.search(
                question, top_k=RETRIEVAL_K
            )

        for configuration_name in configuration_names:
            configuration = CONFIGURATIONS[configuration_name]
            retriever_name = configuration["retriever"]
            retrieved_documents = (
                retrieved_by_method[retriever_name]
                if retriever_name is not None
                else []
            )
            system_prompt, user_prompt = build_rag_prompt(
                question,
                retrieved_documents,
                prompt_version=configuration["prompt_version"],
            )
            generation = generate_answer(
                system_prompt,
                user_prompt,
                model=model,
                generation_settings=generation_settings,
            )
            record = {
                "question_id": row["question_id"],
                "question": question,
                "configuration": configuration_name,
                "prompt_version": configuration["prompt_version"],
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "retrieved_document_ids": [
                    document["document_id"] for document in retrieved_documents
                ],
                "retrieved_passage_texts": [
                    document["text"] for document in retrieved_documents
                ],
                "generated_answer": generation["generated_answer"],
                "model": generation["model"],
                "model_digest": generation["model_digest"],
                "generation_settings": generation["generation_settings"],
                "latency_seconds": generation["latency_seconds"],
                "done_reason": generation["done_reason"],
            }

            metadata = (record["model_digest"], record["generation_settings"])
            if metadata_reference is None:
                metadata_reference = metadata
            elif metadata != metadata_reference:
                raise RuntimeError(
                    "Model digest or generation settings differ across records"
                )
            records.append(record)

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Run controlled RAG baseline configurations."
    )
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument(
        "--config",
        choices=(*CONFIGURATIONS, "all"),
        default="all",
    )
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1")

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    questions = pd.read_csv(data_dir / "questions.csv")
    documents_path = data_dir / "documents.csv"

    selected_names = list(CONFIGURATIONS) if args.config == "all" else [args.config]
    needed_retrievers = {CONFIGURATIONS[name]["retriever"] for name in selected_names}
    bm25_retriever = (
        BM25Retriever(documents_path) if "bm25" in needed_retrievers else None
    )
    dense_retriever = (
        DenseRetriever(documents_path) if "dense" in needed_retrievers else None
    )
    records = run_experiments(
        questions,
        bm25_retriever=bm25_retriever,
        dense_retriever=dense_retriever,
        config=args.config,
        limit=args.limit,
    )

    for record in records:
        print(f"\nConfiguration: {record['configuration']}")
        print("Generated answer:", record["generated_answer"])
        print("Prompt version:", record["prompt_version"])
        print(f"Generation latency: {record['latency_seconds']:.6f} seconds")


if __name__ == "__main__":
    main()
