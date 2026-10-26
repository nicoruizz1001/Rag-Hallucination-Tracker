"""Controlled baseline generation configurations."""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import uuid

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
    on_record=None,
    run_id=None,
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
    run_id = run_id or str(uuid.uuid4())

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
            document_ids = [
                document["document_id"]
                for document in retrieved_documents
                if "document_id" in document
            ]
            passage_texts = [
                document["text"]
                for document in retrieved_documents
                if "text" in document
            ]
            if len(document_ids) != len(passage_texts):
                raise RuntimeError(
                    "Retrieved document-ID and passage-text counts differ"
                )

            system_prompt, user_prompt = build_rag_prompt(
                question,
                retrieved_documents,
                prompt_version=configuration["prompt_version"],
            )
            reference_answer = _parse_reference_answers(row.get("gold_answer", []))
            generation = generate_answer(
                system_prompt,
                user_prompt,
                model=model,
                generation_settings=generation_settings,
            )
            record = {
                "run_id": run_id,
                "created_at_utc": _utc_now(),
                "question_id": row["question_id"],
                "question": question,
                "configuration": configuration_name,
                "retrieval_method": retriever_name or "none",
                "k": configuration["top_k"],
                "retrieved_document_ids": document_ids,
                "retrieved_passage_texts": passage_texts,
                "generated_answer": generation["generated_answer"],
                "reference_answer": reference_answer,
                "is_answerable": _native_value(row.get("is_answerable")),
                "prompt_version": configuration["prompt_version"],
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
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
            if on_record is not None:
                on_record(record)
            records.append(record)

    return records


def _utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _native_value(value):
    """Convert pandas/numpy scalar values into JSON-native values."""
    return value.item() if hasattr(value, "item") else value


def _parse_reference_answers(value):
    """Parse a CSV SQuAD answer list without exposing it to the model."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if pd.isna(value) or not str(value).strip():
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Could not parse gold_answer list: {value!r}") from exc
    if not isinstance(parsed, list):
        raise ValueError("gold_answer must contain a list")
    return parsed


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _installed_package_versions():
    versions = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            versions[name] = distribution.version
    return dict(sorted(versions.items(), key=lambda item: item[0].lower()))


def _manifest_path(jsonl_path):
    return jsonl_path.with_suffix(".manifest.json")


def _new_default_output_path():
    output_dir = Path(__file__).resolve().parent.parent / "results" / "generations"
    output_dir.mkdir(parents=True, exist_ok=True)
    while True:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        candidate = output_dir / f"generations_{stamp}.jsonl"
        if not candidate.exists() and not _manifest_path(candidate).exists():
            return candidate


def _write_manifest(path, manifest, *, exclusive=False):
    if exclusive:
        with path.open("x", encoding="utf-8", newline="\n") as manifest_file:
            json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)
            manifest_file.write("\n")
            manifest_file.flush()
        return

    temporary_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("x", encoding="utf-8", newline="\n") as manifest_file:
            json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)
            manifest_file.write("\n")
            manifest_file.flush()
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def run_and_save_experiments(
    questions,
    bm25_retriever=None,
    dense_retriever=None,
    config="all",
    limit=None,
    model=DEFAULT_MODEL,
    generation_settings=None,
    output_path=None,
    questions_path=None,
    documents_path=None,
):
    """Run configurations and persist every completed record as JSONL."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    if config == "all":
        configuration_names = list(CONFIGURATIONS)
    elif config in CONFIGURATIONS:
        configuration_names = [config]
    else:
        raise ValueError(f"Unknown configuration: {config}")

    project_root = Path(__file__).resolve().parent.parent
    questions_path = Path(questions_path or project_root / "data" / "questions.csv")
    documents_path = Path(documents_path or project_root / "data" / "documents.csv")
    selected_questions = questions.head(limit) if limit is not None else questions
    expected_question_ids = [
        _native_value(value) for value in selected_questions["question_id"].tolist()
    ]
    expected_record_count = len(expected_question_ids) * len(configuration_names)

    jsonl_path = Path(output_path) if output_path is not None else _new_default_output_path()
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = _manifest_path(jsonl_path)
    if jsonl_path.exists():
        raise FileExistsError(f"Output file already exists: {jsonl_path}")
    if manifest_path.exists():
        raise FileExistsError(f"Manifest file already exists: {manifest_path}")

    run_id = str(uuid.uuid4())
    started_at = _utc_now()
    manifest = {
        "run_id": run_id,
        "started_at_utc": started_at,
        "ended_at_utc": None,
        "requested_configurations": configuration_names,
        "question_count": len(expected_question_ids),
        "selected_question_ids": expected_question_ids,
        "questions_sha256": _sha256(questions_path),
        "documents_sha256": _sha256(documents_path),
        "model_name": model,
        "model_digest": None,
        "generation_settings": None,
        "python_version": platform.python_version(),
        "package_versions": _installed_package_versions(),
        "completed_record_count": 0,
        "run_status": "running",
    }

    result_file = None
    manifest_created = False
    try:
        result_file = jsonl_path.open("x", encoding="utf-8", newline="\n")
        _write_manifest(manifest_path, manifest, exclusive=True)
        manifest_created = True

        required_retrievers = {
            CONFIGURATIONS[name]["retriever"] for name in configuration_names
        }
        if "bm25" in required_retrievers and bm25_retriever is None:
            bm25_retriever = BM25Retriever(documents_path)
        if "dense" in required_retrievers and dense_retriever is None:
            dense_retriever = DenseRetriever(documents_path)

        def write_record(record):
            result_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            result_file.flush()
            manifest["completed_record_count"] += 1
            manifest["model_digest"] = record["model_digest"]
            manifest["generation_settings"] = record["generation_settings"]
            _write_manifest(manifest_path, manifest)

        records = run_experiments(
            questions,
            bm25_retriever=bm25_retriever,
            dense_retriever=dense_retriever,
            config=config,
            limit=limit,
            model=model,
            generation_settings=generation_settings,
            on_record=write_record,
            run_id=run_id,
        )
        if len(records) != expected_record_count:
            raise RuntimeError(
                "Completed record count does not match the expected record count"
            )
        manifest["ended_at_utc"] = _utc_now()
        manifest["run_status"] = "completed"
        _write_manifest(manifest_path, manifest)
        return records, jsonl_path, manifest_path
    except BaseException as exc:
        if manifest_created:
            manifest["ended_at_utc"] = _utc_now()
            manifest["run_status"] = "failed"
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            try:
                _write_manifest(manifest_path, manifest)
            except OSError:
                pass
        raise
    finally:
        if result_file is not None:
            result_file.close()


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
    parser.add_argument(
        "--output",
        help="JSONL output path (defaults to a unique file under results/generations/)",
    )
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1")

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    questions = pd.read_csv(data_dir / "questions.csv")
    documents_path = data_dir / "documents.csv"

    records, output_path, manifest_path = run_and_save_experiments(
        questions,
        config=args.config,
        limit=args.limit,
        output_path=args.output,
        questions_path=data_dir / "questions.csv",
        documents_path=documents_path,
    )

    for record in records:
        print(f"\nConfiguration: {record['configuration']}")
        print("Generated answer:", record["generated_answer"])
        print("Prompt version:", record["prompt_version"])
        print(f"Generation latency: {record['latency_seconds']:.6f} seconds")
    print("\nSaved JSONL:", output_path)
    print("Saved manifest:", manifest_path)


if __name__ == "__main__":
    main()
