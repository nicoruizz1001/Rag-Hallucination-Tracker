"""Step-20 human-review workflow for the claim evaluator.

Three stages, run in order:

1. ``sample``: choose held-out generated answers reproducibly (seed 42) from the
   saved generation records alone, before any automatic judgment of them exists
   or is inspected. Questions already inspected during debugging are excluded.
2. ``export``: after the evaluator has been run on the sample, write a blinded
   reviewer packet (complete answers, full supplied passages, extracted claims,
   blank human fields) and a separate coordinator mapping holding source IDs,
   configuration and model names, and every automatic decision.
3. ``compare``: once reviewers have filled the CSVs, compare human labels with
   the automatic ones. Partial labelling is reported as partial; this module
   never declares human validation complete.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GENERATIONS = ROOT / "results" / "generations" / "generations_20261026T205113112864Z.jsonl"
RESULTS_ROOT = ROOT / "results" / "human_review"
INSPECTED_SOURCES = (ROOT / "results" / "claim_evaluation", ROOT / "results" / "generations")
RAG_CONFIGURATIONS = ("bm25_basic", "dense_basic", "dense_abstain")
DEFAULT_SEED = 42
DEFAULT_PER_STRATUM = 6
TARGET_CLAIMS = (50, 100)
SUPPORT_LABELS = ("SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_EVIDENCE")
BEHAVIOR_LABELS = ("full_abstention", "partial_abstention", "no_abstention", "empty_response")
YES_NO = ("yes", "no")
ANSWER_FIELDS = ["item_id", "human_abstention_behavior", "human_missing_claims",
                 "human_omitted_qualifications_relationships", "human_review_notes"]
EXTRACTION_UNAVAILABLE = (
    "EXTRACTION UNAVAILABLE: automatic claim extraction did not produce a valid result "
    "for this answer. Review the complete answer and list every factual claim yourself "
    "in claims.csv (rows `{item_id}-M1`, `-M2`, ...; one blank row is provided)."
)
# Strings that only automatic decisions or source metadata would put in reviewer files.
LEAK_MARKERS = ("SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_EVIDENCE", "needs_review",
                "review_reason", "explanation", "faithfulness", "is_answerable",
                "reference_answer", "llama", "qwen", "bm25", "dense_basic", "dense_abstain",
                "no_retrieval", "dense retrieval", "claim_extraction_v", "claim_judgment_v")
CLAIM_FIELDS = ["claim_id", "item_id", "claim_text", "human_support_label",
                "human_evidence_doc", "human_evidence_excerpt", "human_claim_distorted",
                "human_distortion_note", "human_review_notes"]


def _utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def collect_inspected_question_ids(generations_path, sources=INSPECTED_SOURCES):
    """Question IDs already seen during debugging, with where each was seen.

    Reads only question_id fields: every claim-evaluation run's answers, and any
    other generation file (smoke runs) besides the sampled one. No labels are read.
    """
    generations_path = Path(generations_path).resolve()
    seen = {}
    for source in sources:
        source = Path(source)
        if not source.exists():
            continue
        paths = list(source.glob("*/answers.jsonl")) + [
            p for p in source.glob("*.jsonl") if p.resolve() != generations_path]
        for path in sorted(paths):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                question_id = json.loads(line).get("question_id")
                if question_id:
                    seen.setdefault(question_id, set()).add(
                        str(path.resolve().relative_to(ROOT)) if path.resolve().is_relative_to(ROOT)
                        else str(path))
    return {qid: sorted(where) for qid, where in seen.items()}


def sample(*, generations_path=DEFAULT_GENERATIONS, seed=DEFAULT_SEED,
           per_stratum=DEFAULT_PER_STRATUM, extra_exclusions=(), output_root=RESULTS_ROOT,
           inspected_sources=INSPECTED_SOURCES):
    """Draw the review sample from generation records only (no evaluator output)."""
    generations_path = Path(generations_path)
    records = _read_jsonl(generations_path)
    inspected = collect_inspected_question_ids(generations_path, inspected_sources)
    exclusions = {qid: {"reason": "inspected during debugging", "seen_in": where}
                  for qid, where in inspected.items()}
    for qid in extra_exclusions:
        exclusions.setdefault(qid, {"reason": "excluded manually", "seen_in": []})
    benchmark_qids = {r["question_id"] for r in records}
    strata = {}
    for line_number, record in enumerate(records, start=1):
        if record["configuration"] not in RAG_CONFIGURATIONS:
            continue
        if record["question_id"] in exclusions:
            continue
        key = (record["configuration"], bool(record.get("is_answerable")))
        strata.setdefault(key, []).append((line_number, record))
    rng = random.Random(seed)
    used_questions, selected, stratum_report = set(), [], []
    # Each question appears at most once, so items are not near-duplicate
    # answers to the same question under different configurations.
    for key in sorted(strata):
        candidates = sorted(strata[key], key=lambda item: item[1]["question_id"])
        rng.shuffle(candidates)
        chosen = []
        for line_number, record in candidates:
            if len(chosen) == per_stratum:
                break
            if record["question_id"] not in used_questions:
                used_questions.add(record["question_id"])
                chosen.append((line_number, record))
        if len(chosen) < per_stratum:
            raise ValueError(f"Stratum {key} has only {len(chosen)} eligible questions")
        selected += chosen
        stratum_report.append({"configuration": key[0], "is_answerable": key[1],
                               "eligible_records": len(candidates), "selected": len(chosen)})
    # Shuffle presentation order so item IDs do not reveal configuration.
    rng.shuffle(selected)
    items = [{"item_id": f"R{index:03d}", "generation_line": line_number,
              "question_id": record["question_id"], "configuration": record["configuration"],
              "is_answerable": record.get("is_answerable")}
             for index, (line_number, record) in enumerate(selected, start=1)]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    review_dir = Path(output_root) / f"review_{timestamp}_seed{seed}"
    coordinator = review_dir / "coordinator"
    coordinator.mkdir(parents=True)
    with (coordinator / "sample_input.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for _, record in selected:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "created_at_utc": _utc_now(),
        "procedure": (
            "Records come from the saved generation file only; no evaluator output exists "
            "for the sample when it is drawn. Keep RAG configurations "
            f"{list(RAG_CONFIGURATIONS)} (no_retrieval has no supplied passages). Drop "
            "excluded questions. Stratify by (configuration, is_answerable); within each "
            "stratum, in sorted stratum order, sort candidates by question_id, shuffle with "
            f"random.Random({seed}), and take the first {per_stratum} whose question was not "
            "already selected. Finally shuffle all items with the same generator and number "
            "them R001.. in that order. The sample size was fixed before evaluation and is "
            "not adjusted after seeing automatic judgments."
        ),
        "seed": seed,
        "per_stratum": per_stratum,
        "generations_path": str(generations_path.resolve()),
        "generations_sha256": _sha256_file(generations_path),
        "benchmark_question_count": len(benchmark_qids),
        "excluded_question_ids": {qid: info for qid, info in sorted(exclusions.items())
                                  if qid in benchmark_qids},
        "strata": stratum_report,
        "item_count": len(items),
        "target_claims": list(TARGET_CLAIMS),
        "items": items,
        "sample_input_sha256": _sha256_file(coordinator / "sample_input.jsonl"),
    }
    (coordinator / "sample_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return review_dir


def _write_csv(path, fields, rows):
    # utf-8-sig so spreadsheet tools detect the encoding.
    with Path(path).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


REVIEWER_INSTRUCTIONS = """# Human review packet

Follow `report/annotation_guidelines.md`. Do **not** open `../coordinator/`; it
contains source identifiers and the automatic decisions you are checking.

For each item in `review_packet.md`, read the question, the complete generated
answer, and every supplied passage (Doc 1, Doc 2, ... in the order the generator
saw them). Then fill:

- `answers.csv`, one row per item:
  - `human_abstention_behavior`: full_abstention, partial_abstention,
    no_abstention, or empty_response.
  - `human_missing_claims`: factual statements in the answer that no listed
    claim captures at all. Leave it empty only if nothing is missing.
  - `human_omitted_qualifications_relationships`: names, numbers, dates,
    qualifiers or relationships stated in the answer that a listed claim
    dropped (e.g. "which fort was the first one", "up to an upper limit").
  - `human_review_notes`: anything else.
- `claims.csv`, one row per listed claim:
  - `human_support_label`: one of the three labels in the guidelines,
    judged against all supplied passages.
  - `human_evidence_doc` / `human_evidence_excerpt`: the supporting or
    contradicting passage, e.g. `Doc 2`, plus exact text.
  - `human_claim_distorted`: yes if the claim misstates what the answer says
    (added, reversed, or changed facts), otherwise no; explain in
    `human_distortion_note`.
  - Add rows for missing claims with claim_id `<item_id>-M1`, `-M2`, ...,
    and label them too. Items marked EXTRACTION UNAVAILABLE have no listed
    claims and one blank `-M1` row to start from.

A listed claim may be compound. Count it as supported only if every part is
supported; if parts differ, label the whole claim by its least-supported part
and describe the split in the notes.
"""


RETRY_POLICY = "one retry after a structural/provenance validation failure or done_reason=length"


def evaluator_configuration(eval_manifest):
    """The evaluator settings that must stay fixed for a validation round."""
    prompts = eval_manifest.get("prompts", {})
    return {
        "extraction_model": eval_manifest.get("extraction_model"),
        "extraction_model_digest": eval_manifest.get("extraction_model_digest"),
        "judgment_model": eval_manifest.get("judgment_model"),
        "judgment_model_digest": eval_manifest.get("judgment_model_digest"),
        "prompt_versions": eval_manifest.get("prompt_versions"),
        "prompt_sha256": {key: hashlib.sha256(text.encode("utf-8")).hexdigest()
                          for key, text in sorted(prompts.items())},
        "options": eval_manifest.get("options"),
        "retry_policy": RETRY_POLICY,
    }


def _freeze_evaluator(coordinator, eval_manifest):
    """Record the evaluator configuration once; refuse a different one later."""
    config = evaluator_configuration(eval_manifest)
    path = coordinator / "frozen_evaluator.json"
    if path.exists():
        frozen = json.loads(path.read_text(encoding="utf-8"))
        if frozen["configuration"] != config:
            raise ValueError(f"Evaluation run differs from the frozen evaluator in {path}")
        return frozen
    frozen = {"frozen_at_utc": _utc_now(), "evaluation_run_id": eval_manifest["run_id"],
              "note": "Fixed for this validation round; do not tune after inspecting "
                      "validation judgments.",
              "configuration": config}
    path.write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return frozen


def _extraction_unavailable(answer):
    extraction = answer.get("extraction") or {}
    return answer.get("evaluation_status") == "error" and (
        not extraction or "error" in extraction or answer.get("response_behavior") is None)


def export(review_dir, evaluation_run):
    """Write the blinded reviewer packet and the coordinator mapping."""
    review_dir, evaluation_run = Path(review_dir), Path(evaluation_run)
    coordinator, reviewer = review_dir / "coordinator", review_dir / "reviewer"
    sample_manifest = json.loads((coordinator / "sample_manifest.json").read_text(encoding="utf-8"))
    eval_manifest = json.loads((evaluation_run / "manifest.json").read_text(encoding="utf-8"))
    if eval_manifest["input_sha256"] != _sha256_file(coordinator / "sample_input.jsonl"):
        raise ValueError("The evaluation run was not produced from this sample's input file")
    if eval_manifest.get("run_status") not in {"completed", "completed_with_errors"}:
        raise ValueError(f"Evaluation run status is {eval_manifest.get('run_status')!r}")
    frozen = _freeze_evaluator(coordinator, eval_manifest)
    sources = _read_jsonl(coordinator / "sample_input.jsonl")
    answers = {a["source_record_index"]: a for a in _read_jsonl(evaluation_run / "answers.jsonl")}
    claims_by_answer = {}
    for claim in _read_jsonl(evaluation_run / "claims.jsonl"):
        claims_by_answer.setdefault(claim["answer_id"], []).append(claim)
    missing = [item["item_id"] for index, item in enumerate(sample_manifest["items"], start=1)
               if index not in answers]
    if missing:
        raise ValueError(f"Evaluation run is missing sampled items: {missing}; "
                         "re-run it with --limit at least the sample size")
    reviewer.mkdir()
    packet = ["# Review packet", "",
              "Read `README.md` first. Item order is random.", ""]
    answer_rows, claim_rows, mapping_items, claim_total = [], [], [], 0
    for index, (item, source) in enumerate(zip(sample_manifest["items"], sources), start=1):
        answer = answers[index]
        claims = sorted(claims_by_answer.get(answer["answer_id"], []),
                        key=lambda c: c["claim_index"])
        item_id = item["item_id"]
        unavailable = _extraction_unavailable(answer)
        packet += [f"## {item_id}", "", f"**Question:** {source['question']}", "",
                   "**Generated answer (complete):**", "", "> " + source["generated_answer"]
                   .replace("\n", "\n> "), "", "**Supplied passages:**", ""]
        for doc_index, text in enumerate(source["retrieved_passage_texts"], start=1):
            packet += [f"- **Doc {doc_index}:** {text}"]
        packet += ["", "**Extracted claims:**", ""]
        if unavailable:
            packet += [f"- {EXTRACTION_UNAVAILABLE.format(item_id=item_id)}"]
            claim_rows.append({"claim_id": f"{item_id}-M1", "item_id": item_id,
                               "claim_text": "", **{f: "" for f in CLAIM_FIELDS[3:]}})
        elif not claims:
            packet += ["- (no claims extracted)"]
        mapping_claims = []
        for claim in claims:
            claim_id = f"{item_id}-C{claim['claim_index'] + 1}"
            packet += [f"- `{claim_id}` {claim['claim']}"]
            claim_rows.append({"claim_id": claim_id, "item_id": item_id,
                               "claim_text": claim["claim"],
                               **{f: "" for f in CLAIM_FIELDS[3:]}})
            judgment = claim.get("judgment") or {}
            mapping_claims.append({
                "blind_claim_id": claim_id, "claim_id": claim["claim_id"],
                "claim": claim["claim"], "source_excerpt": claim["source_excerpt"],
                "automatic_label": claim["label"], "evidence": claim["evidence"],
                "explanation": claim["explanation"], "needs_review": claim["needs_review"],
                "review_reason": claim["review_reason"],
                "evaluation_status": claim["evaluation_status"],
                "judgment_error": judgment.get("error"),
                "judgment_attempts": [
                    {"attempt": a.get("attempt"), "done_reason": a.get("done_reason"),
                     "validation_error": a.get("validation_error")}
                    for a in judgment.get("attempts", [])],
            })
        claim_total += len(claims)
        packet.append("")
        answer_rows.append({"item_id": item_id, **{f: "" for f in ANSWER_FIELDS[1:]}})
        extraction = answer.get("extraction") or {}
        mapping_items.append({
            **item, "answer_id": answer["answer_id"],
            "doc_ids": source["retrieved_document_ids"],
            "reference_answer": source.get("reference_answer"),
            "automatic_response_behavior": answer.get("response_behavior"),
            "evaluation_status": answer.get("evaluation_status"),
            "extraction_status": "unavailable" if unavailable else "completed",
            "extraction_error": extraction.get("error") or answer.get("error"),
            "extraction_warnings": extraction.get("extraction_warnings", []),
            "extraction_attempts": [
                {"attempt": a.get("attempt"), "done_reason": a.get("done_reason"),
                 "validation_error": a.get("validation_error")}
                for a in extraction.get("attempts", [])],
            "needs_review": answer.get("needs_review"),
            "review_reason": answer.get("review_reason"),
            "faithfulness": answer.get("faithfulness"),
            "claims": mapping_claims,
        })
    (reviewer / "README.md").write_text(REVIEWER_INSTRUCTIONS, encoding="utf-8")
    (reviewer / "review_packet.md").write_text("\n".join(packet) + "\n", encoding="utf-8")
    _write_csv(reviewer / "answers.csv", ANSWER_FIELDS, answer_rows)
    _write_csv(reviewer / "claims.csv", CLAIM_FIELDS, claim_rows)
    low, high = TARGET_CLAIMS
    mapping = {
        "created_at_utc": _utc_now(),
        "evaluation_run": str(evaluation_run.resolve()),
        "evaluation_run_id": eval_manifest["run_id"],
        "evaluation_run_status": eval_manifest.get("run_status"),
        "frozen_evaluator": frozen["configuration"],
        "extraction_model": eval_manifest.get("extraction_model"),
        "judgment_model": eval_manifest.get("judgment_model"),
        "prompt_versions": eval_manifest.get("prompt_versions"),
        "claim_count": claim_total,
        "claim_count_within_target": low <= claim_total <= high,
        "items": mapping_items,
    }
    (coordinator / "mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return reviewer, claim_total


def verify(review_dir):
    """Check the exported packet against the frozen sample; report totals only."""
    review_dir = Path(review_dir)
    coordinator, reviewer = review_dir / "coordinator", review_dir / "reviewer"
    sample_manifest = json.loads((coordinator / "sample_manifest.json").read_text(encoding="utf-8"))
    mapping = json.loads((coordinator / "mapping.json").read_text(encoding="utf-8"))
    sources = _read_jsonl(coordinator / "sample_input.jsonl")
    answers = _read_csv(reviewer / "answers.csv")
    claims = _read_csv(reviewer / "claims.csv")
    packet = (reviewer / "review_packet.md").read_text(encoding="utf-8")
    reviewer_text = "".join(p.read_text(encoding="utf-8-sig") for p in sorted(reviewer.iterdir()))
    checks = {}
    expected_ids = [item["item_id"] for item in sample_manifest["items"]]
    checks["answer_count_matches_sample"] = (
        len(answers) == len(mapping["items"]) == len(sources) == sample_manifest["item_count"])
    checks["item_ids_match_frozen_sample"] = (
        [row["item_id"] for row in answers] == expected_ids
        and [(i["item_id"], i["question_id"], i["configuration"], i["generation_line"])
             for i in mapping["items"]]
        == [(i["item_id"], i["question_id"], i["configuration"], i["generation_line"])
            for i in sample_manifest["items"]])
    checks["sample_input_unchanged"] = (
        _sha256_file(coordinator / "sample_input.jsonl") == sample_manifest["sample_input_sha256"])
    generations = Path(sample_manifest["generations_path"])
    checks["generation_outputs_unchanged"] = (
        _sha256_file(generations) == sample_manifest["generations_sha256"])
    lines = _read_jsonl(generations)
    checks["sample_records_match_generation_file"] = all(
        lines[item["generation_line"] - 1] == source
        for item, source in zip(sample_manifest["items"], sources))
    sections = packet.split("\n## ")[1:]
    checks["every_answer_has_full_answer_and_evidence"] = len(sections) == len(sources) and all(
        section.startswith(item["item_id"] + "\n")
        and ("> " + source["generated_answer"].replace("\n", "\n> ")) in section
        and all(f"**Doc {n}:** {text}" in section
                for n, text in enumerate(source["retrieved_passage_texts"], start=1))
        and len(source["retrieved_passage_texts"]) > 0
        for section, item, source in zip(sections, sample_manifest["items"], sources))
    # Remove verbatim source and claim text, then look for anything only an
    # automatic decision or source metadata could have contributed.
    scrubbed = reviewer_text
    verbatim = [s["question"] for s in sources] + [s["generated_answer"] for s in sources] + [
        t for s in sources for t in s["retrieved_passage_texts"]] + [
        c["claim"] for item in mapping["items"] for c in item["claims"]]
    for text in sorted(verbatim, key=len, reverse=True):
        scrubbed = scrubbed.replace(text, "")
    leaked = sorted({m for m in LEAK_MARKERS if m in scrubbed})
    secret_values = {str(v) for item in mapping["items"] for v in (
        item["question_id"], item["answer_id"], *item["doc_ids"],
        item["review_reason"] or None, item.get("extraction_error")) if v}
    secret_values |= {str(v) for item in mapping["items"] for c in item["claims"]
                      for v in (c["explanation"], c["review_reason"], c["judgment_error"],
                                c["claim_id"]) if v}
    secret_values |= {str(v) for v in mapping["frozen_evaluator"].values()
                      if isinstance(v, str)}
    # Very short values (e.g. test digests) would match by accident.
    leaked += sorted(v[:40] for v in secret_values if len(v) >= 8 and v in scrubbed)
    checks["no_automatic_decisions_in_reviewer_files"] = not leaked
    checks["csv_columns_are_reviewer_fields_only"] = (
        list(answers[0]) == ANSWER_FIELDS and list(claims[0]) == CLAIM_FIELDS)
    human_cells = [row[f] for row in answers for f in ANSWER_FIELDS if f.startswith("human_")] + [
        row[f] for row in claims for f in CLAIM_FIELDS if f.startswith("human_")]
    checks["no_human_labels_filled"] = all(cell == "" for cell in human_cells)
    unavailable = sum(item["extraction_status"] == "unavailable" for item in mapping["items"])
    judgment_failures = sum(c["evaluation_status"] == "error"
                            for item in mapping["items"] for c in item["claims"])
    result = {
        "verified_at_utc": _utc_now(),
        "all_checks_passed": all(checks.values()),
        "checks": checks,
        "leaked_markers": leaked,
        "totals": {
            "answers": len(answers),
            "claims_extracted": mapping["claim_count"],
            "answers_with_extraction_unavailable": unavailable,
            "answers_with_any_evaluation_error": sum(
                item["evaluation_status"] == "error" for item in mapping["items"]),
            "claim_judgment_failures": judgment_failures,
            "claim_rows_for_review": len(claims),
            "blank_manual_fields": len(human_cells),
            "blank_answer_fields": len(answers) * (len(ANSWER_FIELDS) - 1),
            "blank_claim_fields": len(claims) * sum(f.startswith("human_") for f in CLAIM_FIELDS),
        },
    }
    path = coordinator / f"verification_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path, result


def _read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def compare(review_dir):
    """Compare filled human labels with the automatic decisions."""
    review_dir = Path(review_dir)
    mapping = json.loads((review_dir / "coordinator" / "mapping.json").read_text(encoding="utf-8"))
    answers = _read_csv(review_dir / "reviewer" / "answers.csv")
    claims = _read_csv(review_dir / "reviewer" / "claims.csv")
    errors = []
    for row in claims:
        label = row["human_support_label"].strip()
        if label and label not in SUPPORT_LABELS:
            errors.append(f"{row['claim_id']}: invalid support label {label!r}")
        distorted = row["human_claim_distorted"].strip().lower()
        if distorted and distorted not in YES_NO:
            errors.append(f"{row['claim_id']}: human_claim_distorted must be yes/no")
    for row in answers:
        behavior = row["human_abstention_behavior"].strip()
        if behavior and behavior not in BEHAVIOR_LABELS:
            errors.append(f"{row['item_id']}: invalid abstention behavior {behavior!r}")
    if errors:
        raise ValueError("Invalid human labels:\n" + "\n".join(errors))
    auto_claims = {c["blind_claim_id"]: c for item in mapping["items"] for c in item["claims"]}
    auto_items = {item["item_id"]: item for item in mapping["items"]}
    confusion, unlabeled, auto_failed, added = {}, [], [], []
    for row in claims:
        human = row["human_support_label"].strip()
        auto = auto_claims.get(row["claim_id"])
        if auto is None:
            added.append({"claim_id": row["claim_id"], "human_label": human or None})
            continue
        if not human:
            unlabeled.append(row["claim_id"])
            continue
        automatic = auto["automatic_label"] or "FAILED"
        if automatic == "FAILED":
            auto_failed.append(row["claim_id"])
        confusion.setdefault(human, {}).setdefault(automatic, 0)
        confusion[human][automatic] += 1
    labeled = sum(sum(v.values()) for v in confusion.values())
    agree = sum(confusion.get(label, {}).get(label, 0) for label in SUPPORT_LABELS)
    behavior_pairs = [(row["human_abstention_behavior"].strip(),
                       auto_items[row["item_id"]]["automatic_response_behavior"])
                      for row in answers if row["human_abstention_behavior"].strip()]
    report = {
        "created_at_utc": _utc_now(),
        "status": "partial" if unlabeled or len(behavior_pairs) < len(answers) else "all rows labelled",
        "note": ("Agreement with one reviewer's labels on a small sample; this is not by "
                 "itself complete human validation."),
        "evaluation_run_id": mapping["evaluation_run_id"],
        "claims_listed": len(auto_claims),
        "claims_labelled": labeled,
        "claims_unlabelled": len(unlabeled),
        "support_agreement": {"agree": agree, "labelled": labeled},
        "confusion_human_by_automatic": confusion,
        "automatic_failures_among_labelled": len(auto_failed),
        "reviewer_added_claims": added,
        "items": len(answers),
        "abstention_labelled": len(behavior_pairs),
        "abstention_agreement": sum(h == a for h, a in behavior_pairs),
        "items_with_missing_claims": sum(
            1 for row in answers if row["human_missing_claims"].strip()),
        "items_with_omitted_qualifications_relationships": sum(
            1 for row in answers if row["human_omitted_qualifications_relationships"].strip()),
        "claims_marked_distorted": sum(
            1 for row in claims if row["human_claim_distorted"].strip().lower() == "yes"),
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = review_dir / "coordinator" / f"agreement_{timestamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample", help="draw the held-out review sample (no model calls)")
    s.add_argument("--generations", type=Path, default=DEFAULT_GENERATIONS)
    s.add_argument("--seed", type=int, default=DEFAULT_SEED)
    s.add_argument("--per-stratum", type=int, default=DEFAULT_PER_STRATUM,
                   help="answers per (configuration, answerability) stratum")
    s.add_argument("--exclude-question-id", action="append", default=[],
                   help="additional question ID to exclude (repeatable)")
    s.add_argument("--output-root", type=Path, default=RESULTS_ROOT)
    e = sub.add_parser("export", help="write the blinded packet and coordinator mapping")
    e.add_argument("--review-dir", type=Path, required=True)
    e.add_argument("--evaluation-run", type=Path, required=True)
    v = sub.add_parser("verify", help="check the exported packet; prints totals only")
    v.add_argument("--review-dir", type=Path, required=True)
    c = sub.add_parser("compare", help="compare filled human labels with automatic ones")
    c.add_argument("--review-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "sample":
        review_dir = sample(generations_path=args.generations, seed=args.seed,
                            per_stratum=args.per_stratum,
                            extra_exclusions=args.exclude_question_id,
                            output_root=args.output_root)
        manifest = json.loads((review_dir / "coordinator" / "sample_manifest.json")
                              .read_text(encoding="utf-8"))
        print(f"Sample: {review_dir}")
        print(f"Items: {manifest['item_count']}; excluded questions: "
              f"{len(manifest['excluded_question_ids'])}")
    elif args.command == "export":
        reviewer, claim_total = export(args.review_dir, args.evaluation_run)
        low, high = TARGET_CLAIMS
        print(f"Reviewer packet: {reviewer}")
        print(f"Claims to review: {claim_total} (target {low}-{high}"
              + ("" if low <= claim_total <= high else "; OUTSIDE TARGET, see README") + ")")
    elif args.command == "verify":
        # Totals and check results only: no automatic labels or case-level output.
        path, result = verify(args.review_dir)
        print(f"Verification: {path}")
        for name, passed in result["checks"].items():
            print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        for name, value in result["totals"].items():
            print(f"  {name}: {value}")
        return 0 if result["all_checks_passed"] else 1
    else:
        path, report = compare(args.review_dir)
        print(f"Agreement report: {path}")
        print(f"Status: {report['status']}; support agreement "
              f"{report['support_agreement']['agree']}/{report['support_agreement']['labelled']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
