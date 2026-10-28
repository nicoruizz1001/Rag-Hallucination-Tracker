"""Live diagnostic run of the claim evaluator on a small synthetic case set.

This is a development diagnostic for finding evaluator failure modes and for
controlled comparisons between evaluator models. The cases were written while
investigating known failures, so results are NOT an unbiased accuracy estimate
or independent validation, and must not be reported as one.

Expected labels are stored in a separate file and are loaded only after every
evaluator call has finished, so they can never reach a prompt.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import uuid

import ollama

if __package__:
    from . import claim_evaluation as ce
else:
    import claim_evaluation as ce


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "evaluator_diagnostics"
CASES_PATH = DATA_DIR / "cases.json"
EXPECTED_PATH = DATA_DIR / "expected_labels.json"
RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results" / "evaluator_diagnostics"
CONDITIONS = ("relevant_only", "with_distractors")
LABELS = ("SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_EVIDENCE")
DISCLAIMER = (
    "Development diagnostic set, written while investigating known evaluator "
    "failures. It is small and not independently sampled, so these numbers are "
    "NOT an unbiased accuracy benchmark or independent validation."
)
# Content hashes of the frozen diagnostic set (canonical JSON, so line-ending
# conversion on checkout does not change them). Editing either file breaks
# comparability with earlier runs; do so only with a new set version.
FROZEN_CONTENT_SHA256 = {
    "cases": "989d49233049bfe2b714a59b40147a8d8c99e4401d78d8d2e92417430b259361",
    "expected": "ac136683832f6158ccfbb617c1bd8ce99aae8ec3f0a219a5e8ab1f9d274abd00",
}


def _utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_sha256(path):
    """Hash of the parsed JSON, independent of whitespace and line endings."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return _text_sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                   separators=(",", ":")))


def check_frozen(cases_path, expected_path, *, allow_unfrozen=False):
    hashes = {"cases": content_sha256(cases_path), "expected": content_sha256(expected_path)}
    frozen = hashes == FROZEN_CONTENT_SHA256
    if not frozen and not allow_unfrozen:
        raise ValueError(
            "Diagnostic set differs from the frozen version "
            f"(got {hashes}, frozen {FROZEN_CONTENT_SHA256}); "
            "pass --allow-unfrozen to run anyway (results will not be comparable)"
        )
    return {**hashes, "matches_frozen_set": frozen}


def _document(text):
    # Opaque content hashes, matching the IDs used in real generation records.
    return {"document_id": _text_sha256(text), "text": text}


def build_condition_documents(case, distractor_pool):
    """Return the passage lists for each condition, and the distractor IDs.

    The ordering is deterministic and identical for every evaluator model.
    """
    relevant = [_document(text) for text in case["relevant_passages"]]
    distractors = [_document(distractor_pool[key]) for key in case["distractors"]]
    # Put the relevant passage between distractors to avoid a first-position bias.
    mixed = distractors[:1] + relevant + distractors[1:]
    return (
        {"relevant_only": relevant, "with_distractors": mixed},
        {doc["document_id"] for doc in distractors},
    )


def evaluator_fingerprint(stages):
    """Exact prompts, schemas and options shared by a comparison."""
    schema_text = lambda schema: json.dumps(schema, sort_keys=True)
    return {
        "prompt_versions": {"extraction": ce.EXTRACTION_PROMPT_VERSION,
                            "judgment": ce.JUDGMENT_PROMPT_VERSION},
        "prompts": {"extraction_system": ce.EXTRACTION_SYSTEM_PROMPT,
                    "judgment_system": ce.JUDGMENT_SYSTEM_PROMPT},
        "prompt_sha256": {"extraction": _text_sha256(ce.EXTRACTION_SYSTEM_PROMPT),
                          "judgment": _text_sha256(ce.JUDGMENT_SYSTEM_PROMPT)},
        "schema_sha256": {"extraction": _text_sha256(schema_text(ce.EXTRACTION_SCHEMA)),
                          "judgment": _text_sha256(schema_text(ce.JUDGMENT_SCHEMA))},
        "options": {stage: stages[stage]["options"] for stage in ("extraction", "judgment")},
    }


class RequestRecorder:
    """Wrap ollama.chat to record the literal request and raw response."""

    def __init__(self):
        self.records = []
        self.context = {}

    @contextmanager
    def installed(self):
        original = ollama.chat

        def recording_chat(**kwargs):
            response = original(**kwargs)
            field = (lambda name: response.get(name) if isinstance(response, dict)
                     else getattr(response, name, None))
            message = field("message")
            content = (message.get("content") if isinstance(message, dict)
                       else getattr(message, "content", None))
            self.records.append({
                **self.context,
                "request": {key: kwargs.get(key) for key in
                            ("model", "messages", "options", "format", "stream")},
                "response_content": content,
                "done_reason": field("done_reason"),
                "prompt_eval_count": field("prompt_eval_count"),
                "eval_count": field("eval_count"),
            })
            return response

        ollama.chat = recording_chat
        try:
            yield self
        finally:
            ollama.chat = original


def load_reused_extractions(run_dir, cases_hashes):
    """Load saved extractions so several judges see byte-identical claims."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    source_hash = (manifest.get("diagnostic_set") or {}).get("cases")
    if source_hash != cases_hashes["cases"]:
        raise ValueError(f"{run_dir} used a different diagnostic case set")
    if manifest["prompt_versions"]["extraction"] != ce.EXTRACTION_PROMPT_VERSION:
        raise ValueError(
            f"{run_dir} used extraction prompt {manifest['prompt_versions']['extraction']}, "
            f"current is {ce.EXTRACTION_PROMPT_VERSION}"
        )
    records = [json.loads(line) for line in
               (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    return {
        "source_run_id": manifest["run_id"],
        "extraction_model": manifest["stages"]["extraction"]["model"],
        "extraction_model_digest": manifest.get("model_digests", {}).get("extraction"),
        "by_case": {record["case_id"]: record for record in records},
    }


def run_cases(cases_doc, *, stages, recorder, reused=None):
    """Call the evaluator for every case. Uses no expected labels."""
    records = []
    pool = cases_doc["distractor_passages"]
    extraction_stage, judgment_stage = stages["extraction"], stages["judgment"]
    for case in cases_doc["cases"]:
        documents, distractor_ids = build_condition_documents(case, pool)
        recorder.context = {"case_id": case["case_id"], "stage": "extraction",
                            "condition": None, "intended_documents": None}
        # Extraction sees only the question and answer, so one extraction is
        # shared by both conditions and label changes isolate the passages.
        started = time.perf_counter()
        if reused is not None:
            source = reused["by_case"][case["case_id"]]
            extraction, extraction_error = source["extraction"], source["extraction_error"]
        else:
            try:
                extraction = ce.extract_claims(
                    case["question"], case["generated_answer"],
                    model=extraction_stage["model"],
                    generation_settings=extraction_stage["settings"] or None,
                )
                extraction_error = None
            except ce.StructuredOutputError as exc:
                extraction = {"claims": [], "response_behavior": None,
                              "attempts": exc.attempts, "extraction_warnings": []}
                extraction_error = str(exc)
        record = {
            "case_id": case["case_id"], "category": case["category"],
            "question": case["question"], "generated_answer": case["generated_answer"],
            "documents": documents, "distractor_ids": sorted(distractor_ids),
            "extraction": extraction, "extraction_error": extraction_error,
            "extraction_reused_from": reused["source_run_id"] if reused else None,
            "extraction_wall_seconds": time.perf_counter() - started,
            "judgments": {condition: [] for condition in CONDITIONS},
        }
        started = time.perf_counter()
        for condition in CONDITIONS:
            for index, claim in enumerate(extraction["claims"]):
                recorder.context = {
                    "case_id": case["case_id"], "stage": "judgment",
                    "condition": condition, "claim_index": index,
                    "intended_documents": documents[condition],
                }
                try:
                    judgment = ce.check_claim(
                        claim["claim"], documents[condition],
                        model=judgment_stage["model"],
                        generation_settings=judgment_stage["settings"] or None,
                    )
                    judgment_error = None
                except ce.StructuredOutputError as exc:
                    judgment = {"label": None, "evidence": [], "explanation": "",
                                "attempts": exc.attempts}
                    judgment_error = str(exc)
                record["judgments"][condition].append({
                    "claim_index": index, "claim": claim["claim"],
                    "label": judgment["label"], "evidence": judgment["evidence"],
                    "explanation": judgment["explanation"],
                    "needs_review": judgment.get("needs_review"),
                    "attempts": judgment["attempts"], "error": judgment_error,
                })
        record["judgment_wall_seconds"] = time.perf_counter() - started
        records.append(record)
    return records


def _matches(claim_text, fact):
    text = claim_text.casefold()
    return (all(token.casefold() in text for token in fact["must_contain"])
            and not any(token.casefold() in text for token in fact.get("must_not_contain", [])))


def score(records, expected_doc):
    """Compare evaluator output with expected behaviour, per case and condition.

    Failed judgments stay in the results (label None) and count as incorrect.
    """
    cases = []
    for record in records:
        expected = expected_doc["cases"][record["case_id"]]
        claims = record["extraction"]["claims"]
        case = {
            "case_id": record["case_id"], "category": record["category"],
            "generated_answer": record["generated_answer"],
            "expected_behavior": expected["response_behavior"],
            "actual_behavior": record["extraction"]["response_behavior"],
            "extraction_error": record["extraction_error"],
            "extraction_reused_from": record.get("extraction_reused_from"),
            "extraction_warnings": record["extraction"].get("extraction_warnings", []),
            "extraction_retries": max(0, len(record["extraction"].get("attempts", [])) - 1),
            "required_fact_count": sum(1 for f in expected["facts"] if not f.get("optional")),
            "claims": [claim["claim"] for claim in claims],
            "fact_judgments": [],
            "omissions": [], "label_errors": [], "label_changes": [],
            "irrelevant_citations": [], "unmatched_claims": [], "judgment_errors": [],
        }
        matched_indices = set()
        for fact in expected["facts"]:
            indices = [i for i, claim in enumerate(claims) if _matches(claim["claim"], fact)]
            matched_indices.update(indices)
            if not indices and not fact.get("optional"):
                case["omissions"].append(fact["fact_id"])
            for condition in CONDITIONS:
                for index in indices:
                    judgment = record["judgments"][condition][index]
                    case["fact_judgments"].append({
                        "fact_id": fact["fact_id"], "condition": condition,
                        "claim_index": index, "claim": claims[index]["claim"],
                        "expected": fact["label"], "actual": judgment["label"],
                        "failed": judgment["error"] is not None,
                    })
                    if judgment["label"] != fact["label"]:
                        case["label_errors"].append({
                            "condition": condition, "fact_id": fact["fact_id"],
                            "claim": claims[index]["claim"],
                            "expected": fact["label"], "actual": judgment["label"],
                        })
        case["unmatched_claims"] = [
            {"claim": claims[i]["claim"],
             "labels": {c: record["judgments"][c][i]["label"] for c in CONDITIONS}}
            for i in range(len(claims)) if i not in matched_indices
        ]
        for index, claim in enumerate(claims):
            labels = {c: record["judgments"][c][index]["label"] for c in CONDITIONS}
            if len(set(labels.values())) > 1:
                case["label_changes"].append({"claim": claim["claim"], **labels})
            for condition in CONDITIONS:
                judgment = record["judgments"][condition][index]
                if judgment["error"]:
                    case["judgment_errors"].append({"condition": condition,
                                                    "claim": claim["claim"],
                                                    "error": judgment["error"]})
                for item in judgment["evidence"]:
                    if item["document_id"] in record["distractor_ids"]:
                        case["irrelevant_citations"].append({
                            "condition": condition, "claim": claim["claim"],
                            "claim_index": index,
                            "label": judgment["label"], "excerpt": item["excerpt"],
                        })
        case["behavior_correct"] = case["expected_behavior"] == case["actual_behavior"]
        cases.append(case)
    return cases


def check_requests(request_records, stages=None):
    """Confirm prompts, passages, schemas and settings reached Ollama intact."""
    issues = []
    stats = {"requests": len(request_records), "done_reasons": {},
             "max_prompt_eval_count": 0, "num_ctx": set(), "mid_sentence_explanations": 0}
    for number, rec in enumerate(request_records, start=1):
        request = rec["request"]
        where = f"request {number} ({rec['case_id']}, {rec['stage']}, {rec.get('condition')})"
        messages = request["messages"] or []
        system = messages[0]["content"] if messages else None
        user = messages[1]["content"] if len(messages) > 1 else ""
        stats["done_reasons"][rec["done_reason"]] = stats["done_reasons"].get(rec["done_reason"], 0) + 1
        num_ctx = (request["options"] or {}).get("num_ctx")
        stats["num_ctx"].add(num_ctx)
        tokens = rec.get("prompt_eval_count")
        if isinstance(tokens, int):
            stats["max_prompt_eval_count"] = max(stats["max_prompt_eval_count"], tokens)
            if isinstance(num_ctx, int) and tokens >= num_ctx:
                issues.append(f"{where}: prompt_eval_count {tokens} reached num_ctx {num_ctx}")
        else:
            issues.append(f"{where}: Ollama returned no prompt_eval_count")
        if request["stream"] is not False:
            issues.append(f"{where}: stream was not disabled")
        if stages is not None:
            stage = stages[rec["stage"]]
            if request["model"] != stage["model"]:
                issues.append(f"{where}: model {request['model']!r} != configured {stage['model']!r}")
            if request["options"] != stage["options"]:
                issues.append(f"{where}: options {request['options']} != configured {stage['options']}")
        if rec["stage"] == "extraction":
            expected_system, expected_schema = ce.EXTRACTION_SYSTEM_PROMPT, ce.EXTRACTION_SCHEMA
        else:
            expected_system, expected_schema = ce.JUDGMENT_SYSTEM_PROMPT, ce.JUDGMENT_SCHEMA
        if system != expected_system:
            issues.append(f"{where}: system prompt differs from the {rec['stage']} prompt constant")
        if request["format"] != expected_schema:
            issues.append(f"{where}: format schema differs from the {rec['stage']} schema")
        if rec["stage"] == "judgment":
            try:
                # Retry prompts append a correction note after the JSON payload.
                payload, _ = json.JSONDecoder().raw_decode(user.split("\n", 1)[1])
            except (IndexError, json.JSONDecodeError):
                issues.append(f"{where}: user prompt JSON could not be parsed")
                continue
            if payload.get("retrieved_documents") != rec["intended_documents"]:
                issues.append(f"{where}: passages in the prompt differ from the intended passages")
            try:
                explanation = json.loads(rec["response_content"] or "")["explanation"]
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            if explanation.strip() and not re.search(r"[.!?][\"')\]]*$", explanation.strip()):
                stats["mid_sentence_explanations"] += 1
                issues.append(f"{where}: raw explanation ends mid-sentence "
                              f"(done_reason={rec['done_reason']}): ...{explanation[-60:]!r}")
    stats["num_ctx"] = sorted(n for n in stats["num_ctx"] if n is not None)
    return stats, issues


def runtime_summary(records):
    """Model latency and call counts per stage, from the saved attempts."""
    summary = {stage: {"calls": 0, "retries": 0, "model_seconds": 0.0, "wall_seconds": 0.0}
               for stage in ("extraction", "judgment")}
    for record in records:
        if not record.get("extraction_reused_from"):
            attempts = record["extraction"].get("attempts", [])
            summary["extraction"]["calls"] += len(attempts)
            summary["extraction"]["retries"] += max(0, len(attempts) - 1)
            summary["extraction"]["model_seconds"] += sum(a.get("latency_seconds") or 0 for a in attempts)
            summary["extraction"]["wall_seconds"] += record.get("extraction_wall_seconds", 0.0)
        for condition in CONDITIONS:
            for judgment in record["judgments"][condition]:
                attempts = judgment["attempts"]
                summary["judgment"]["calls"] += len(attempts)
                summary["judgment"]["retries"] += max(0, len(attempts) - 1)
                summary["judgment"]["model_seconds"] += sum(a.get("latency_seconds") or 0 for a in attempts)
        summary["judgment"]["wall_seconds"] += record.get("judgment_wall_seconds", 0.0)
    return summary


def summarize(cases, records):
    """Aggregate counts with explicit denominators."""
    judgments = [j for case in cases for j in case["fact_judgments"]]
    per_class = {}
    for label in LABELS:
        rows = [j for j in judgments if j["expected"] == label]
        predicted = {}
        for j in rows:
            key = j["actual"] or "FAILED"
            predicted[key] = predicted.get(key, 0) + 1
        per_class[label] = {"correct": sum(j["actual"] == label for j in rows),
                            "total": len(rows), "predicted": predicted}
    per_condition = {
        c: {"correct": sum(j["actual"] == j["expected"] for j in judgments if j["condition"] == c),
            "total": sum(1 for j in judgments if j["condition"] == c)}
        for c in CONDITIONS
    }
    claim_count = sum(len(case["claims"]) for case in cases)
    distractor_judgments = sum(len(r["judgments"]["with_distractors"]) for r in records)
    citing = {(case["case_id"], c["claim_index"]) for case in cases
              for c in case["irrelevant_citations"] if c["condition"] == "with_distractors"}
    return {
        "labeled_judgments": {"correct": sum(j["actual"] == j["expected"] for j in judgments),
                              "total": len(judgments),
                              "failed": sum(j["failed"] for j in judgments)},
        "per_class": per_class,
        "per_condition": per_condition,
        "label_changes": {"count": sum(len(c["label_changes"]) for c in cases),
                          "claims_judged_in_both": claim_count},
        "irrelevant_citations": {
            "excerpts": sum(len(c["irrelevant_citations"]) for c in cases),
            "judgments_citing_distractor": len(citing),
            "with_distractor_judgments": distractor_judgments,
        },
        "behavior": {"correct": sum(c["behavior_correct"] for c in cases), "total": len(cases)},
        "extraction_failures": {"count": sum(1 for c in cases if c["extraction_error"]),
                                "total": len(cases)},
        "omissions": {"count": sum(len(c["omissions"]) for c in cases),
                      "required_facts": sum(c["required_fact_count"] for c in cases)},
        "unmatched_claims": {"count": sum(len(c["unmatched_claims"]) for c in cases),
                             "claims": claim_count},
        "extraction_warnings": sum(len(c["extraction_warnings"]) for c in cases),
        "judgment_failures": {"count": sum(len(c["judgment_errors"]) for c in cases),
                              "judgments": sum(len(r["judgments"][c]) for r in records
                                               for c in CONDITIONS)},
        "runtime": runtime_summary(records),
    }


def _ratio(part, whole):
    return f"{part}/{whole}" + (f" ({100 * part / whole:.0f}%)" if whole else "")


def render_report(manifest, cases, request_stats, request_issues):
    summary = manifest["summary"]
    stages = manifest["stages"]
    lines = [
        "# Claim evaluator diagnostic report", "",
        f"> {DISCLAIMER}", "",
        f"- Run: `{manifest['run_id']}`",
        f"- Extraction model: `{stages['extraction']['model']}` "
        f"(digest `{manifest['model_digests'].get('extraction')}`)"
        + (f", reused from `{manifest['extraction_reused_from']}`" if manifest.get("extraction_reused_from") else ""),
        f"- Judgment model: `{stages['judgment']['model']}` (digest `{manifest['model_digests'].get('judgment')}`)",
        f"- Prompt versions: {manifest['prompt_versions']}",
        f"- Diagnostic set matches frozen version: {manifest['diagnostic_set']['matches_frozen_set']}",
        f"- Cases: {len(cases)}; conditions: {', '.join(CONDITIONS)}", "",
        "## Summary", "",
        f"- Correct labels / labeled judgments: {_ratio(summary['labeled_judgments']['correct'], summary['labeled_judgments']['total'])}"
        f" (failed judgments included as incorrect: {summary['labeled_judgments']['failed']})",
    ]
    lines += [f"- {label}: {_ratio(v['correct'], v['total'])}; predicted {v['predicted']}"
              for label, v in summary["per_class"].items()]
    lines += [
        f"- Response-behavior correct: {_ratio(summary['behavior']['correct'], summary['behavior']['total'])}",
        f"- Extraction failures (invalid after retry): {_ratio(summary['extraction_failures']['count'], summary['extraction_failures']['total'])}",
        f"- Extraction omissions: {_ratio(summary['omissions']['count'], summary['omissions']['required_facts'])} required facts",
        f"- Unmatched claims (manual review; possible invented claims): {_ratio(summary['unmatched_claims']['count'], summary['unmatched_claims']['claims'])}",
        f"- Label changes when distractors were added: {_ratio(summary['label_changes']['count'], summary['label_changes']['claims_judged_in_both'])} claims",
        f"- Irrelevant (distractor) citations: {summary['irrelevant_citations']['excerpts']} excerpts; "
        f"{_ratio(summary['irrelevant_citations']['judgments_citing_distractor'], summary['irrelevant_citations']['with_distractor_judgments'])} with-distractor judgments cite one",
        f"- Judgment failures: {_ratio(summary['judgment_failures']['count'], summary['judgment_failures']['judgments'])}",
        "", "## Request inspection", "",
        f"- Requests sent to Ollama: {request_stats['requests']}",
        f"- done_reason counts: {request_stats['done_reasons']}",
        f"- Max prompt_eval_count: {request_stats['max_prompt_eval_count']} (num_ctx {request_stats['num_ctx']})",
        f"- Raw explanations ending without terminal punctuation: {request_stats['mid_sentence_explanations']}",
        f"- Request issues: {len(request_issues)}",
    ]
    lines += [f"  - {issue}" for issue in request_issues]
    lines += ["", "## Per-case results", ""]
    for case in cases:
        flag = "" if case["behavior_correct"] else " **(behavior error)**"
        lines += [f"### {case['case_id']} ({case['category']})", "",
                  f"Answer: {case['generated_answer']}", "",
                  f"- Behavior: expected `{case['expected_behavior']}`, got `{case['actual_behavior']}`{flag}"]
        if case["extraction_error"]:
            lines.append(f"- Extraction error: {case['extraction_error']}")
        if case["extraction_retries"]:
            lines.append(f"- Extraction retries: {case['extraction_retries']}")
        lines += [f"- Claim: {claim}" for claim in case["claims"]]
        lines += [f"- Extraction warning: {w}" for w in case["extraction_warnings"]]
        lines += [f"- **Omission**: {fact}" for fact in case["omissions"]]
        lines += [f"- **Label error** [{e['condition']}] {e['fact_id']}: expected {e['expected']}, "
                  f"got {e['actual']} — {e['claim']}" for e in case["label_errors"]]
        lines += [f"- **Label change** with distractors: {c['relevant_only']} -> "
                  f"{c['with_distractors']} — {c['claim']}" for c in case["label_changes"]]
        lines += [f"- **Irrelevant citation** [{c['condition']}, {c['label']}] for '{c['claim']}': "
                  f"{c['excerpt'][:120]!r}" for c in case["irrelevant_citations"]]
        lines += [f"- Unmatched claim: {c['claim']} — labels {c['labels']}" for c in case["unmatched_claims"]]
        lines += [f"- **Judgment failure** [{e['condition']}] {e['claim']}: {e['error']}"
                  for e in case["judgment_errors"]]
        lines.append("")
    return "\n".join(lines)


def _write_scored_outputs(run_dir, manifest, records, request_records, expected_path):
    expected_doc = json.loads(Path(expected_path).read_text(encoding="utf-8"))
    cases = score(records, expected_doc)
    request_stats, request_issues = check_requests(request_records, manifest.get("stages"))
    manifest.update({
        "expected_path": str(Path(expected_path).resolve()),
        "expected_sha256": content_sha256(expected_path),
        "request_stats": request_stats, "request_issues": request_issues,
        "summary": summarize(cases, records),
    })
    (run_dir / "scored_cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "report.md").write_text(
        render_report(manifest, cases, request_stats, request_issues) + "\n", encoding="utf-8")


def run_diagnostics(*, model=ce.DEFAULT_MODEL, extraction_model=None, judgment_model=None,
                    extraction_settings=None, judgment_settings=None,
                    cases_path=CASES_PATH, expected_path=EXPECTED_PATH,
                    output_root=RESULTS_ROOT, reuse_extraction=None, allow_unfrozen=False):
    diagnostic_set = check_frozen(cases_path, expected_path, allow_unfrozen=allow_unfrozen)
    stages = ce.stage_configuration(
        model=model, extraction_model=extraction_model, judgment_model=judgment_model,
        extraction_settings=extraction_settings, judgment_settings=judgment_settings,
    )
    reused = load_reused_extractions(reuse_extraction, diagnostic_set) if reuse_extraction else None
    if reused:
        stages["extraction"]["model"] = reused["extraction_model"]
    cases_doc = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    run_id = f"evaluator_diag_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{uuid.uuid4().hex[:12]}"
    run_dir = root / run_id
    run_dir.mkdir()
    started_utc, started = _utc_now(), time.perf_counter()
    recorder = RequestRecorder()
    with recorder.installed():
        records = run_cases(cases_doc, stages=stages, recorder=recorder, reused=reused)
    wall_seconds = time.perf_counter() - started

    def digest(stage):
        for record in records:
            attempts = (record["extraction"].get("attempts", []) if stage == "extraction"
                        else [a for c in CONDITIONS for j in record["judgments"][c] for a in j["attempts"]])
            for attempt in attempts:
                if attempt.get("model_digest"):
                    return attempt["model_digest"]
        return None

    manifest = {
        "run_id": run_id, "started_at_utc": started_utc, "ended_at_utc": _utc_now(),
        "wall_seconds": wall_seconds,
        "disclaimer": DISCLAIMER,
        "stages": stages,
        "model_digests": {
            "extraction": reused["extraction_model_digest"] if reused else digest("extraction"),
            "judgment": digest("judgment"),
        },
        "extraction_reused_from": reused["source_run_id"] if reused else None,
        **evaluator_fingerprint(stages),
        "cases_path": str(Path(cases_path).resolve()),
        "diagnostic_set": diagnostic_set,
        "conditions": list(CONDITIONS),
    }
    with (run_dir / "records.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (run_dir / "requests.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for record in recorder.records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    # Expected labels are read only now, after every model call has finished.
    _write_scored_outputs(run_dir, manifest, records, recorder.records, expected_path)
    return run_dir


def rescore(run_dir, *, expected_path=EXPECTED_PATH):
    """Re-score saved records and requests without any model calls."""
    run_dir = Path(run_dir)
    read_jsonl = lambda name: [json.loads(line) for line in
                               (run_dir / name).read_text(encoding="utf-8").splitlines()]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["rescored_at_utc"] = _utc_now()
    _write_scored_outputs(run_dir, manifest, read_jsonl("records.jsonl"),
                          read_jsonl("requests.jsonl"), expected_path)
    return run_dir


COMPARABILITY_KEYS = ("diagnostic_set", "prompt_sha256", "schema_sha256", "options", "conditions")


def compare(run_dirs, *, output_root=RESULTS_ROOT):
    """Write a side-by-side markdown table for saved diagnostic runs."""
    manifests = [json.loads((Path(d) / "manifest.json").read_text(encoding="utf-8"))
                 for d in run_dirs]
    for manifest in manifests:
        if "summary" not in manifest or "stages" not in manifest:
            raise ValueError(f"{manifest['run_id']} predates the comparison format; "
                             "re-run it with the current evaluator_diagnostics.py")
    mismatches = [
        key for key in COMPARABILITY_KEYS
        if len({json.dumps(m.get(key), sort_keys=True) for m in manifests}) > 1
    ]
    s = [m["summary"] for m in manifests]
    rt = lambda m, stage, key: m["summary"]["runtime"][stage][key]
    rows = [
        ("Extraction model", [m["stages"]["extraction"]["model"]
                              + (f" (reused from {m['extraction_reused_from']})" if m.get("extraction_reused_from") else "")
                              for m in manifests]),
        ("Judgment model", [m["stages"]["judgment"]["model"] for m in manifests]),
        ("Judgment model digest", [str(m["model_digests"]["judgment"])[:12] for m in manifests]),
        ("Prompt versions", [f"{m['prompt_versions']['extraction']} / {m['prompt_versions']['judgment']}"
                             for m in manifests]),
        ("**Correct labels / labeled judgments**",
         [_ratio(x["labeled_judgments"]["correct"], x["labeled_judgments"]["total"]) for x in s]),
        ("  of which failed (counted incorrect)", [str(x["labeled_judgments"]["failed"]) for x in s]),
    ]
    for label in LABELS:
        rows.append((f"{label} correct", [_ratio(x["per_class"][label]["correct"],
                                                  x["per_class"][label]["total"]) for x in s]))
    rows.append(("CONTRADICTED predicted as", [
        ", ".join(f"{k}: {v}" for k, v in sorted(x["per_class"]["CONTRADICTED"]["predicted"].items()))
        for x in s]))
    for condition in CONDITIONS:
        rows.append((f"Correct, {condition}", [_ratio(x["per_condition"][condition]["correct"],
                                                      x["per_condition"][condition]["total"]) for x in s]))
    rows += [
        ("Label changes after adding distractors",
         [_ratio(x["label_changes"]["count"], x["label_changes"]["claims_judged_in_both"]) for x in s]),
        ("Judgments citing a distractor", [
            _ratio(x["irrelevant_citations"]["judgments_citing_distractor"],
                   x["irrelevant_citations"]["with_distractor_judgments"]) for x in s]),
        ("Distractor excerpts cited (total)", [str(x["irrelevant_citations"]["excerpts"]) for x in s]),
        ("Response behavior correct", [_ratio(x["behavior"]["correct"], x["behavior"]["total"]) for x in s]),
        ("Extraction omissions / required facts",
         [_ratio(x["omissions"]["count"], x["omissions"]["required_facts"]) for x in s]),
        ("Unmatched claims (possible invented; manual review)",
         [_ratio(x["unmatched_claims"]["count"], x["unmatched_claims"]["claims"]) for x in s]),
        ("Extraction failures after retry", [_ratio(x["extraction_failures"]["count"],
                                                    x["extraction_failures"]["total"]) for x in s]),
        ("Judgment failures after retry", [_ratio(x["judgment_failures"]["count"],
                                                  x["judgment_failures"]["judgments"]) for x in s]),
        ("Extraction retries / calls", [f"{rt(m, 'extraction', 'retries')}/{rt(m, 'extraction', 'calls')}"
                                        for m in manifests]),
        ("Judgment retries / calls", [f"{rt(m, 'judgment', 'retries')}/{rt(m, 'judgment', 'calls')}"
                                      for m in manifests]),
        ("Model time extraction / judgment (s)",
         [f"{rt(m, 'extraction', 'model_seconds'):.0f} / {rt(m, 'judgment', 'model_seconds'):.0f}"
          for m in manifests]),
        ("Run wall-clock (s)", [f"{m['wall_seconds']:.0f}" for m in manifests]),
    ]
    header = "| Metric | " + " | ".join(f"`{m['run_id']}`" for m in manifests) + " |"
    lines = [
        "# Evaluator comparison (development diagnostic)", "",
        f"> {DISCLAIMER}", "",
        "Denominators: labeled judgments are (claim, condition) pairs whose claim matches an "
        "expected fact; they depend on the extracted claims, so they differ between runs unless "
        "extraction is reused (`--reuse-extraction`). Failed judgments remain in the denominator "
        "as incorrect. Unmatched claims are listed in each run's report.md for manual review.", "",
        "Comparable configuration (frozen cases/labels, prompts, schemas, options, conditions): "
        + ("**yes**" if not mismatches else f"**NO** — differs in {', '.join(mismatches)}"), "",
        header, "|" + "---|" * (len(manifests) + 1),
    ]
    lines += ["| " + name + " | " + " | ".join(values) + " |" for name, values in rows]
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / f"comparison_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.md"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ce.add_stage_arguments(parser)
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--expected", type=Path, default=EXPECTED_PATH)
    parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--reuse-extraction", type=Path, metavar="RUN_DIR",
                        help="reuse the saved extraction of an earlier run (same case set and "
                             "extraction prompt) so only the judge varies")
    parser.add_argument("--allow-unfrozen", action="store_true",
                        help="run even if the case/label files differ from the frozen set")
    parser.add_argument("--rescore", type=Path, metavar="RUN_DIR",
                        help="re-score a saved run without calling the model")
    parser.add_argument("--compare", type=Path, nargs="+", metavar="RUN_DIR",
                        help="write a comparison table for saved runs (no model calls)")
    args = parser.parse_args(argv)
    if args.rescore:
        run_dir = rescore(args.rescore, expected_path=args.expected)
        print(f"Re-scored: {run_dir / 'report.md'}")
        return 0
    if args.compare:
        path = compare(args.compare, output_root=args.output_root)
        print(f"Comparison table: {path}")
        return 0
    run_dir = run_diagnostics(
        **ce.stage_kwargs(args), cases_path=args.cases, expected_path=args.expected,
        output_root=args.output_root, reuse_extraction=args.reuse_extraction,
        allow_unfrozen=args.allow_unfrozen,
    )
    print(f"Saved evaluator diagnostic run: {run_dir}")
    print(f"Report: {run_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
