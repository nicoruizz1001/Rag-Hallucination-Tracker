# Human review packet

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
