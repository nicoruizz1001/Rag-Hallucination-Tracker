# Claim Support Annotation Guidelines

These guidelines define how to annotate generated answers against the passages supplied to the generator. They distinguish evidence support from answer correctness and citation quality. Examples below are explicitly synthetic and are not benchmark results.

## Claim-support labels

Assign one label to each independently checkable factual claim, using all passages supplied to the generator:

- **SUPPORTED:** The supplied passages establish the complete claim through explicit statements, faithful paraphrases, or straightforward inference.
- **CONTRADICTED:** The supplied passages establish information incompatible with the claim. Missing information alone is not contradiction.
- **NOT_ENOUGH_EVIDENCE:** The supplied passages neither establish nor contradict the complete claim. This does not necessarily mean the claim is factually false.

Judge the complete claim, including its dates, quantities, negations, and qualifications. A passage that supports only part of a claim does not support the whole claim.

## Annotation procedure

1. **Extract claims.** Split each generated answer into independently checkable factual claims. Preserve dates, quantities, negations, and qualifications. Do not introduce facts during claim extraction. Keep the original response alongside the extracted claims so omissions and distortions can be reviewed.
2. **Assess the evidence.** Judge support against all passages actually supplied to the generator, not only passages cited in the response or the original assigned passage. For each claim, record relevant document IDs and exact evidence excerpts. If evidence conflicts or is ambiguous, flag the claim for human review and give a reason.
3. **Evaluate separate dimensions.** Record these separately:
   - **Evidence support:** the claim-support label above.
   - **Question answering:** whether the response correctly answers the question asked.
   - **Citation support:** whether the cited documents support the claim associated with each citation.

   A supported statement can still fail to answer the question. A claim can also be supported by a supplied passage while the response cites a different, irrelevant document.
4. **Record abstention.** Track **full abstention** when the response withholds the requested answer, **partial abstention** when it answers some part but withholds another, and **no abstention** when it does not refuse or withhold an answer. Recognize natural refusals (for example, “I can’t determine that from the information provided”) as well as explicit text such as “Insufficient information.” A refusal accompanied by a factual claim is not pure abstention; label the withheld part and annotate the factual claim normally. Keep empty responses in a separate category; do not treat an empty response as an abstention. If an answer contains zero factual claims, claim faithfulness is **undefined**, not 100%.
5. **Handle SQuAD answerability carefully.** SQuAD answerability labels apply to the original assigned passage. Do not automatically use those labels to judge abstentions when multiple retrieved passages were supplied. Evidence-conditioned answerability remains unverified until assessed against the passages supplied to the generator.
6. **Handle the no-retrieval baseline.** Document-grounded faithfulness is not applicable when no evidence was supplied. Evaluate that baseline’s correctness separately.

## Synthetic examples

All passages, questions, answers, and document IDs in this section are invented for illustration.

### Supported paraphrase

**Question:** When did the fictional Alder Library open?

**Supplied passage (doc-A):** “Alder Library welcomed its first visitors in 2018.”

**Generated answer:** “The Alder Library opened in 2018.”

**Claim:** The Alder Library opened in 2018.  
**Label:** SUPPORTED  
**Evidence:** doc-A — “Alder Library welcomed its first visitors in 2018.”  
**Question answering:** Correct.

### Contradiction

**Question:** Did the fictional ferry run on Sundays in 2022?

**Supplied passage (doc-B):** “The ferry did not operate on Sundays in 2022.”

**Generated answer:** “The ferry ran on Sundays in 2022.”

**Claim:** The ferry ran on Sundays in 2022.  
**Label:** CONTRADICTED  
**Evidence:** doc-B — “The ferry did not operate on Sundays in 2022.”  
**Question answering:** Incorrect.

### Unsupported added detail

**Question:** When did the fictional observatory open?

**Supplied passage (doc-C):** “The observatory opened in 2019.”

**Generated answer:** “The observatory opened in 2019 after a $4 million renovation.”

**Claims:**

| Claim | Label | Evidence |
|---|---|---|
| The observatory opened in 2019. | SUPPORTED | doc-C — “The observatory opened in 2019.” |
| The observatory opened after a $4 million renovation. | NOT_ENOUGH_EVIDENCE | No supplied passage establishes the renovation or its cost. |

**Question answering:** The opening date is answered, but the added detail is unsupported.

### One sentence with claims that have different labels

**Question:** Which fictional town hosted the 2021 festival, and how many days did it last?

**Supplied passages:**

- doc-D: “The 2021 festival took place in Pineford.”
- doc-E: “The festival lasted three days.”

**Generated answer:** “The festival was held in Pineford and lasted four days.”

| Claim | Label | Evidence |
|---|---|---|
| The festival was held in Pineford. | SUPPORTED | doc-D — “The 2021 festival took place in Pineford.” |
| The festival lasted four days. | CONTRADICTED | doc-E — “The festival lasted three days.” |

**Question answering:** Partially correct; the duration is wrong.

### Supported statement that fails to answer the question

**Question:** What was the fictional bridge’s maximum load?

**Supplied passage (doc-F):** “The bridge opened in 2005. Its maximum load is 12 tonnes.”

**Generated answer:** “The bridge opened in 2005.”

**Claim:** The bridge opened in 2005.  
**Label:** SUPPORTED  
**Evidence:** doc-F — “The bridge opened in 2005.”  
**Question answering:** Incorrect/incomplete; the answer gives no maximum load.

### Citation pointing to the wrong document

**Question:** What is the fictional museum’s annual visitor limit?

**Supplied passages:**

- doc-G: “The museum admits at most 20,000 visitors per year.”
- doc-H: “The museum café serves lunch until 2 p.m.”

**Generated answer:** “The annual limit is 20,000 visitors [doc-H].”

**Claim:** The museum admits at most 20,000 visitors per year.  
**Evidence support:** SUPPORTED by doc-G — “The museum admits at most 20,000 visitors per year.”  
**Citation support:** Fails; cited doc-H does not support the annual visitor limit.  
**Question answering:** Correct fact, with an incorrect citation.

### Pure abstention

**Question:** What is the fictional island’s population?

**Supplied passage (doc-I):** “The island has a lighthouse.”

**Generated answer:** “Insufficient information to determine the population.”

**Factual claims:** None.  
**Abstention:** Full abstention; the explicit phrase is recognized as an abstention.  
**Claim faithfulness:** Undefined because there are zero factual claims.  
**Question answering:** Appropriate abstention given the supplied evidence.

### Abstention accompanied by a factual claim

**Question:** What is the fictional island’s population?

**Supplied passage (doc-J):** “The island has a lighthouse.”

**Generated answer:** “I can’t determine its population, but the island has a lighthouse.”

**Claim:** The island has a lighthouse.  
**Label:** SUPPORTED  
**Evidence:** doc-J — “The island has a lighthouse.”  
**Abstention:** Partial abstention; the response declines to provide the population but also makes a factual claim.  
**Question answering:** Does not provide the requested population.

## Human validation

Sample claims for human review. Reviewers must label each sampled claim without seeing the automatic evaluator’s decisions. They should see the question, the original generated response, the extracted claim, and all passages supplied to the generator, so they can check both extraction and support.

For every reviewed item, record the reviewer’s support label, relevant document IDs, exact evidence excerpts, and whether the automatic claim extraction omitted or distorted any factual content. Include a **needs-review** flag and a reason whenever evidence is conflicting or ambiguous, the claim cannot be reliably interpreted, a citation mapping is unclear, or extraction appears incomplete or distorted. Resolve or report flagged items separately; do not silently force an uncertain item into a definitive label.
