# Claim evaluator diagnostic report

> Development diagnostic set, written while investigating known evaluator failures. It is small and not independently sampled, so these numbers are NOT an unbiased accuracy benchmark or independent validation.

- Run: `evaluator_diag_20261028T010828943558Z_50ac1e1618aa`
- Extraction model: `qwen2.5:7b-instruct` (digest `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`)
- Judgment model: `qwen2.5:7b-instruct` (digest `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`)
- Prompt versions: {'extraction': 'claim_extraction_v5', 'judgment': 'claim_judgment_v9'}
- Diagnostic set matches frozen version: True
- Cases: 14; conditions: relevant_only, with_distractors

## Summary

- Correct labels / labeled judgments: 24/24 (100%) (failed judgments included as incorrect: 0)
- SUPPORTED: 18/18 (100%); predicted {'SUPPORTED': 18}
- CONTRADICTED: 6/6 (100%); predicted {'CONTRADICTED': 6}
- NOT_ENOUGH_EVIDENCE: 0/0; predicted {}
- Response-behavior correct: 13/14 (93%)
- Extraction failures (invalid after retry): 0/14 (0%)
- Extraction omissions: 7/18 (39%) required facts
- Unmatched claims (manual review; possible invented claims): 2/14 (14%)
- Label changes when distractors were added: 0/14 (0%) claims
- Irrelevant (distractor) citations: 0 excerpts; 0/14 (0%) with-distractor judgments cite one
- Judgment failures: 0/28 (0%)

## Request inspection

- Requests sent to Ollama: 42
- done_reason counts: {'stop': 42}
- Max prompt_eval_count: 1130 (num_ctx [8192])
- Raw explanations ending without terminal punctuation: 0
- Request issues: 0

## Per-case results

### explicit_supported (explicitly_supported)

Answer: The Halvorsen Bridge opened to traffic in 1931.

- Behavior: expected `no_abstention`, got `no_abstention`
- Claim: The Halvorsen Bridge opened to traffic in 1931.

### faithful_paraphrase (faithful_paraphrase)

Answer: The Lagos Seed Bank was established in 2004 by Dr. Amara Okafor.

- Behavior: expected `no_abstention`, got `no_abstention`
- Claim: The Lagos Seed Bank was established in 2004.
- Claim: Dr. Amara Okafor established the Lagos Seed Bank.

### pronoun_resolution (pronoun_resolution)

Answer: Teodora Vance discovered comet C/1994 Vance.

- Behavior: expected `no_abstention`, got `no_abstention`
- Claim: Teodora Vance discovered comet C/1994 Vance.

### surname_resolution (pronoun_resolution)

Answer: Rafael Ortiz carved the north portal of the Linden Chapel.

- Behavior: expected `no_abstention`, got `no_abstention`
- Claim: Rafael Ortiz carved the north portal of the Linden Chapel.

### contradicted_quantity (direct_contradiction)

Answer: The Kestrel Canal is 58 kilometres long.

- Behavior: expected `no_abstention`, got `no_abstention`
- Claim: The Kestrel Canal is 58 kilometres long.

### contradicted_negation (direct_contradiction)

Answer: The Selby Choir did not perform at the 1962 Harbour Festival.

- Behavior: expected `no_abstention`, got `partial_abstention` **(behavior error)**
- Claim: The Selby Choir did not perform at the 1962 Harbour Festival.

### unsupported_added_quantity (unsupported_quantity_or_date)

Answer: The Orchard Line railway opened in 1902 and employed 300 workers.

- Behavior: expected `no_abstention`, got `no_abstention`
- Claim: The Orchard Line railway opened in 1902.
- Extraction warning: Number '300' from the answer appears in no extracted claim
- **Omission**: workers_300

### unsupported_added_date (unsupported_quantity_or_date)

Answer: Mira Solberg composed the Fjordsong Symphony in 1874.

- Behavior: expected `no_abstention`, got `no_abstention`
- Claim: Mira Solberg composed the Fjordsong Symphony.
- Extraction warning: Number '1874' from the answer appears in no extracted claim
- **Omission**: composed_1874

### pure_refusal_self_knowledge (pure_refusal)

Answer: I don't have information about the Kestrel Canal. Could you provide more context about which region you mean?

- Behavior: expected `full_abstention`, got `full_abstention`

### pure_refusal_documents (pure_refusal)

Answer: The provided documents do not say when the Orchard Line closed.

- Behavior: expected `full_abstention`, got `full_abstention`

### partial_refusal (pure_refusal)

Answer: I could not find when the Kestrel Canal opened, but it has eleven locks.

- Behavior: expected `partial_abstention`, got `partial_abstention`
- Claim: The Kestrel Canal has eleven locks.

### multi_relation_role (multiple_relationships)

Answer: Hana Ruiz, the station's second director, introduced overnight observation shifts in 1979.

- Behavior: expected `no_abstention`, got `no_abstention`
- Claim: Hana Ruiz introduced overnight observation shifts at the Tarn Valley Weather Station.
- Claim: The overnight observation shifts were introduced in 1979.
- Extraction warning: No claim keeps description "station's second director" together with name 'Hana Ruiz'
- **Omission**: second_director
- Unmatched claim: The overnight observation shifts were introduced in 1979. — labels {'relevant_only': 'SUPPORTED', 'with_distractors': 'SUPPORTED'}

### multi_relation_mixed (multiple_relationships)

Answer: The Ardent Telescope, built by Corvane Optics, was installed on Mount Selby in 1967.

- Behavior: expected `no_abstention`, got `no_abstention`
- Claim: The Ardent Telescope was installed on Mount Selby in 1967.
- **Omission**: built_by_corvane

### multi_relation_ordinal (multiple_relationships)

Answer: Lena Hartmann opened her second bookshop, Inkwell & Oak, in Bremen in 2011.

- Behavior: expected `no_abstention`, got `no_abstention`
- Claim: Lena Hartmann opened her second bookshop in Bremen in 2011.
- **Omission**: second_bookshop_identity
- **Omission**: in_bremen
- **Omission**: in_2011
- Unmatched claim: Lena Hartmann opened her second bookshop in Bremen in 2011. — labels {'relevant_only': 'SUPPORTED', 'with_distractors': 'SUPPORTED'}

