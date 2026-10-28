# Claim evaluator diagnostic report

> Development diagnostic set, written while investigating known evaluator failures. It is small and not independently sampled, so these numbers are NOT an unbiased accuracy benchmark or independent validation.

- Run: `evaluator_diag_20261028T005839983616Z_7d12cdf673ea`
- Extraction model: `llama3.2:3b` (digest `a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72`), reused from `evaluator_diag_20261028T003414829923Z_66fdf154564c`
- Judgment model: `qwen2.5:7b-instruct` (digest `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`)
- Prompt versions: {'extraction': 'claim_extraction_v5', 'judgment': 'claim_judgment_v9'}
- Diagnostic set matches frozen version: True
- Cases: 14; conditions: relevant_only, with_distractors

## Summary

- Correct labels / labeled judgments: 34/36 (94%) (failed judgments included as incorrect: 2)
- SUPPORTED: 26/26 (100%); predicted {'SUPPORTED': 26}
- CONTRADICTED: 5/6 (83%); predicted {'CONTRADICTED': 5, 'FAILED': 1}
- NOT_ENOUGH_EVIDENCE: 3/4 (75%); predicted {'NOT_ENOUGH_EVIDENCE': 3, 'FAILED': 1}
- Response-behavior correct: 14/14 (100%)
- Extraction failures (invalid after retry): 0/14 (0%)
- Extraction omissions: 2/18 (11%) required facts
- Unmatched claims (manual review; possible invented claims): 1/18 (6%)
- Label changes when distractors were added: 2/18 (11%) claims
- Irrelevant (distractor) citations: 0 excerpts; 0/18 (0%) with-distractor judgments cite one
- Judgment failures: 2/36 (6%)

## Request inspection

- Requests sent to Ollama: 39
- done_reason counts: {'stop': 35, 'length': 4}
- Max prompt_eval_count: 1212 (num_ctx [8192])
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

- Behavior: expected `no_abstention`, got `no_abstention`
- Claim: The Selby Choir did not perform at the 1962 Harbour Festival.

### unsupported_added_quantity (unsupported_quantity_or_date)

Answer: The Orchard Line railway opened in 1902 and employed 300 workers.

- Behavior: expected `no_abstention`, got `no_abstention`
- Claim: The Orchard Line railway opened in 1902.
- Claim: The Orchard Line railway employed 300 workers.
- **Label error** [with_distractors] workers_300: expected NOT_ENOUGH_EVIDENCE, got None — The Orchard Line railway employed 300 workers.
- **Label change** with distractors: NOT_ENOUGH_EVIDENCE -> None — The Orchard Line railway employed 300 workers.
- **Judgment failure** [with_distractors] The Orchard Line railway employed 300 workers.: claim judgment output invalid after one retry: JSONDecodeError: Unterminated string starting at: line 9 column 18 (char 252); done_reason=length; response may be truncated or incomplete

### unsupported_added_date (unsupported_quantity_or_date)

Answer: Mira Solberg composed the Fjordsong Symphony in 1874.

- Behavior: expected `no_abstention`, got `no_abstention`
- Claim: Mira Solberg composed the Fjordsong Symphony.
- Claim: The Fjordsong Symphony was composed in 1874.

### pure_refusal_self_knowledge (pure_refusal)

Answer: I don't have information about the Kestrel Canal. Could you provide more context about which region you mean?

- Behavior: expected `full_abstention`, got `full_abstention`
- Extraction retries: 1

### pure_refusal_documents (pure_refusal)

Answer: The provided documents do not say when the Orchard Line closed.

- Behavior: expected `full_abstention`, got `full_abstention`
- Extraction retries: 1

### partial_refusal (pure_refusal)

Answer: I could not find when the Kestrel Canal opened, but it has eleven locks.

- Behavior: expected `partial_abstention`, got `partial_abstention`
- Claim: The Kestrel Canal has eleven locks.

### multi_relation_role (multiple_relationships)

Answer: Hana Ruiz, the station's second director, introduced overnight observation shifts in 1979.

- Behavior: expected `no_abstention`, got `no_abstention`
- Claim: Hana Ruiz introduced overnight observation shifts in 1979.
- Claim: The Tarn Valley Weather Station had a second director.
- Extraction warning: No claim keeps description "station's second director" together with name 'Hana Ruiz'
- **Omission**: second_director
- Unmatched claim: The Tarn Valley Weather Station had a second director. — labels {'relevant_only': 'SUPPORTED', 'with_distractors': 'SUPPORTED'}

### multi_relation_mixed (multiple_relationships)

Answer: The Ardent Telescope, built by Corvane Optics, was installed on Mount Selby in 1967.

- Behavior: expected `no_abstention`, got `no_abstention`
- Claim: The Ardent Telescope was installed on Mount Selby.
- Claim: The Ardent Telescope was installed in 1967.
- **Omission**: built_by_corvane
- **Label error** [with_distractors] installed_1967: expected CONTRADICTED, got None — The Ardent Telescope was installed in 1967.
- **Label change** with distractors: CONTRADICTED -> None — The Ardent Telescope was installed in 1967.
- **Judgment failure** [with_distractors] The Ardent Telescope was installed in 1967.: claim judgment output invalid after one retry: JSONDecodeError: Unterminated string starting at: line 9 column 18 (char 278); done_reason=length; response may be truncated or incomplete

### multi_relation_ordinal (multiple_relationships)

Answer: Lena Hartmann opened her second bookshop, Inkwell & Oak, in Bremen in 2011.

- Behavior: expected `no_abstention`, got `no_abstention`
- Claim: Lena Hartmann opened Inkwell & Oak in Bremen in 2011.
- Claim: Inkwell & Oak is Lena Hartmann's second bookshop.

