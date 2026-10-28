# Claim evaluator diagnostic report

> Development diagnostic set, written while investigating known evaluator failures. It is small and not independently sampled, so these numbers are NOT an unbiased accuracy benchmark or independent validation.

- Run: `evaluator_diag_20261028T003614356445Z_5fed886a5423`
- Extraction model: `llama3.2:3b` (digest `a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72`), reused from `evaluator_diag_20261028T003414829923Z_66fdf154564c`
- Judgment model: `llama3.2:3b` (digest `a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72`)
- Prompt versions: {'extraction': 'claim_extraction_v5', 'judgment': 'claim_judgment_v9'}
- Diagnostic set matches frozen version: True
- Cases: 14; conditions: relevant_only, with_distractors

## Summary

- Correct labels / labeled judgments: 22/36 (61%) (failed judgments included as incorrect: 1)
- SUPPORTED: 18/26 (69%); predicted {'SUPPORTED': 18, 'NOT_ENOUGH_EVIDENCE': 7, 'FAILED': 1}
- CONTRADICTED: 0/6 (0%); predicted {'NOT_ENOUGH_EVIDENCE': 4, 'SUPPORTED': 2}
- NOT_ENOUGH_EVIDENCE: 4/4 (100%); predicted {'NOT_ENOUGH_EVIDENCE': 4}
- Response-behavior correct: 14/14 (100%)
- Extraction failures (invalid after retry): 0/14 (0%)
- Extraction omissions: 2/18 (11%) required facts
- Unmatched claims (manual review; possible invented claims): 1/18 (6%)
- Label changes when distractors were added: 6/18 (33%) claims
- Irrelevant (distractor) citations: 20 excerpts; 16/18 (89%) with-distractor judgments cite one
- Judgment failures: 1/36 (3%)

## Request inspection

- Requests sent to Ollama: 44
- done_reason counts: {'stop': 44}
- Max prompt_eval_count: 1081 (num_ctx [8192])
- Raw explanations ending without terminal punctuation: 3
- Request issues: 3
  - request 13 (contradicted_quantity, judgment, relevant_only): raw explanation ends mid-sentence (done_reason=stop): ...'Missing length information'
  - request 26 (unsupported_added_date, judgment, relevant_only): raw explanation ends mid-sentence (done_reason=stop): ...'Passage provides composer but not date or entity'
  - request 38 (multi_relation_mixed, judgment, relevant_only): raw explanation ends mid-sentence (done_reason=stop): ...'Missing installation date'

## Per-case results

### explicit_supported (explicitly_supported)

Answer: The Halvorsen Bridge opened to traffic in 1931.

- Behavior: expected `no_abstention`, got `no_abstention`
- Claim: The Halvorsen Bridge opened to traffic in 1931.
- **Irrelevant citation** [with_distractors, SUPPORTED] for 'The Halvorsen Bridge opened to traffic in 1931.': 'A sourdough starter is a fermented mixture of flour and water that contains wild yeast and lactic acid bacteria.'

### faithful_paraphrase (faithful_paraphrase)

Answer: The Lagos Seed Bank was established in 2004 by Dr. Amara Okafor.

- Behavior: expected `no_abstention`, got `no_abstention`
- Claim: The Lagos Seed Bank was established in 2004.
- Claim: Dr. Amara Okafor established the Lagos Seed Bank.
- **Irrelevant citation** [with_distractors, SUPPORTED] for 'The Lagos Seed Bank was established in 2004.': 'The Accra Herbarium was established in 2006 by the botanist Kwame Mensah, who catalogued over 9,000 specimens.'
- **Irrelevant citation** [with_distractors, SUPPORTED] for 'Dr. Amara Okafor established the Lagos Seed Bank.': 'A sourdough starter is a fermented mixture of flour and water that contains wild yeast and lactic acid bacteria.'

### pronoun_resolution (pronoun_resolution)

Answer: Teodora Vance discovered comet C/1994 Vance.

- Behavior: expected `no_abstention`, got `no_abstention`
- Claim: Teodora Vance discovered comet C/1994 Vance.

### surname_resolution (pronoun_resolution)

Answer: Rafael Ortiz carved the north portal of the Linden Chapel.

- Behavior: expected `no_abstention`, got `no_abstention`
- Claim: Rafael Ortiz carved the north portal of the Linden Chapel.
- **Label error** [relevant_only] carved_north_portal: expected SUPPORTED, got NOT_ENOUGH_EVIDENCE — Rafael Ortiz carved the north portal of the Linden Chapel.
- **Label error** [with_distractors] carved_north_portal: expected SUPPORTED, got NOT_ENOUGH_EVIDENCE — Rafael Ortiz carved the north portal of the Linden Chapel.
- **Irrelevant citation** [with_distractors, NOT_ENOUGH_EVIDENCE] for 'Rafael Ortiz carved the north portal of the Linden Chapel.': 'A sourdough starter is a fermented mixture of flour and water that contains wild yeast and lactic acid bacteria.'
- **Irrelevant citation** [with_distractors, NOT_ENOUGH_EVIDENCE] for 'Rafael Ortiz carved the north portal of the Linden Chapel.': 'The Harrow Line employed 300 workers at its peak in 1910. The line closed to passengers in 1958.'

### contradicted_quantity (direct_contradiction)

Answer: The Kestrel Canal is 58 kilometres long.

- Behavior: expected `no_abstention`, got `no_abstention`
- Claim: The Kestrel Canal is 58 kilometres long.
- **Label error** [relevant_only] length_58: expected CONTRADICTED, got NOT_ENOUGH_EVIDENCE — The Kestrel Canal is 58 kilometres long.
- **Label error** [with_distractors] length_58: expected CONTRADICTED, got NOT_ENOUGH_EVIDENCE — The Kestrel Canal is 58 kilometres long.
- **Irrelevant citation** [with_distractors, NOT_ENOUGH_EVIDENCE] for 'The Kestrel Canal is 58 kilometres long.': 'The Brennan Canal is 58 kilometres long and was completed in 1821.'
- **Irrelevant citation** [with_distractors, NOT_ENOUGH_EVIDENCE] for 'The Kestrel Canal is 58 kilometres long.': 'A sourdough starter is a fermented mixture of flour and water that contains wild yeast and lactic acid bacteria.'

### contradicted_negation (direct_contradiction)

Answer: The Selby Choir did not perform at the 1962 Harbour Festival.

- Behavior: expected `no_abstention`, got `no_abstention`
- Claim: The Selby Choir did not perform at the 1962 Harbour Festival.
- **Label error** [relevant_only] did_not_perform: expected CONTRADICTED, got SUPPORTED — The Selby Choir did not perform at the 1962 Harbour Festival.
- **Label error** [with_distractors] did_not_perform: expected CONTRADICTED, got SUPPORTED — The Selby Choir did not perform at the 1962 Harbour Festival.
- **Irrelevant citation** [with_distractors, SUPPORTED] for 'The Selby Choir did not perform at the 1962 Harbour Festival.': 'A sourdough starter is a fermented mixture of flour and water that contains wild yeast and lactic acid bacteria.'

### unsupported_added_quantity (unsupported_quantity_or_date)

Answer: The Orchard Line railway opened in 1902 and employed 300 workers.

- Behavior: expected `no_abstention`, got `no_abstention`
- Claim: The Orchard Line railway opened in 1902.
- Claim: The Orchard Line railway employed 300 workers.
- **Irrelevant citation** [with_distractors, SUPPORTED] for 'The Orchard Line railway opened in 1902.': 'A sourdough starter is a fermented mixture of flour and water that contains wild yeast and lactic acid bacteria.'
- **Irrelevant citation** [with_distractors, NOT_ENOUGH_EVIDENCE] for 'The Orchard Line railway employed 300 workers.': 'The Harrow Line employed 300 workers at its peak in 1910.'

### unsupported_added_date (unsupported_quantity_or_date)

Answer: Mira Solberg composed the Fjordsong Symphony in 1874.

- Behavior: expected `no_abstention`, got `no_abstention`
- Claim: Mira Solberg composed the Fjordsong Symphony.
- Claim: The Fjordsong Symphony was composed in 1874.
- **Label error** [with_distractors] composer_only: expected SUPPORTED, got NOT_ENOUGH_EVIDENCE — Mira Solberg composed the Fjordsong Symphony.
- **Label change** with distractors: SUPPORTED -> NOT_ENOUGH_EVIDENCE — Mira Solberg composed the Fjordsong Symphony.
- **Irrelevant citation** [with_distractors, NOT_ENOUGH_EVIDENCE] for 'Mira Solberg composed the Fjordsong Symphony.': 'A sourdough starter is a fermented mixture of flour and water that contains wild yeast and lactic acid bacteria.'
- **Irrelevant citation** [with_distractors, NOT_ENOUGH_EVIDENCE] for 'Mira Solberg composed the Fjordsong Symphony.': 'The Marlow Brass Band performed at the 1962 Harbour Festival and returned the following year.'
- **Irrelevant citation** [with_distractors, NOT_ENOUGH_EVIDENCE] for 'The Fjordsong Symphony was composed in 1874.': 'A sourdough starter is a fermented mixture of flour and water that contains wild yeast and lactic acid bacteria.'
- **Irrelevant citation** [with_distractors, NOT_ENOUGH_EVIDENCE] for 'The Fjordsong Symphony was composed in 1874.': 'The Marlow Brass Band performed at the 1962 Harbour Festival and returned the following year.'

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
- **Label error** [with_distractors] eleven_locks: expected SUPPORTED, got None — The Kestrel Canal has eleven locks.
- **Label change** with distractors: SUPPORTED -> None — The Kestrel Canal has eleven locks.
- **Judgment failure** [with_distractors] The Kestrel Canal has eleven locks.: claim judgment output invalid after one retry: ValueError: Evidence item 0 excerpt is not verbatim in its passage

### multi_relation_role (multiple_relationships)

Answer: Hana Ruiz, the station's second director, introduced overnight observation shifts in 1979.

- Behavior: expected `no_abstention`, got `no_abstention`
- Claim: Hana Ruiz introduced overnight observation shifts in 1979.
- Claim: The Tarn Valley Weather Station had a second director.
- Extraction warning: No claim keeps description "station's second director" together with name 'Hana Ruiz'
- **Omission**: second_director
- **Label error** [with_distractors] introduced_shifts: expected SUPPORTED, got NOT_ENOUGH_EVIDENCE — Hana Ruiz introduced overnight observation shifts in 1979.
- **Label change** with distractors: SUPPORTED -> NOT_ENOUGH_EVIDENCE — Hana Ruiz introduced overnight observation shifts in 1979.
- **Label change** with distractors: SUPPORTED -> NOT_ENOUGH_EVIDENCE — The Tarn Valley Weather Station had a second director.
- **Irrelevant citation** [with_distractors, NOT_ENOUGH_EVIDENCE] for 'Hana Ruiz introduced overnight observation shifts in 1979.': 'A sourdough starter is a fermented mixture of flour and water that contains wild yeast and lactic acid bacteria.'
- **Irrelevant citation** [with_distractors, NOT_ENOUGH_EVIDENCE] for 'The Tarn Valley Weather Station had a second director.': 'A sourdough starter is a fermented mixture of flour and water that contains wild yeast and lactic acid bacteria.'
- Unmatched claim: The Tarn Valley Weather Station had a second director. — labels {'relevant_only': 'SUPPORTED', 'with_distractors': 'NOT_ENOUGH_EVIDENCE'}

### multi_relation_mixed (multiple_relationships)

Answer: The Ardent Telescope, built by Corvane Optics, was installed on Mount Selby in 1967.

- Behavior: expected `no_abstention`, got `no_abstention`
- Claim: The Ardent Telescope was installed on Mount Selby.
- Claim: The Ardent Telescope was installed in 1967.
- **Omission**: built_by_corvane
- **Label error** [relevant_only] installed_1967: expected CONTRADICTED, got NOT_ENOUGH_EVIDENCE — The Ardent Telescope was installed in 1967.
- **Label error** [with_distractors] installed_1967: expected CONTRADICTED, got NOT_ENOUGH_EVIDENCE — The Ardent Telescope was installed in 1967.
- **Label error** [with_distractors] on_mount_selby: expected SUPPORTED, got NOT_ENOUGH_EVIDENCE — The Ardent Telescope was installed on Mount Selby.
- **Label change** with distractors: SUPPORTED -> NOT_ENOUGH_EVIDENCE — The Ardent Telescope was installed on Mount Selby.
- **Irrelevant citation** [with_distractors, NOT_ENOUGH_EVIDENCE] for 'The Ardent Telescope was installed on Mount Selby.': 'The Meridian Telescope was installed on Mount Corra in 1967 by Halden Instruments.'
- **Irrelevant citation** [with_distractors, NOT_ENOUGH_EVIDENCE] for 'The Ardent Telescope was installed in 1967.': 'The Meridian Telescope was installed on Mount Corra in 1967 by Halden Instruments.'

### multi_relation_ordinal (multiple_relationships)

Answer: Lena Hartmann opened her second bookshop, Inkwell & Oak, in Bremen in 2011.

- Behavior: expected `no_abstention`, got `no_abstention`
- Claim: Lena Hartmann opened Inkwell & Oak in Bremen in 2011.
- Claim: Inkwell & Oak is Lena Hartmann's second bookshop.
- **Label error** [with_distractors] in_bremen: expected SUPPORTED, got NOT_ENOUGH_EVIDENCE — Lena Hartmann opened Inkwell & Oak in Bremen in 2011.
- **Label error** [with_distractors] in_2011: expected SUPPORTED, got NOT_ENOUGH_EVIDENCE — Lena Hartmann opened Inkwell & Oak in Bremen in 2011.
- **Label change** with distractors: SUPPORTED -> NOT_ENOUGH_EVIDENCE — Lena Hartmann opened Inkwell & Oak in Bremen in 2011.
- **Irrelevant citation** [with_distractors, NOT_ENOUGH_EVIDENCE] for 'Lena Hartmann opened Inkwell & Oak in Bremen in 2011.': 'A sourdough starter is a fermented mixture of flour and water that contains wild yeast and lactic acid bacteria.'
- **Irrelevant citation** [with_distractors, SUPPORTED] for 'Inkwell & Oak is Lena Hartmann's second bookshop.': 'A sourdough starter is a fermented mixture of flour and water that contains wild yeast and lactic acid bacteria.'

