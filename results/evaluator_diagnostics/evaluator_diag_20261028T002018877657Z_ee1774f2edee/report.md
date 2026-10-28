# Claim evaluator diagnostic report

> Development diagnostic set, written while investigating known evaluator failures. It is small and not independently sampled, so these numbers are NOT an unbiased accuracy benchmark.

- Run: `evaluator_diag_20261028T002018877657Z_ee1774f2edee`
- Evaluator model: `llama3.2:3b` (digest `a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72`)
- Prompt versions: {'extraction': 'claim_extraction_v4', 'judgment': 'claim_judgment_v8'}
- Cases: 14; conditions: relevant_only, with_distractors

## Summary

- Response-behavior errors: 1
- Extraction failures (invalid after retry): 1
- Extraction omissions (expected fact in no claim): 2
- Label errors (claim-condition pairs): 13
- Label changes when distractors were added: 10
- Irrelevant (distractor) citations: 21
- Judgment failures: 2

## Request inspection

- Requests sent to Ollama: 67
- done_reason counts: {'stop': 67}
- Max prompt_eval_count: 1067 (num_ctx [8192])
- Raw explanations ending mid-sentence: 2
- Request issues: 2
  - request 47 (multi_relation_role, judgment, relevant_only): raw explanation ends mid-sentence (done_reason=stop): ...'Established date and role of director'
  - request 65 (multi_relation_ordinal, judgment, with_distractors): raw explanation ends mid-sentence (done_reason=stop): ..."nformation about the bookshop's location and date of opening"

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
- **Label error** [with_distractors] carved_north_portal: expected SUPPORTED, got NOT_ENOUGH_EVIDENCE — Rafael Ortiz carved the north portal of the Linden Chapel.
- **Label change** with distractors: SUPPORTED -> NOT_ENOUGH_EVIDENCE — Rafael Ortiz carved the north portal of the Linden Chapel.
- **Irrelevant citation** [with_distractors, NOT_ENOUGH_EVIDENCE] for 'Rafael Ortiz carved the north portal of the Linden Chapel.': 'A sourdough starter is a fermented mixture of flour and water that contains wild yeast and lactic acid bacteria.'

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
- **Irrelevant citation** [with_distractors, SUPPORTED] for 'The Selby Choir did not perform at the 1962 Harbour Festival.': 'The Marlow Brass Band performed at the 1962 Harbour Festival and returned the following year.'

### unsupported_added_quantity (unsupported_quantity_or_date)

Answer: The Orchard Line railway opened in 1902 and employed 300 workers.

- Behavior: expected `no_abstention`, got `no_abstention`
- Claim: The Orchard Line railway opened in 1902.
- Claim: The Orchard Line railway employed 300 workers.
- **Label error** [with_distractors] opened_1902: expected SUPPORTED, got None — The Orchard Line railway opened in 1902.
- **Label change** with distractors: SUPPORTED -> None — The Orchard Line railway opened in 1902.
- **Irrelevant citation** [with_distractors, NOT_ENOUGH_EVIDENCE] for 'The Orchard Line railway employed 300 workers.': 'A sourdough starter is a fermented mixture of flour and water that contains wild yeast and lactic acid bacteria.'
- **Judgment failure** [with_distractors] The Orchard Line railway opened in 1902.: claim judgment output invalid after one retry: ValueError: Explanation must contain no more than 2 short sentences

### unsupported_added_date (unsupported_quantity_or_date)

Answer: Mira Solberg composed the Fjordsong Symphony in 1874.

- Behavior: expected `no_abstention`, got `no_abstention`
- Claim: Mira Solberg composed the Fjordsong Symphony.
- Claim: The Fjordsong Symphony was composed in 1874.
- **Irrelevant citation** [with_distractors, SUPPORTED] for 'Mira Solberg composed the Fjordsong Symphony.': 'A sourdough starter is a fermented mixture of flour and water that contains wild yeast and lactic acid bacteria.'
- **Irrelevant citation** [with_distractors, NOT_ENOUGH_EVIDENCE] for 'The Fjordsong Symphony was composed in 1874.': 'A sourdough starter is a fermented mixture of flour and water that contains wild yeast and lactic acid bacteria.'

### pure_refusal_self_knowledge (pure_refusal)

Answer: I don't have information about the Kestrel Canal. Could you provide more context about which region you mean?

- Behavior: expected `full_abstention`, got `None` **(behavior error)**
- Extraction error: claim extraction output invalid after one retry: ValueError: Claim 0 is an abstention, self-knowledge, or clarification statement, not a factual claim about the world
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
- Claim: Hana Ruiz introduced overnight observation shifts
- Claim: The station's second director introduced overnight observation shifts in 1979
- Claim: The Tarn Valley Weather Station had a second director
- Claim: The Tarn Valley Weather Station was established in 1979
- Extraction warning: No claim keeps description "station's second director" together with name 'Hana Ruiz'
- **Omission**: second_director
- **Label change** with distractors: SUPPORTED -> NOT_ENOUGH_EVIDENCE — The Tarn Valley Weather Station had a second director
- **Label change** with distractors: SUPPORTED -> NOT_ENOUGH_EVIDENCE — The Tarn Valley Weather Station was established in 1979
- **Irrelevant citation** [with_distractors, SUPPORTED] for 'Hana Ruiz introduced overnight observation shifts': 'In 1983, Ivo Brandt became the third director of the Ashford Coastal Weather Station.'
- **Irrelevant citation** [with_distractors, SUPPORTED] for 'The station's second director introduced overnight observation shifts in 1979': "In 1983, Ivo Brandt became the third director of the Ashford Coastal Weather Station. He replaced the station's wind ins"
- **Irrelevant citation** [with_distractors, NOT_ENOUGH_EVIDENCE] for 'The Tarn Valley Weather Station had a second director': 'A sourdough starter is a fermented mixture of flour and water that contains wild yeast and lactic acid bacteria.'
- **Irrelevant citation** [with_distractors, NOT_ENOUGH_EVIDENCE] for 'The Tarn Valley Weather Station was established in 1979': 'In 1983, Ivo Brandt became the third director of the Ashford Coastal Weather Station.'
- Unmatched claim: The station's second director introduced overnight observation shifts in 1979 — labels {'relevant_only': 'SUPPORTED', 'with_distractors': 'SUPPORTED'}
- Unmatched claim: The Tarn Valley Weather Station had a second director — labels {'relevant_only': 'SUPPORTED', 'with_distractors': 'NOT_ENOUGH_EVIDENCE'}
- Unmatched claim: The Tarn Valley Weather Station was established in 1979 — labels {'relevant_only': 'SUPPORTED', 'with_distractors': 'NOT_ENOUGH_EVIDENCE'}

### multi_relation_mixed (multiple_relationships)

Answer: The Ardent Telescope, built by Corvane Optics, was installed on Mount Selby in 1967.

- Behavior: expected `no_abstention`, got `no_abstention`
- Claim: The Ardent Telescope was installed on Mount Selby.
- Claim: The Ardent Telescope was installed in 1967.
- **Omission**: built_by_corvane
- **Label error** [relevant_only] installed_1967: expected CONTRADICTED, got NOT_ENOUGH_EVIDENCE — The Ardent Telescope was installed in 1967.
- **Label error** [with_distractors] installed_1967: expected CONTRADICTED, got SUPPORTED — The Ardent Telescope was installed in 1967.
- **Label change** with distractors: NOT_ENOUGH_EVIDENCE -> SUPPORTED — The Ardent Telescope was installed in 1967.
- **Irrelevant citation** [with_distractors, SUPPORTED] for 'The Ardent Telescope was installed on Mount Selby.': 'The Meridian Telescope was installed on Mount Corra in 1967 by Halden Instruments.'
- **Irrelevant citation** [with_distractors, SUPPORTED] for 'The Ardent Telescope was installed in 1967.': 'The Meridian Telescope was installed on Mount Corra in 1967 by Halden Instruments.'

### multi_relation_ordinal (multiple_relationships)

Answer: Lena Hartmann opened her second bookshop, Inkwell & Oak, in Bremen in 2011.

- Behavior: expected `no_abstention`, got `no_abstention`
- Claim: Lena Hartmann opened her second bookshop, Inkwell & Oak.
- Claim: Inkwell & Oak was opened in Bremen.
- Claim: Inkwell & Oak was opened in 2011.
- Claim: Bremen is the location where Inkwell & Oak was opened.
- **Label error** [with_distractors] second_bookshop_identity: expected SUPPORTED, got NOT_ENOUGH_EVIDENCE — Lena Hartmann opened her second bookshop, Inkwell & Oak.
- **Label error** [with_distractors] in_bremen: expected SUPPORTED, got NOT_ENOUGH_EVIDENCE — Inkwell & Oak was opened in Bremen.
- **Label error** [with_distractors] in_bremen: expected SUPPORTED, got NOT_ENOUGH_EVIDENCE — Bremen is the location where Inkwell & Oak was opened.
- **Label error** [with_distractors] in_2011: expected SUPPORTED, got NOT_ENOUGH_EVIDENCE — Inkwell & Oak was opened in 2011.
- **Label change** with distractors: SUPPORTED -> NOT_ENOUGH_EVIDENCE — Lena Hartmann opened her second bookshop, Inkwell & Oak.
- **Label change** with distractors: SUPPORTED -> NOT_ENOUGH_EVIDENCE — Inkwell & Oak was opened in Bremen.
- **Label change** with distractors: SUPPORTED -> NOT_ENOUGH_EVIDENCE — Inkwell & Oak was opened in 2011.
- **Label change** with distractors: SUPPORTED -> NOT_ENOUGH_EVIDENCE — Bremen is the location where Inkwell & Oak was opened.
- **Irrelevant citation** [with_distractors, NOT_ENOUGH_EVIDENCE] for 'Lena Hartmann opened her second bookshop, Inkwell & Oak.': 'A sourdough starter is a fermented mixture of flour and water that contains wild yeast and lactic acid bacteria.'
- **Irrelevant citation** [with_distractors, NOT_ENOUGH_EVIDENCE] for 'Inkwell & Oak was opened in Bremen.': 'A sourdough starter is a fermented mixture of flour and water that contains wild yeast and lactic acid bacteria.'
- **Irrelevant citation** [with_distractors, NOT_ENOUGH_EVIDENCE] for 'Inkwell & Oak was opened in Bremen.': 'The Marlow Brass Band performed at the 1962 Harbour Festival and returned the following year.'
- **Irrelevant citation** [with_distractors, NOT_ENOUGH_EVIDENCE] for 'Inkwell & Oak was opened in 2011.': 'A sourdough starter is a fermented mixture of flour and water that contains wild yeast and lactic acid bacteria.'
- **Irrelevant citation** [with_distractors, NOT_ENOUGH_EVIDENCE] for 'Bremen is the location where Inkwell & Oak was opened.': 'A sourdough starter is a fermented mixture of flour and water that contains wild yeast and lactic acid bacteria.'

