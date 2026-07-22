# Earshift Codex corpus prompt

Using `rules/phonotactics.yaml` as the controlling rule table, author three
rounds of pronounceable, semantically opaque candidate scripts for `en-US-mae`,
`es-MX-cdmx`, and `pt-BR-sp`. Each script must contain 18–24 invented tokens and
30–42 intended syllables, with two declarative phrases, one internal comma, and
a final period. Use 55–70% content-like tokens and 3–5 recurring invented
function-word fillers for rhythm. Do not use real words, names, abbreviations,
numbers, recognizable productive morphology, or examples copied from the rule
table. Supply token role, intended IPA target, syllable count, stress index, and
the applicable stable rule IDs. The local pinned wordfreq and eSpeak gates—not
the model—make the final real-word and pronounced-homophone decisions.

Round 0 contains 20 candidates per profile. Rounds 1 and 2 contain 10 fallback
candidates per profile. Return the exact `CorpusBundle` schema and record this
prompt and the rule-table checksums in provenance.
