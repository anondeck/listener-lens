from __future__ import annotations

import json
import unicodedata

import pytest

from earshift_bakeoff import azure_lens_builder as lane
from earshift_bakeoff.config import ROOT


def test_map_covers_both_frozen_inventories_and_rule_targets() -> None:
    reachability = json.loads(
        (
            ROOT
            / "artifacts"
            / "product-matrix"
            / "20260717-bilingual-g2p-reachability-v1"
            / "results.json"
        ).read_text(encoding="utf-8")
    )
    tables = lane.load_ipa_map()
    by_language = {"en": tables["en-US"], "pt": tables["pt-BR"]}
    for profile in reachability["profiles"]:
        table = by_language[profile["language"]]
        for symbol in profile["source_symbol_counts"]:
            assert symbol in table, f"unmapped source symbol {symbol!r}"
        for rule in profile["rules"]:
            for symbol in (rule["source"], rule["target"]):
                mappable = symbol in table or all(
                    piece in table for piece in symbol
                )
                assert mappable, f"unmapped rule symbol {symbol!r}"


def test_apply_segment_rules_is_single_pass_and_nasal_safe() -> None:
    vowel_rules = [
        {"id": "x.e_i", "source": "e", "target": "i"},
        {"id": "x.i_o", "source": "i", "target": "o"},
    ]
    lens, applied = lane._apply_segment_rules("pele", vowel_rules, [], frozenset())
    assert lens == "pili"
    assert {item.rule_id for item in applied} == {"x.e_i"}, (
        "an emitted target must not re-match as a source"
    )
    chained, chained_applied = lane._apply_segment_rules(
        "pie", vowel_rules, [], frozenset()
    )
    assert chained == "poi"
    assert {item.rule_id for item in chained_applied} == {"x.e_i", "x.i_o"}
    nasal, nasal_applied = lane._apply_segment_rules(
        "pẽ", vowel_rules, [], frozenset()
    )
    assert nasal == "pẽ"
    assert nasal_applied == ()


def test_apply_segment_rules_honours_consonant_context() -> None:
    consonant_rules = [
        {"id": "c.any", "source": "θ", "target": "t", "contexts": ["any"]},
        {"id": "c.tap", "source": "ɾ", "target": "T", "contexts": ["intervocalic"]},
    ]
    vowels = frozenset("ae")
    # θ substitutes anywhere; the intervocalic tap only between vowels.
    lens, applied = lane._apply_segment_rules("aɾaθ", [], consonant_rules, vowels)
    assert lens == "aTat"
    assert {item.rule_id for item in applied} == {"c.tap", "c.any"}
    # a word-final ɾ is not intervocalic, so the tap rule must not fire.
    edge, edge_applied = lane._apply_segment_rules("aɾ", [], consonant_rules, vowels)
    assert edge == "aɾ"
    assert edge_applied == ()


def test_map_symbols_fails_closed_on_unknown_symbol() -> None:
    table = lane.load_ipa_map()["en-US"]
    with pytest.raises(lane.AzureLensBuilderError, match="unmapped"):
        lane._map_symbols("k§t", table, context="test")


def test_en_pair_tags_only_affected_words() -> None:
    pair = lane.build_pair("The cat naps", "en-US-to-pt-BR-listener-v2")
    assert pair["locale"] == "en-US"
    assert pair["voice"] == "en-US-AvaNeural"
    assert pair["affected_word_count"] >= 2
    assert "enpt.ae_eh" in pair["applied_rule_ids"]
    assert "enpt.schwa_reduced_a" in pair["map_neutralized_rule_ids"], (
        "the ə→ɐ rule collapses through the en map and must be reported as "
        "map-neutralized, never as applied"
    )
    assert not set(pair["applied_rule_ids"]) & set(pair["map_neutralized_rule_ids"])
    neutral, lens = pair["ssml_neutral"], pair["ssml_lens"]
    assert neutral.count("<phoneme") == lens.count("<phoneme")
    assert neutral != lens
    assert "æ" in neutral and "ɛ" in lens
    # Only affected words are tagged, on both sides. "The" is now affected
    # because its ð stops to d, so it is tagged where it used to stay plain.
    assert neutral.count("<phoneme") == pair["affected_word_count"]
    assert ">The</phoneme>" in neutral and ">The</phoneme>" in lens
    # Consonants and epenthesis render now (the ð in "The" stops to d; the
    # coda /t/ in cat and /ps/ in naps take a perceptual /i/), so they leave
    # the omitted bucket entirely; only the still-unrendered prosody remains.
    assert "enpt.eth_d" in pair["applied_rule_ids"]
    assert "enpt.illicit_coda_epenthetic_i" in pair["applied_rule_ids"]
    assert not (
        {"enpt.theta_t", "enpt.eth_d", "enpt.illicit_coda_epenthetic_i"}
        & set(pair["omitted_rule_ids"])
    )
    # Every rule family renders now; nothing is left in the omitted bucket.
    # The stress-swap rule is implemented but has no initial-secondary-stress
    # word here, so it reports as context-absent; the en polar rule is an
    # explicit identity and stays out of the changed accounting entirely,
    # exactly like identity vowel rules.
    assert pair["omitted_rule_ids"] == []
    assert "enpt.lexical_stress_initial_bias" in pair["context_absent_rule_ids"]
    assert pair["prosody"] == {"polar_question": False, "contour_applied": False}
    assert pair["api_calls_made"] == 0
    cat = next(row for row in pair["words"] if row["written"].lower() == "cat")
    assert "æ" in cat["source_phone"] and "ɛ" in cat["lens_phone"]


def test_en_th_stopping_renders_both_interdentals() -> None:
    pair = lane.build_pair(
        "My brother thinks", "en-US-to-pt-BR-listener-v2"
    )
    assert {"enpt.eth_d", "enpt.theta_t"} <= set(pair["applied_rule_ids"])
    assert not ({"enpt.eth_d", "enpt.theta_t"} & set(pair["map_neutralized_rule_ids"]))
    brother = next(r for r in pair["words"] if r["written"].lower() == "brother")
    assert "ð" in brother["source_phone"]
    assert "d" in brother["lens_phone"] and "ð" not in brother["lens_phone"]


def test_pt_consonants_palatal_nasal_and_dorsal_r_render() -> None:
    pair = lane.build_pair("A manha corre", "pt-BR-to-en-US-listener-v2")
    assert "pten.palatal_nasal_n" in pair["applied_rule_ids"]  # ɲ → n
    # The dorsal r still reaches the ph string — the map fix that stopped it
    # rendering as ʃ holds — but the pt-BR voice returns byte-identical audio
    # for x and h, so the rule is reported inaudible instead of applied.
    # Fixing the map made the substitution expressible; it could not make the
    # voice pronounce it.
    assert "pten.dorsal_r_h" in pair["renderer_inaudible_rule_ids"]
    assert "pten.dorsal_r_h" not in pair["applied_rule_ids"]
    assert "ʃ" not in pair["ssml_neutral"]
    assert "x" in pair["ssml_neutral"] and "h" in pair["ssml_lens"]


def test_untriggered_consonants_are_context_absent_not_omitted() -> None:
    pair = lane.build_pair("O povo", "pt-BR-to-en-US-listener-v2")
    consonants = {
        "pten.palatal_lateral_yod",
        "pten.dorsal_r_h",
        "pten.palatal_nasal_n",
        "pten.tap_flap",
    }
    assert consonants <= set(pair["context_absent_rule_ids"])
    assert not (consonants & set(pair["omitted_rule_ids"]))


def test_en_epenthesis_inserts_i_after_illegal_codas_only() -> None:
    pair = lane.build_pair("club brother", "en-US-to-pt-BR-listener-v2")
    assert "enpt.illicit_coda_epenthetic_i" in pair["applied_rule_ids"]
    club = next(r for r in pair["words"] if r["written"].lower() == "club")
    # kl is a legal BP onset and must survive; only the final /b/ takes /i/.
    assert club["lens_phone"].startswith("kl")
    assert club["lens_phone"].endswith("i")
    assert club["source_phone"].count("i") + 1 == club["lens_phone"].count("i")
    brother = next(r for r in pair["words"] if r["written"].lower() == "brother")
    # br is a legal onset: no epenthesis splits it.
    assert brother["lens_phone"].startswith("bɹ")
    assert "enpt.illicit_coda_epenthetic_i" not in brother["applied_rule_ids"]


def test_epenthesis_is_lens_only_and_context_absent_when_licit() -> None:
    pair = lane.build_pair("cat", "en-US-to-pt-BR-listener-v2")
    cat = pair["words"][0]
    # The /i/ is a lens-only perception; the neutral coda stays licit /t/.
    assert cat["lens_phone"].endswith("i") and not cat["source_phone"].endswith("i")
    # "brother" renders (ð→d) but has no illegal coda, so the epenthesis rule
    # is context-absent, never omitted.
    licit = lane.build_pair("brother", "en-US-to-pt-BR-listener-v2")
    assert "enpt.illicit_coda_epenthetic_i" in licit["context_absent_rule_ids"]
    assert "enpt.illicit_coda_epenthetic_i" not in licit["omitted_rule_ids"]


def test_stress_swap_is_matched_but_renderer_inaudible() -> None:
    # "understand" carries the exact structure the swap rule cites (initial
    # secondary stress, later primary), so the rule matches — and the stress
    # probe shows every current voice returns byte-identical audio when only
    # the mark moves. The honest report is therefore renderer-inaudible, and
    # the mark must NOT move in the ph string: shipping a difference the
    # voice provably cannot voice is the fail-open shape this lane refuses.
    pair = lane.build_pair("understand", "en-US-to-pt-BR-listener-v2")
    assert "enpt.lexical_stress_initial_bias" in pair["renderer_inaudible_rule_ids"]
    assert "enpt.lexical_stress_initial_bias" not in pair["applied_rule_ids"]
    word = pair["words"][0]
    assert word["source_phone"].index("ˌ") < word["source_phone"].index("ˈ")
    assert word["lens_phone"].index("ˌ") < word["lens_phone"].index("ˈ"), (
        "the mark must stay where the voice will actually say it"
    )


def test_stress_shift_rules_are_accounted_not_silently_skipped() -> None:
    # shift_primary_stress_to_final fell out of the stress-rule filter when it
    # was added (the filter matched one operation literal), so French and
    # Turkish stress rules were never applied AND never reported — absent
    # from every bucket. Every stress rule must land in exactly one bucket.
    pair = lane.build_pair(
        "My brother is basically running.", "en-US-to-fr-FR-listener-v1"
    )
    buckets = (
        set(pair["applied_rule_ids"])
        | set(pair["renderer_inaudible_rule_ids"])
        | set(pair["context_absent_rule_ids"])
        | set(pair["map_neutralized_rule_ids"])
    )
    assert "enfr.stress_final" in buckets
    assert "enfr.stress_final" not in pair["applied_rule_ids"]


def test_pt_polar_question_contour_maps_to_final_fall() -> None:
    pair = lane.build_pair("O povo corre?", "pt-BR-to-en-US-listener-v2")
    assert pair["prosody"] == {"polar_question": True, "contour_applied": True}
    assert "pten.polar_rise_fall_statement" in pair["applied_rule_ids"]
    assert pair["ssml_neutral"].count("?</voice>") == 1
    assert pair["ssml_lens"].count("?") == 0
    assert ".</voice>" in pair["ssml_lens"]


def test_pt_wh_question_keeps_the_rise_on_both_sides() -> None:
    pair = lane.build_pair("Onde o povo corre?", "pt-BR-to-en-US-listener-v2")
    assert pair["prosody"] == {"polar_question": False, "contour_applied": False}
    assert "pten.polar_rise_fall_statement" in pair["context_absent_rule_ids"]
    assert pair["ssml_neutral"].count("?</voice>") == 1
    assert pair["ssml_lens"].count("?</voice>") == 1


def test_en_polar_question_has_no_contour_rule_and_keeps_the_rise() -> None:
    pair = lane.build_pair("Does the cat nap?", "en-US-to-pt-BR-listener-v2")
    assert pair["prosody"] == {"polar_question": True, "contour_applied": False}
    assert "enpt.english_polar_rise_identity" not in pair["applied_rule_ids"]
    assert "enpt.english_polar_rise_identity" not in pair["context_absent_rule_ids"]
    assert pair["ssml_neutral"].count("?</voice>") == 1
    assert pair["ssml_lens"].count("?</voice>") == 1


def test_pt_pair_builds_reciprocal_direction() -> None:
    pair = lane.build_pair("O povo corre", "pt-BR-to-en-US-listener-v2")
    assert pair["locale"] == "pt-BR"
    assert pair["voice"] == "pt-BR-FranciscaNeural"
    assert pair["affected_word_count"] >= 1
    assert pair["ssml_neutral"] != pair["ssml_lens"]
    assert 'xml:lang="pt-BR"' in pair["ssml_neutral"]


def test_unsupported_text_fails_closed() -> None:
    # No mappable vowel shift and no interdental/consonant trigger, so the
    # lane fails closed rather than returning an identical pair.
    with pytest.raises(lane.AzureLensBuilderError, match="no supported changed"):
        lane.build_pair("we see", "en-US-to-pt-BR-listener-v2")


def test_italian_listener_lane_renders_through_the_generic_adapter() -> None:
    pair = lane.build_pair("Il gatto dorme nella casa", "it-IT-to-en-US-listener-v1")
    assert pair["locale"] == "it-IT"
    assert pair["voice"] == "it-IT-ElsaNeural"
    # The trill target reaches the ph string, so the map fix holds, but the
    # it-IT voice renders r and ɹ identically: Azure voices impose their own
    # rhotic. The rule is reported, never counted as heard.
    assert "iten.trill_approximant" in pair["renderer_inaudible_rule_ids"]
    assert "ɹ" in pair["ssml_lens"] and "ɹ" not in pair["ssml_neutral"]
    # The direction still does something audible on its vowels.
    assert {"iten.e_dress", "iten.o_thought"} <= set(pair["applied_rule_ids"])
    assert pair["map_neutralized_rule_ids"] == []
    assert pair["omitted_rule_ids"] == []


def test_italian_palatals_recategorise_for_an_english_listener() -> None:
    pair = lane.build_pair("Mia figlia mangia gli gnocchi", "it-IT-to-en-US-listener-v1")
    applied = set(pair["applied_rule_ids"])
    assert {"iten.palatal_lateral_l", "iten.palatal_nasal_n"} <= applied
    figlia = next(r for r in pair["words"] if r["written"].lower() == "figlia")
    assert "ʎ" in figlia["source_phone"] and "ʎ" not in figlia["lens_phone"]


def test_english_through_italian_renders_on_the_english_voice() -> None:
    pair = lane.build_pair("This thing runs through the house", "en-US-to-it-IT-listener-v1")
    # en->it renders on the source-language (English) voice; the Italian lens
    # targets must be receipted in the en-US map, not the it-IT map.
    assert pair["locale"] == "en-US"
    assert pair["voice"] == "en-US-AvaNeural"
    applied = set(pair["applied_rule_ids"])
    # th-stopping and the vowel merges are audible on the English voice.
    assert {"enit.th_stop_t", "enit.eth_stop_d"} <= applied
    # The approximant->trill reaches the ph string but the en-US voice renders
    # ɹ and r identically, so it is reported rather than counted as heard.
    assert "enit.approximant_trill" in pair["renderer_inaudible_rule_ids"]
    thing = next(r for r in pair["words"] if r["written"].lower() == "thing")
    assert thing["source_phone"].startswith("θ") and thing["lens_phone"].startswith("t")
    assert "ŋ" in thing["source_phone"] and "ŋ" not in thing["lens_phone"]
    # The trill target must actually reach the ph string, not fold back to ɹ.
    assert "r" in pair["ssml_lens"] and "ɹ" not in pair["ssml_lens"]
    assert pair["map_neutralized_rule_ids"] == []
    assert pair["omitted_rule_ids"] == []


def test_english_lens_direction_is_distinct_from_the_portuguese_one() -> None:
    italian = lane.build_pair("The cat naps", "en-US-to-it-IT-listener-v1")
    portuguese = lane.build_pair("The cat naps", "en-US-to-pt-BR-listener-v2")
    # Same English source, same voice, different listener categories, so the
    # lens ph strings must differ between the two directions.
    assert italian["voice"] == portuguese["voice"] == "en-US-AvaNeural"
    assert italian["ssml_lens"] != portuguese["ssml_lens"]


def test_english_through_german_renders_the_iconic_consonants() -> None:
    pair = lane.build_pair("We think that water is warm", "en-US-to-de-DE-listener-v1")
    assert pair["locale"] == "en-US"
    assert pair["voice"] == "en-US-AvaNeural"
    applied = set(pair["applied_rule_ids"])
    # w->v, th->s and th->z are the audible signature German effects.
    assert {"ende.w_v", "ende.th_s", "ende.eth_z"} <= applied
    # The uvular rhotic reaches the ph string but the en-US voice collapses
    # ɹ and ʁ, so it is reported inaudible rather than claimed.
    assert "ende.approximant_uvular" in pair["renderer_inaudible_rule_ids"]
    we = next(r for r in pair["words"] if r["written"].lower() == "we")
    assert we["source_phone"].startswith("w") and we["lens_phone"].startswith("v")
    assert "ʁ" in pair["ssml_lens"] and "ʁ" not in pair["ssml_neutral"]
    assert pair["map_neutralized_rule_ids"] == []


def test_english_diphthongs_are_not_recategorised_for_german() -> None:
    # German parses /aɪ aʊ ɔʏ/ natively; the deriver's cross-diphthong pairings
    # are excluded by design, so "day"/"boat" carry no vowel rule.
    pair = lane.build_pair("They row the boat", "en-US-to-de-DE-listener-v1")
    boat = next(r for r in pair["words"] if r["written"].lower() == "boat")
    assert "oʊ" in boat["source_phone"] or "O" in boat["source_phone"]
    assert not any(rule.startswith("ende.") and "o" in rule for rule in boat["applied_rule_ids"])


def test_map_symbols_compose_step_is_locale_gated_to_de() -> None:
    # espeak emits the ich-laut as precomposed U+00E7; the adapter's NFD
    # normalization hands it on as c + combining cedilla (U+0327), and the
    # de-DE locale's NFC compose step restores U+00E7 before the per-codepoint
    # map walk. Written with escapes so the two codepoints stay unambiguous.
    tables = lane.load_ipa_map()
    de = tables["de-DE"]
    ich_nfd = "ɪç"
    composed = lane._map_symbols(ich_nfd, de, context="t", normalization="NFC")
    # The receipted Azure form is the precomposed codepoint, so assert the
    # codepoints rather than the glyphs: NFD and NFC both *look* like "ɪç".
    assert composed == "ɪç"
    assert "̧" not in composed
    # The default NFD walk no longer fails closed here: bare c and the cedilla
    # became mapped once other locales needed them as cross-inventory lens
    # targets. It now yields the canonically equivalent decomposed form, which
    # is why the locale gate — not the map — is what guarantees the receipted
    # spelling reaches Azure.
    decomposed = lane._map_symbols(ich_nfd, de, context="t")
    assert decomposed == "ɪç"
    assert unicodedata.normalize("NFC", decomposed) == composed
    # The validated pt-BR nasal path keeps its two independent codepoints
    # under the unchanged default walk.
    pt = tables["pt-BR"]
    assert lane._map_symbols("wɐ̃", pt, context="t") == "wɐ̃"
    assert lane._MAP_NORMALIZATION_BY_LOCALE.get("de-DE") == "NFC"
    assert all(
        lane._MAP_NORMALIZATION_BY_LOCALE.get(locale, "NFD") == "NFD"
        for locale in ("en-US", "pt-BR", "it-IT")
    )


def test_supported_profile_ids_includes_the_german_source_direction() -> None:
    # The single source of truth the Italian 422 fix introduced must track
    # the registry, or the service rejects a direction everything else wired.
    assert "de-DE-to-en-US-listener-v1" in lane.supported_profile_ids()


def test_german_listener_renders_the_ich_laut_as_sh() -> None:
    pair = lane.build_pair("Ich denke dass das Wasser warm ist", "de-DE-to-en-US-listener-v1")
    assert pair["locale"] == "de-DE"
    assert pair["voice"] == "de-DE-KatjaNeural"
    applied = set(pair["applied_rule_ids"])
    assert {"deen.ich_laut_sh", "deen.tap_approximant"} <= applied
    # The de-DE voice renders a and ɑ identically, so the low-vowel shift
    # is reported rather than counted; the ich-laut and tap still carry it.
    assert "deen.a_ɑ" in pair["renderer_inaudible_rule_ids"]
    ich = next(r for r in pair["words"] if r["written"] == "Ich")
    # The adapter surface is the NFD c + combining cedilla; the rule source
    # matches that form, and the neutral side maps back to precomposed ç.
    assert ich["source_phone"] == "ɪç"
    assert ich["lens_phone"] == "ɪʃ"
    assert 'ph="ɪç"' in pair["ssml_neutral"]
    assert 'ph="ɪʃ"' in pair["ssml_lens"]
    warm = next(r for r in pair["words"] if r["written"] == "warm")
    assert warm["source_phone"] == "vˈaɾm"
    assert warm["lens_phone"] == "vˈɑɹm"
    assert pair["map_neutralized_rule_ids"] == []
    assert pair["omitted_rule_ids"] == []


def test_german_ach_laut_and_pf_recategorise() -> None:
    pair = lane.build_pair("Auch das Pferd", "de-DE-to-en-US-listener-v1")
    applied = set(pair["applied_rule_ids"])
    assert {"deen.ach_laut_k", "deen.pf_f"} <= applied
    auch = next(r for r in pair["words"] if r["written"] == "Auch")
    assert "x" in auch["source_phone"] and "x" not in auch["lens_phone"]
    pferd = next(r for r in pair["words"] if r["written"] == "Pferd")
    assert pferd["source_phone"].startswith("pf")
    assert pferd["lens_phone"].startswith("f")


def test_german_front_rounded_vowels_recategorise() -> None:
    pair = lane.build_pair("Schön über München", "de-DE-to-en-US-listener-v1")
    applied = set(pair["applied_rule_ids"])
    assert {"deen.ø_ɛ", "deen.y_u", "deen.ʏ_ɪ"} <= applied
    muenchen = next(r for r in pair["words"] if r["written"] == "München")
    # München carries both the lax front rounded vowel and the ich-laut.
    assert muenchen["lens_phone"] == "mˈɪnʃən"
    assert pair["map_neutralized_rule_ids"] == []


def test_german_oy_diphthong_maps_to_the_english_oy() -> None:
    pair = lane.build_pair("Die Leute sind neu", "de-DE-to-en-US-listener-v1")
    assert "deen.ɔø_ɔɪ" in pair["applied_rule_ids"]
    assert "ɔɪ" in pair["ssml_lens"] and "ɔɪ" not in pair["ssml_neutral"]


def test_german_sentence_without_a_trigger_fails_closed() -> None:
    with pytest.raises(lane.AzureLensBuilderError, match="no supported changed"):
        lane.build_pair("Denken", "de-DE-to-en-US-listener-v1")


def test_supported_profile_ids_includes_the_spanish_directions() -> None:
    assert "es-ES-to-en-US-listener-v1" in lane.supported_profile_ids()
    assert "en-US-to-es-ES-listener-v1" in lane.supported_profile_ids()


def test_spanish_compose_gate_is_not_extended_past_de() -> None:
    # The NFC compose step exists for the German ich-laut alone; Spanish has
    # no multi-codepoint phoneme (the inventory walk found no composing
    # bigram), so es-ES stays on the default per-codepoint NFD walk.
    assert lane._MAP_NORMALIZATION_BY_LOCALE.get("es-ES", "NFD") == "NFD"


def test_spanish_map_covers_the_observed_espeak_inventory() -> None:
    # The es-ES table is authored against the real adapter inventory
    # (including the [β ð ɣ] allophones the deriver docstring warns about).
    # Any symbol espeak adds later fails closed at build time, but the
    # common-word inventory must never be unmapped.
    from earshift_bakeoff.azure_source_adapters import EspeakSourceAdapter

    adapter = EspeakSourceAdapter.load("es-ES")
    analysis = adapter.analyze(
        "El perro bebe agua y el niño come jamón en Madrid. "
        "La calle zapato cinco gracias gente México bajo dedo amigo "
        "guerra pingüino queso tierra fuego hay pausa buey reina seis "
        "fútbol parking show jazz apto yo ritmo árbol"
    )
    table = lane.load_ipa_map()["es-ES"]
    missing = sorted(
        {symbol for word in analysis.words for symbol in word.phone if symbol not in table}
    )
    assert missing == []


def test_spanish_listener_renders_the_surface_allophones() -> None:
    pair = lane.build_pair("El perro bebe agua", "es-ES-to-en-US-listener-v1")
    assert pair["locale"] == "es-ES"
    assert pair["voice"] == "es-ES-ElviraNeural"
    applied = set(pair["applied_rule_ids"])
    # The phonemic deriver never sees [β ð ɣ]; the lens files them under the
    # English categories a listener actually hears (Habana -> Havana, amigo).
    assert {"esen.gamma_g", "esen.trill_approximant"} <= applied
    # β and v are byte-identical on this voice; the substitution still
    # reaches the ph string but is never claimed as heard.
    assert "esen.beta_v" in pair["renderer_inaudible_rule_ids"]
    perro = next(r for r in pair["words"] if r["written"] == "perro")
    assert perro["source_phone"] == "pˈero"
    assert perro["lens_phone"] == "pˈɛɹɔ"
    bebe = next(r for r in pair["words"] if r["written"] == "bebe")
    assert "β" in bebe["source_phone"] and "v" in bebe["lens_phone"]
    assert pair["map_neutralized_rule_ids"] == []
    assert pair["omitted_rule_ids"] == []


def test_spanish_jota_and_palatal_nasal_recategorise() -> None:
    pair = lane.build_pair("El niño come jamón", "es-ES-to-en-US-listener-v1")
    applied = set(pair["applied_rule_ids"])
    assert "esen.palatal_nasal_n" in applied
    # The es-ES voice collapses x and h, so the jota recategorisation is
    # expressible but inaudible on this renderer.
    assert "esen.jota_h" in pair["renderer_inaudible_rule_ids"]
    jamon = next(r for r in pair["words"] if r["written"] == "jamón")
    assert jamon["source_phone"].startswith("x") and jamon["lens_phone"].startswith("h")


def test_spanish_tap_and_eth_carry_no_rule_by_design() -> None:
    # English owns /ð/ and owns the flap allophonically, so the lens leaves
    # them alone; the a->ɑ rule is what shifts the word. "Madrid" alone can no
    # longer be the fixture: a->ɑ is inaudible on the Spanish voice, so a
    # sentence containing only that rule now fails closed rather than
    # rendering two identical takes. The palatal words carry the audible
    # rules that let the pair build, without changing what is asserted here.
    pair = lane.build_pair("Madrid llave año", "es-ES-to-en-US-listener-v1")
    madrid = pair["words"][0]
    assert madrid["applied_rule_ids"] == ["esen.a_aa"]
    assert "ð" in madrid["source_phone"] and "ð" in madrid["lens_phone"]
    assert "ɾ" in madrid["source_phone"] and "ɾ" in madrid["lens_phone"]
    # And the rule that shifted it is reported as inaudible, not applied.
    assert "esen.a_aa" in pair["renderer_inaudible_rule_ids"]


def test_a_direction_whose_rules_all_collapse_fails_closed() -> None:
    # Every matched rule inaudible means two identical takes. The ph-based
    # guard passes that case (the strings do differ), so it used to reach the
    # Worker as a bare contract error — differing SSML, zero affected words,
    # empty applied list. It must refuse here instead, name the reason, and
    # spend no Azure calls.
    with pytest.raises(lane.AzureLensBuilderError, match="no audible"):
        lane.build_pair("Madrid", "es-ES-to-en-US-listener-v1")


def test_english_through_spanish_renders_the_signature_accent() -> None:
    pair = lane.build_pair("I have five red hats", "en-US-to-es-ES-listener-v1")
    assert pair["locale"] == "en-US"
    assert pair["voice"] == "en-US-AvaNeural"
    applied = set(pair["applied_rule_ids"])
    assert {"enes.v_b", "enes.ae_a"} <= applied
    # h->x and ɹ->r reach the ph string but the en-US voice renders both
    # pairs identically, so they are reported inaudible.
    assert {"enes.h_x", "enes.approximant_trill"} <= set(
        pair["renderer_inaudible_rule_ids"]
    )
    have = next(r for r in pair["words"] if r["written"] == "have")
    assert have["source_phone"] == "hæv"
    assert have["lens_phone"] == "xab"
    # The aɪ diphthong parses natively for a Spanish listener: five keeps it.
    five = next(r for r in pair["words"] if r["written"] == "five")
    assert five["applied_rule_ids"] == ["enes.v_b"]
    assert pair["map_neutralized_rule_ids"] == []


def test_english_goat_monophthongises_but_face_is_native_to_spanish() -> None:
    pair = lane.build_pair("They row the boat", "en-US-to-es-ES-listener-v1")
    boat = next(r for r in pair["words"] if r["written"] == "boat")
    assert boat["lens_phone"] == "bˈot"
    they = next(r for r in pair["words"] if r["written"] == "They")
    # ð->d fires, but the FACE vowel carries no rule: Spanish owns /ej/.
    assert they["applied_rule_ids"] == ["enes.eth_d"]
    assert "A" in they["lens_phone"]


def test_english_z_and_interdental_for_a_spanish_listener() -> None:
    pair = lane.build_pair("This zoo", "en-US-to-es-ES-listener-v1")
    applied = set(pair["applied_rule_ids"])
    assert {"enes.eth_d", "enes.z_s"} <= applied
    # ɪ->i is silent on the en-US voice.
    assert "enes.ih_i" in pair["renderer_inaudible_rule_ids"]
    zoo = next(r for r in pair["words"] if r["written"] == "zoo")
    assert zoo["lens_phone"] == "sˈu"


def test_spanish_sentence_without_a_trigger_fails_closed() -> None:
    with pytest.raises(lane.AzureLensBuilderError, match="no supported changed"):
        lane.build_pair("Sí sí", "es-ES-to-en-US-listener-v1")


def test_spanish_prothesis_repairs_word_initial_s_clusters() -> None:
    pair = lane.build_pair("I study Spanish at school", "en-US-to-es-ES-listener-v1")
    assert "enes.prothetic_e" in pair["applied_rule_ids"]
    school = next(r for r in pair["words"] if r["written"].lower() == "school")
    assert school["lens_phone"].startswith("e") and not school["source_phone"].startswith("e")
    # /s/ followed by a vowel is a legal Spanish onset and must not prothesise.
    legal = lane.build_pair("The sun is warm", "en-US-to-es-ES-listener-v1")
    sun = next(r for r in legal["words"] if r["written"].lower() == "sun")
    assert "enes.prothetic_e" not in sun["applied_rule_ids"]


def test_italian_paragoge_resolves_word_final_consonants() -> None:
    pair = lane.build_pair("The cat sat", "en-US-to-it-IT-listener-v1")
    assert "enit.paragogic_e" in pair["applied_rule_ids"]
    cat = next(r for r in pair["words"] if r["written"].lower() == "cat")
    assert cat["source_phone"].endswith("t") and cat["lens_phone"].endswith("e")


def test_german_glottal_onset_covers_every_vowel_initial_word() -> None:
    pair = lane.build_pair("An apple I ate out east", "en-US-to-de-DE-listener-v1")
    # Reduced vowels and the Misaki diphthong shorthand must all trigger it.
    for word in pair["words"]:
        assert "ende.glottal_onset" in word["applied_rule_ids"], word["written"]
        assert word["lens_phone"].lstrip("ˈˌ").startswith("ʔ")
    consonantal = lane.build_pair("We think", "en-US-to-de-DE-listener-v1")
    assert "ende.glottal_onset" not in consonantal["applied_rule_ids"]


def test_every_direction_recategorises_at_least_one_segment() -> None:
    """Parity check: no direction may be an inaudible no-op.

    Segment recategorisation is the lens, and every direction must carry some.
    The structural and prosodic families deliberately are NOT required: they
    describe repairs a listener language actually performs, and not every
    language performs one. Russian stress is lexically mobile, Greek has no
    epenthesis, Romanian neither devoices finally nor fixes stress. Asserting
    all four families everywhere would only be satisfiable by inventing rules
    for those languages, which is precisely the fabrication the lane's
    evidence policy forbids — and a fabricated rule passes a parity check
    exactly as well as a real one.
    """

    profiles = lane.load_azure_profiles()
    assert len(profiles) > 800, "the generated matrix should be loaded"
    for profile_id, profile in profiles.items():
        segments = profile["vowel_rules"] + profile.get("consonant_rules", [])
        assert segments, f"{profile_id} recategorises nothing"


def test_curated_directions_keep_their_hand_authored_families() -> None:
    """The curated registry must not be shadowed by its generated twin.

    Both registries key the same directions by the same ids on purpose. The
    curated entries carry hand-authored structural and prosodic rules plus
    per-symbol acceptance receipts, so they have to win the collision; loading
    them second is what guarantees that. This pins the ordering, because the
    failure mode is silent — a generated baseline would still render, just
    without the curation.
    """

    profiles = lane.load_azure_profiles()
    for profile_id in (
        "en-US-to-it-IT-listener-v1",
        "en-US-to-de-DE-listener-v1",
        "en-US-to-es-ES-listener-v1",
    ):
        profile = profiles[profile_id]
        assert profile["insertion_rules"], profile_id
        assert profile["prosody_rules"], profile_id


def test_reverse_directions_flatten_the_polar_question_rise() -> None:
    for text, profile_id in [
        ("Il gatto dorme?", "it-IT-to-en-US-listener-v1"),
        ("El perro bebe agua?", "es-ES-to-en-US-listener-v1"),
        ("Ist das Wasser warm?", "de-DE-to-en-US-listener-v1"),
    ]:
        pair = lane.build_pair(text, profile_id)
        assert pair["prosody"] == {"polar_question": True, "contour_applied": True}
        assert pair["ssml_neutral"].count("?</voice>") == 1
        assert pair["ssml_lens"].count("?") == 0


def test_unknown_profile_fails_closed() -> None:
    with pytest.raises(lane.AzureLensBuilderError, match="unsupported profile"):
        lane.build_pair("hello", "xx-XX-to-yy-YY-listener-v1")


def test_inaudible_rules_are_reported_not_counted_as_applied() -> None:
    # Azure's Spanish voice renders /a/ and /ɑ/, /e/ and /ɛ/, /o/ and /ɔ/
    # byte-identically: the map keeps them distinct, the SSML differs, and the
    # listener hears nothing. probe_rule_distinctness_v1 receipts that, and the
    # lane must report it rather than claim the shift happened.
    pair = lane.build_pair("padre madre gato", "es-ES-to-en-US-listener-v1")
    inaudible = set(pair["renderer_inaudible_rule_ids"])
    applied = set(pair["applied_rule_ids"])
    # The curated profile carries no stamped verdict, so this also proves the
    # receipt is looked up by phone pair rather than read off the rule.
    assert {"esen.a_aa", "esen.e_dress", "esen.o_thought"} <= inaudible
    # The two buckets are disjoint by construction.
    assert not (inaudible & applied)
    # A word carrying only collapsed pairs is not counted as shifted, because
    # the ph string differs while the audio does not.
    assert pair["affected_word_count"] == sum(
        1
        for row in pair["words"]
        if set(row["applied_rule_ids"]) & applied
    )


def test_audible_consonants_still_apply_alongside_inaudible_vowels() -> None:
    # The Spanish vowel lens is silent on this voice, but the palatal and
    # trill recategorisations are not, so the direction still does something.
    pair = lane.build_pair("gente llave carro", "es-ES-to-en-US-listener-v1")
    applied = set(pair["applied_rule_ids"])
    assert applied, "the direction must not be entirely silent"
    assert not (applied & set(pair["renderer_inaudible_rule_ids"]))


def test_thin_directions_are_suppressed_from_the_supported_set() -> None:
    # es-ES -> ca-ES survives on two audible consonant rules and a stress
    # swap. Catalan is close enough to Spanish that there is nothing else to
    # hear, so the direction is withheld rather than shipped as a near-null
    # pair the listener would read as a bug.
    suppressed = lane.suppressed_profile_ids()
    assert "es-ES-to-ca-ES-listener-v1" in suppressed
    assert "es-ES-to-ca-ES-listener-v1" not in lane.supported_profile_ids()
    # The reverse direction is a real lens and must survive: thinness is a
    # property of the pair, not of Catalan.
    assert "ca-ES-to-es-ES-listener-v1" in lane.supported_profile_ids()


def test_suppression_never_reaches_the_curated_or_frozen_profiles() -> None:
    # Every hand-authored direction clears the threshold on its own evidence;
    # none is protected by an exception, so this asserts the real margin.
    for profile_id in (
        "en-US-to-pt-BR-listener-v2",
        "pt-BR-to-en-US-listener-v2",
        "en-US-to-it-IT-listener-v1",
        "it-IT-to-en-US-listener-v1",
        "en-US-to-de-DE-listener-v1",
        "de-DE-to-en-US-listener-v1",
        "en-US-to-es-ES-listener-v1",
        "es-ES-to-en-US-listener-v1",
    ):
        assert profile_id in lane.supported_profile_ids()
        assert profile_id not in lane.suppressed_profile_ids()


def test_building_a_suppressed_direction_fails_closed() -> None:
    # Request validation rejects these first, but the builder gates too: a
    # suppressed direction rendering two near-identical takes is worse than an
    # error, because nothing in the output would report the problem.
    with pytest.raises(lane.AzureLensBuilderError) as excinfo:
        lane.build_pair("hola", "es-ES-to-ca-ES-listener-v1")
    assert "suppressed" in str(excinfo.value)


def test_every_source_language_keeps_at_least_one_direction() -> None:
    # The reason suppression is done per-direction rather than per-language:
    # no language loses its whole lens, so all 30 stay on the menu.
    supported = lane.supported_profile_ids()
    sources = {
        profile["source_locale"]
        for profile_id, profile in lane.load_azure_profiles().items()
        if profile_id in supported and profile.get("source_locale")
    }
    assert len(sources) == 30


def test_stress_only_directions_do_not_clear_the_threshold() -> None:
    # Prosody rules are excluded from the count deliberately. They have no
    # distinctness receipts, and a direction whose only effect is a moved
    # accent has not re-pronounced anything through another sound system.
    profile = {
        "id": "synthetic",
        "source_locale": "es-ES",
        "vowel_rules": [],
        "consonant_rules": [],
        "insertion_rules": [],
        "prosody_rules": [
            {"id": "x.stress", "operation": "swap_primary_and_initial_secondary_stress"}
        ],
    }
    assert lane.audible_rule_count(profile) == 0


def test_french_directions_are_supported_both_ways() -> None:
    supported = lane.supported_profile_ids()
    assert "en-US-to-fr-FR-listener-v1" in supported
    assert "fr-FR-to-en-US-listener-v1" in supported


def test_french_lens_applies_the_hand_corrected_vowel_targets() -> None:
    # The deriver sent English /æ/ to /e/ and /ʊ/ to /ɔ/, which would render
    # "cat" as "ket" and mis-height "book". Curation moved them to the French
    # categories listeners actually use, and curation must win over the matrix.
    profile = lane.load_azure_profiles()["en-US-to-fr-FR-listener-v1"]
    targets = {rule["id"]: rule["target"] for rule in profile["vowel_rules"]}
    assert targets["enfr.trap_a"] == "a"
    assert targets["enfr.foot_u"] == "u"


def test_french_listener_unrounds_the_two_front_rounded_vowels_distinctly() -> None:
    # The deriver merged /ø/ and /œ/ onto /ɛ/. An English listener does not
    # have those two French categories, but unrounding them yields different
    # vowels, so collapsing both would invent a merger.
    profile = lane.load_azure_profiles()["fr-FR-to-en-US-listener-v1"]
    targets = {rule["id"]: rule["target"] for rule in profile["vowel_rules"]}
    assert targets["fren.eu_e"] == "e"
    assert targets["fren.oeu_eh"] == "ɛ"
    assert targets["fren.eu_e"] != targets["fren.oeu_eh"]


def test_french_front_rounded_vowel_is_the_audible_signature() -> None:
    # tu -> "too" is the effect this direction exists to show, and unlike the
    # uvular R it survives the renderer.
    pair = lane.build_pair("Le petit chien du sud regarde deux fleurs bleues.",
                           "fr-FR-to-en-US-listener-v1")
    assert "fren.front_rounded_u" in pair["applied_rule_ids"]
    assert "fren.front_rounded_u" not in pair["renderer_inaudible_rule_ids"]


def test_both_french_rhotic_rules_are_reported_inaudible_not_applied() -> None:
    # The uvular R is the signature of the pair in both directions and the
    # renderer collapses it each way. Keeping the rule and labelling it makes
    # the absence visible instead of letting the lane claim a shift it cannot
    # produce.
    fren = lane.build_pair("Le rat regarde la rue.", "fr-FR-to-en-US-listener-v1")
    assert "fren.rhotic_approximant" in fren["renderer_inaudible_rule_ids"]
    assert "fren.rhotic_approximant" not in fren["applied_rule_ids"]
    enfr = lane.build_pair("The red car runs.", "en-US-to-fr-FR-listener-v1")
    assert "enfr.rhotic_uvular" in enfr["renderer_inaudible_rule_ids"]
    assert "enfr.rhotic_uvular" not in enfr["applied_rule_ids"]


def test_clitic_binding_falls_back_to_per_word_alignment() -> None:
    # espeak binds a clitic to its host: Slovene "V kožuščku" comes back as
    # one group ukɔʒˈuːʃʧku, so the phrase failed the one-group-per-word check
    # and the direction rendered nothing. Since most Slavic sentences carry a
    # monosyllabic preposition, three locales were unusable on real input.
    analysis = lane._adapter_for("sl-SI", None).analyze(
        "V kožuščku hudobnega fanta stopiclja mizar"
    )
    assert len(analysis.words) == 6
    assert analysis.words[0].source == "V"
    assert analysis.words[0].phone


def test_ordinary_words_do_not_fail_closed_on_map_gaps() -> None:
    # The inventory behind the IPA map was discovered from ~28 hand-picked
    # words per locale, so phones living in ordinary-but-unlisted words never
    # reached the map: "llibre" needs /ʎ/-to-/ʒ/, "Cocuk" needs /ʤ/.
    import unicodedata

    maps = lane.load_ipa_map()
    for locale, text in (
        ("ca-ES", "El noi llegeix un llibre."),
        ("tr-TR", "Cocuk evde kitap okuyor."),
        ("ro-RO", "Fata are o carte veche."),
        ("es-MX", "El perro bebe agua fria."),
    ):
        analysis = lane._adapter_for(locale, None).analyze(text)
        unmapped = {
            char
            for word in analysis.words
            for char in unicodedata.normalize("NFD", word.phone or "")
            if char.strip() and char not in maps[locale]
        }
        assert not unmapped, f"{locale} cannot render {text!r}: {sorted(unmapped)}"


def test_espeak_shorthand_and_artifacts_never_reach_a_ph_attribute() -> None:
    # Q is Misaki's GB diphthong shorthand, not IPA, and the Russian voice
    # emits a bare quote after some /u/ vowels. Both were mapped to themselves
    # in every locale that emits them, and Azure rejects both outright.
    maps = lane.load_ipa_map()
    for locale in ("nl-NL", "sv-SE", "ro-RO"):
        assert maps[locale]["Q"]["azure_ipa"] == "əʊ"
    assert maps["ru-RU"]['"']["azure_ipa"] == ""
    assert maps["ru-RU"]['"']["fidelity"] == "drop"


def test_bare_integers_are_spelled_out_for_the_misaki_locales() -> None:
    # The 28 espeak locales expand digits natively; the English and Portuguese
    # adapters gate non-word tokens instead, so "I have 7 cats" failed closed.
    # Spelling integers out routes them through the same verified G2P path as
    # any other word — nothing is guessed.
    pair = lane.build_pair("I have 7 cats and 2 dogs.", "en-US-to-pt-BR-listener-v2")
    assert "seven" in pair["normalized_text"]
    assert pair["applied_rule_ids"]
    pair = lane.build_pair("Eu tenho 7 gatos.", "pt-BR-to-en-US-listener-v2")
    assert "sete" in pair["normalized_text"]


def test_number_spellers_match_known_forms() -> None:
    assert lane._en_number_words(115) == "one hundred fifteen"
    assert lane._en_number_words(250000) == "two hundred fifty thousand"
    assert lane._pt_number_words(100) == "cem"
    assert lane._pt_number_words(101) == "cento e um"
    assert lane._pt_number_words(2024) == "dois mil e vinte e quatro"
    assert lane._pt_number_words(1100) == "mil e cem"


def test_english_diacritics_fold_but_no_other_locale_is_touched() -> None:
    # English marks are decoration (naïve -> naive); everywhere else they are
    # phonemic and must never be stripped.
    pair = lane.build_pair("My naïve fiancé loves café au lait.",
                           "en-US-to-pt-BR-listener-v2")
    assert "naive" in pair["normalized_text"]
    assert lane._normalize_source_text("de-DE", "Ich habe 7 schöne Katzen.") == \
        "Ich habe 7 schöne Katzen."
    assert lane._normalize_source_text("pt-BR", "ação é você") == "ação é você"


def test_hard_numeric_tokens_still_fail_closed_and_name_the_token() -> None:
    # "3.14", "1,000", "2nd" are not bare integers; guessing their reading
    # would be unverified audio, so they refuse — but the refusal now names
    # the token so the user knows exactly what to rewrite.
    with pytest.raises(Exception) as excinfo:
        lane.build_pair("Pi is 3.14 exactly.", "en-US-to-pt-BR-listener-v2")
    assert "3.14" in str(excinfo.value)


def test_curated_listener_facts_propagate_across_sources() -> None:
    # Assimilation is a property of the listener: the curated en->it decision
    # that Italian listeners stop /θ/ to /t/ must hold when the θ arrives
    # from Greek, a direction no human ever edited.
    profiles = lane.load_azure_profiles()
    greek = profiles["el-GR-to-it-IT-listener-v1"]
    theta = [r for r in greek["consonant_rules"] if r.get("source") == "θ"]
    assert theta and theta[0]["target"] == "t"
    assert theta[0]["evidence_tier"] == "propagated_from_curated_pair"


def test_conflicting_donor_facts_are_withheld_not_averaged() -> None:
    # English listeners hear Spanish /x/ as h (jalapeño) but German /x/ as k
    # (Bach): genuinely source-dependent, so no propagated fact may exist and
    # the Russian direction keeps its derived tier for /x/.
    profiles = lane.load_azure_profiles()
    russian = profiles["ru-RU-to-en-US-listener-v1"]
    x_rules = [r for r in russian["consonant_rules"] if r.get("source") == "x"]
    assert x_rules
    assert x_rules[0]["evidence_tier"] != "propagated_from_curated_pair"


def test_wh_questions_are_not_polar_in_any_contour_locale() -> None:
    # The contour rule is scoped to yes/no questions, but the wh-word list was
    # English and Portuguese only, so every non-English "?" that did not open
    # with an English wh-word was classified polar — "Perché corri?",
    # "Warum läufst du?" and "¿Dónde está la casa?" all took a contour the
    # rule does not license. Each contour locale needs its own openers.
    wh = [
        ("Perché corri?", "it-IT"),
        ("Warum läufst du?", "de-DE"),
        ("¿Dónde está la casa?", "es-ES"),
        ("Onde o povo corre?", "pt-BR"),
        ("Why is it warm?", "en-US"),
    ]
    polar = [
        ("Corri veloce?", "it-IT"),
        ("Läufst du schnell?", "de-DE"),
        ("¿Tienes la llave?", "es-ES"),
        ("O povo corre?", "pt-BR"),
        ("Is it warm?", "en-US"),
    ]
    for text, locale in wh:
        assert not lane._is_polar_question(text, locale), text
    for text, locale in polar:
        assert lane._is_polar_question(text, locale), text


def test_polar_detection_fails_closed_on_an_unlisted_locale() -> None:
    # Without a locale's interrogative openers there is no way to tell a
    # wh-question from a polar one. Guessing polar would apply an unlicensed
    # contour, so an unlisted locale claims nothing.
    assert lane._is_polar_question("Je to teplé?", "cs-CZ") is False
    assert lane._is_polar_question("Je to teplé?", None) is False


def test_spanish_inverted_punctuation_renders() -> None:
    # espeak echoes sentence punctuation into the phone string, so the
    # inverted mark bound to the first word and failed as an unmapped adapter
    # symbol — every Spanish question written the way Spanish writes questions
    # was unrenderable.
    pair = lane.build_pair("¿Tienes la llave?", "es-ES-to-en-US-listener-v1")
    assert pair["prosody"]["polar_question"] is True
    assert "¿" not in pair["ssml_neutral"] and "¿" not in pair["ssml_lens"]
    assert lane.build_pair("¡Hola año!", "es-ES-to-en-US-listener-v1")


def test_gujarati_retroflex_lateral_is_repaired_not_read_as_r() -> None:
    # espeak-gu emits ળ as "r." — an r plus a syllable separator — so શાળા
    # read shaaraa on both sides of the pair, the neutral track included.
    # The separator is what makes it recoverable: the real Gujarati rhotic ર
    # is written ɾ and never a bare r, so "r." identifies ળ unambiguously.
    from earshift_bakeoff.azure_source_adapters import EspeakSourceAdapter

    adapter = EspeakSourceAdapter.load("gu-IN")
    school = adapter.analyze("શાળા").words[0].phone
    assert "ɭ" in school and "r" not in school
    # The plain rhotic must be untouched by the repair.
    house = adapter.analyze("ઘર").words[0].phone
    assert "ɾ" in house and "ɭ" not in house


def test_declared_repairs_reach_discovery_and_the_map() -> None:
    # The repair lives in the runtime adapter, but discovery builds the IPA
    # map from its own phonemisation. If the two disagree the map is built for
    # symbols the lane no longer emits, and the ones it now does emit go
    # unmapped and fail closed — trading a wrong sound for a hard error.
    import json

    from earshift_bakeoff.azure_source_adapters import _PHONE_REPAIRS

    symbols = json.loads(
        (
            ROOT / "artifacts" / "lens-surface-symbols-v1" / "symbols.json"
        ).read_text(encoding="utf-8")
    )["locales"]
    table = lane.load_ipa_map()
    for locale, repairs in _PHONE_REPAIRS.items():
        for _, right in repairs:
            assert right in symbols[locale]["symbols"], (locale, right)
            assert right in table[locale], (locale, right)

def test_listener_voices_agree_with_the_scripts_side_map() -> None:
    # LISTENER_VOICES is duplicated into runtime source because the deploy
    # container cannot import the build-time scripts. This is the agreement
    # check that duplication comment promises: a voice drifting between the
    # two maps would silently pin the speaker track to a different voice than
    # the one the probes receipted.
    import sys

    sys.path.insert(0, str(lane.ROOT / "scripts"))
    from lens_language_data_v1 import AZURE_VOICE

    assert lane.LISTENER_VOICES == AZURE_VOICE


def test_build_pair_emits_a_rule_free_speaker_track() -> None:
    # The speaker track is production, not perception: the listener's own
    # voice reading the raw text. No phoneme tags may appear — a leaked ph
    # attribute would smuggle lens rules into a clip labeled rule-free — and
    # it must name the listener's voice, not the source's.
    pair = lane.build_pair("The cat naps.", "en-US-to-de-DE-listener-v1")
    assert pair["listener_locale"] == "de-DE"
    # German is one of the four overrides: Katja reads English near-natively,
    # so the track uses Klarissa instead.
    assert pair["speaker_voice"] == "de-DE-KlarissaNeural"
    ssml = pair["ssml_speaker"]
    assert "<phoneme" not in ssml
    assert 'xml:lang="de-DE"' in ssml
    assert "de-DE-KlarissaNeural" in ssml
    assert "The cat naps" in ssml
    assert pair["voice"] != pair["speaker_voice"]


def test_speaker_track_overrides_only_the_four_audited_locales() -> None:
    # The override table is the reason this track ships at thirty languages
    # instead of twenty-five. It must override exactly the locales whose
    # pinned voice failed the audition, and leave every other locale — and
    # every lens voice — alone.
    assert set(lane.C_TRACK_VOICE_OVERRIDES) == {"de-DE", "es-MX", "it-IT", "nl-NL"}
    for locale, voice in lane.C_TRACK_VOICE_OVERRIDES.items():
        assert voice != lane.LISTENER_VOICES[locale], locale
        assert voice.startswith(locale + "-"), locale
    # Swedish was the false alarm; it keeps its pinned voice.
    assert lane.speaker_voice_for("sv-SE") == lane.LISTENER_VOICES["sv-SE"]
    # Everything unlisted falls through untouched.
    for locale, voice in lane.LISTENER_VOICES.items():
        if locale not in lane.C_TRACK_VOICE_OVERRIDES:
            assert lane.speaker_voice_for(locale) == voice, locale


def test_speaker_track_overrides_do_not_touch_the_lens_voices() -> None:
    # A receipt is only evidence for the voice it was taken on. If an override
    # leaked into the lens's voice table it would invalidate that locale's
    # per-symbol acceptance and rule-distinctness receipts silently.
    for locale, voice in lane.C_TRACK_VOICE_OVERRIDES.items():
        pair = lane.build_pair("The cat naps.", f"en-US-to-{locale}-listener-v1")
        assert pair["voice"] == lane.DEFAULT_VOICES.get("en-US", pair["voice"])
        assert voice not in pair["ssml_neutral"]
        assert voice not in pair["ssml_lens"]
        assert voice in pair["ssml_speaker"]


def test_frozen_pair_speaker_track_uses_the_partner_locale() -> None:
    # The frozen en<->pt profiles predate listener_locale, so the speaker
    # track falls back to the partner locale rather than crashing or reading
    # the text in the source's own voice.
    pair = lane.build_pair("The cat naps.", "en-US-to-pt-BR-listener-v2")
    assert pair["listener_locale"] == "pt-BR"
    assert pair["speaker_voice"] == "pt-BR-FranciscaNeural"
