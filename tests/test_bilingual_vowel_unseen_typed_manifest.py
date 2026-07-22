from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json

from earshift_bakeoff.config import ROOT, stable_json


MANIFEST = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260718-bilingual-vowel-unseen-typed-manifest-v1"
    / "manifest.json"
)


def _load() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _semantic_hash(payload: dict) -> str:
    semantic = dict(payload)
    semantic.pop("record_sha256", None)
    return hashlib.sha256(stable_json(semantic).encode("utf-8")).hexdigest()


def test_unseen_manifest_is_complete_zero_render_and_nonpromotional() -> None:
    manifest = _load()

    assert manifest["record_sha256"] == _semantic_hash(manifest)
    assert manifest["classification"] == (
        "unseen_real_g2p_typed_fixtures_frozen_before_audio"
    )
    assert manifest["production_enabled"] is False
    assert manifest["api_calls_made"] == 0
    assert manifest["audio_renders_made"] == 0
    assert manifest["cell_count"] == 28
    assert manifest["rule_group_count"] == 15
    assert manifest["logical_slot_count"] == len(manifest["slots"]) == 84
    assert manifest["expected_occurrence_count"] == 112
    assert all(row["product_enabled"] is False for row in manifest["slots"])


def test_unseen_manifest_uses_three_distinct_real_words_per_rule() -> None:
    manifest = _load()

    assert len(manifest["selected_words_by_rule"]) == 15
    for contexts in manifest["selected_words_by_rule"].values():
        assert set(contexts) == set(manifest["context_order"])
        assert len(set(contexts.values())) == 3
    assert all(
        row["fixture_spec"]["target_word_canonical_rank"] >= 256
        for row in manifest["slots"]
    )


def test_unseen_manifest_shares_text_across_eligible_voices() -> None:
    manifest = _load()
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in manifest["slots"]:
        grouped[(row["profile_id"], row["rule_id"], row["context"])].append(row)

    for rows in grouped.values():
        assert len({row["fixture_spec"]["text"] for row in rows}) == 1
        assert len({row["fixture_spec"]["target_word"] for row in rows}) == 1
        assert all(row["gate_receipt"]["written_and_espeak_gate_pass"] for row in rows)
        assert all(row["gate_receipt"]["supplemental_phone_gates_pass"] for row in rows)
        assert all(row["gate_receipt"]["model_representable"] for row in rows)
        assert all(row["gate_receipt"]["punctuation_preserved"] for row in rows)
        assert all(row["gate_receipt"]["repeated_word_invariant_pass"] for row in rows)


def test_unseen_manifest_keeps_voice_and_candidate_rung_denominators() -> None:
    manifest = _load()

    assert Counter(row["voice_id"] for row in manifest["candidate_cells"]) == {
        "af_heart": 8,
        "am_michael": 9,
        "pf_dora": 5,
        "pm_alex": 6,
    }
    assert Counter(row["candidate_rung"] for row in manifest["candidate_cells"]) == {
        "adaptive_strength": 6,
        "full_context": 1,
        "v8": 20,
        "word_context": 1,
    }
    assert Counter(row["context"] for row in manifest["slots"]) == {
        "real_g2p_phrase_medial": 28,
        "real_g2p_phrase_final": 28,
        "real_g2p_repeated_target": 28,
    }
