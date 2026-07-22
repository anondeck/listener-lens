from __future__ import annotations

from earshift_bakeoff.en_ae_shape_coverage_v1 import (
    EN_AE_SHAPE_COVERAGE_VERSION,
    characterize,
    classify_ae_word,
)


def test_classifier_matches_strict_shell_contract_exactly() -> None:
    assert classify_ae_word("kˈæt") == {
        "ae_count": 1,
        "blockers": (),
        "strict_supported": True,
    }
    assert classify_ae_word("mˌæp")["strict_supported"] is True


def test_classifier_names_each_structural_blocker() -> None:
    assert classify_ae_word("ɡɹˈæb")["blockers"] == ("onset_cluster",)
    assert classify_ae_word("hˈænd")["blockers"] == ("coda_cluster",)
    assert classify_ae_word("ˈæt")["blockers"] == ("no_onset",)
    assert classify_ae_word("bˈæ")["blockers"] == ("no_coda",)
    assert classify_ae_word("ˈæsk")["blockers"] == ("coda_cluster", "no_onset")
    assert classify_ae_word("hˈæpi")["blockers"] == ("multisyllabic",)
    assert classify_ae_word("æm")["blockers"] == (
        "no_onset",
        "unstressed_target",
    )
    assert classify_ae_word("bˈækpˌæk")["blockers"] == (
        "multiple_targets",
        "multisyllabic",
    )
    assert classify_ae_word("dˈæː")["blockers"] == ("nonconforming_symbols",)
    assert classify_ae_word("kˈot")["ae_count"] == 0


def test_bounded_characterization_is_structurally_sound() -> None:
    record = characterize(limit=1200)
    assert record["version"] == EN_AE_SHAPE_COVERAGE_VERSION
    assert record["limit"] == 1200
    assert record["api_calls_made"] == 0
    assert record["production_enabled"] is False
    assert record["ae_word_count"] == sum(
        row["word_count"] for row in record["buckets"]
    )
    assert record["ae_word_count"] > 0
    shares = [row["freq_share_of_ae"] for row in record["buckets"]]
    assert abs(sum(shares) - 1.0) < 1e-9
    assert all(row["word_count"] > 0 for row in record["buckets"])
    strict_rows = [row for row in record["buckets"] if row["bucket"] == "strict"]
    assert len(strict_rows) == 1
    assert "back" in strict_rows[0]["examples"] or strict_rows[0]["word_count"] > 0
    assert len(record["record_sha256"]) == 64
