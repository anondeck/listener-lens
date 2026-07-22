import json

from earshift_bakeoff.release_gate import (
    CANDIDATE_FEATURE_FLAGS,
    LOCAL_TOGGLEABLE_CANDIDATE_FLAGS,
    candidate_flag_configuration_errors,
    collect_release_checks,
    release_check,
)


def test_release_gate_has_no_structural_failures() -> None:
    checks = collect_release_checks()

    assert not [check for check in checks if check.status == "FAIL"]
    assert {
        "release-records",
        "third-party-inventory",
        "working-name-boundary",
        "sensitive-and-local-files",
        "listener-rule-provenance",
        "shipping-audio-mode",
        "shipping-bundle",
        "bilingual-product-matrix",
        "disabled-candidate-flags",
        "concept-evidence-boundary",
        "protected-activity-worker",
        "typed-runtime-contract",
        "runtime-security-and-privacy",
        "blind-ratings",
        "shipping-ai-voice-disclosure",
        "video-path",
        "deployment-record",
    } == {check.name for check in checks}
    deployment = next(check for check in checks if check.name == "deployment-record")
    assert deployment.status == "WAIT"
    assert "historical URL is not release evidence" in deployment.detail
    assert release_check(strict=False) == 0
    assert release_check(strict=True) == 4


def test_candidate_flag_release_check_rejects_any_non_false_layer(tmp_path) -> None:
    project = {"candidate_features": {flag: False for flag in CANDIDATE_FEATURE_FLAGS}}
    wrangler = {"vars": {flag: "false" for flag in CANDIDATE_FEATURE_FLAGS}}
    (tmp_path / "worker").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "bakeoff.yaml").write_text(json.dumps(project), encoding="utf-8")
    (tmp_path / "wrangler.runtime-v2.jsonc").write_text(
        json.dumps(wrangler), encoding="utf-8"
    )
    (tmp_path / ".env.example").write_text(
        "".join(f"{flag}=false\n" for flag in CANDIDATE_FEATURE_FLAGS),
        encoding="utf-8",
    )
    (tmp_path / "worker" / "typed-audio.js").write_text(
        "candidateFlagsAreExactlyFalse env?.[flag] === 'false' "
        + " ".join(CANDIDATE_FEATURE_FLAGS),
        encoding="utf-8",
    )
    (tmp_path / "scripts" / "dev-local.mjs").write_text(
        " ".join(
            f"['false', 'true'].includes(candidateFlagEnv.{flag})"
            for flag in LOCAL_TOGGLEABLE_CANDIDATE_FLAGS
        )
        + " kokoroCandidateEnabled && bilingualCandidateEnabled "
        + "candidateFlagEnv[name] !== 'false' "
        + " ".join(CANDIDATE_FEATURE_FLAGS),
        encoding="utf-8",
    )
    assert candidate_flag_configuration_errors(tmp_path) == []

    wrangler["vars"][CANDIDATE_FEATURE_FLAGS[0]] = "FALSE"
    (tmp_path / "wrangler.runtime-v2.jsonc").write_text(
        json.dumps(wrangler), encoding="utf-8"
    )
    errors = candidate_flag_configuration_errors(tmp_path)
    assert any(
        "wrangler.runtime-v2.jsonc KOKORO_ENGLISH_CANDIDATE_ENABLED" in error
        for error in errors
    )
