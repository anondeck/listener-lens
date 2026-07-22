from __future__ import annotations

import hashlib
import json
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from earshift_bakeoff.config import sha256_json
from earshift_bakeoff.util import sha256_file


ROOT = Path(__file__).resolve().parents[1]
HUB = ROOT / "artifacts" / "review-hub" / "20260717-human-review-queue-v1"
VOICE_ROOT = (
    ROOT / "artifacts" / "voice-screen" / "20260717-kokoro-bilingual-voice-screen-v1"
)
TRACK_D_ROOT = ROOT / "artifacts" / "research" / "20260717-ptbr-to-ae-listener-lens-v1"

EXPECTED_CHAIN_IDS = {
    "track-a-frozen-replication-v1",
    "track-a-typed-diagnostic-v1",
    "track-a-fresh-confirmation-v1",
    "track-b-en-voice-screen-v1",
    "track-b-ptbr-voice-screen-v1",
    "track-c-pt-native-index-v1",
    "track-c-ptbr-g2p-characterization-v1",
    "track-c-ptbr-renderer-qc-v1",
    "track-d-reciprocal-feasibility-v1",
    "track-e-capability-matrix-v1",
    "track-f-container-benchmark-v1",
    "track-f-browser-onnx-audit-v1",
    "track-g-docs-review-hub-v1",
}
QUEUE_CHAIN_IDS = [
    "track-b-en-voice-screen-v1",
    "track-b-ptbr-voice-screen-v1",
    "track-c-ptbr-renderer-qc-v1",
    "track-d-reciprocal-feasibility-v1",
]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ledger() -> dict[str, Any]:
    return _load(HUB / "status-ledger.json")


def _queue() -> dict[str, Any]:
    return _load(HUB / "queue.json")


def _by_chain(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {row["chain_id"]: row for row in value[key]}


def _assert_safe_repo_path(value: str) -> Path:
    path = Path(value)
    assert not path.is_absolute()
    assert ".." not in path.parts
    resolved = (ROOT / path).resolve()
    assert resolved.is_relative_to(ROOT.resolve())
    return resolved


def _assert_frozen_binding_or_append_only_devlog(
    ledger: dict[str, Any], binding: dict[str, str]
) -> None:
    path = _assert_safe_repo_path(binding["path"])
    assert path.is_file()
    if binding["path"] != "DEVLOG.md":
        assert sha256_file(path) == binding["sha256"]
        return

    # DEVLOG.md is explicitly append-only. The frozen ledger binds its exact
    # repository-head snapshot; later entries may follow that byte-identical
    # prefix without rewriting the historical record.
    completed = subprocess.run(
        [
            "git",
            "show",
            f"{ledger['generated_from_repository_head']}:DEVLOG.md",
        ],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    historical = completed.stdout
    assert hashlib.sha256(historical).hexdigest() == binding["sha256"]
    assert path.read_bytes().startswith(historical)


class _HubHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.current_chain: str | None = None
        self.articles: list[dict[str, str]] = []
        self.anchors: list[tuple[str | None, dict[str, str]]] = []
        self.meta: list[dict[str, str]] = []
        self.all_attributes: list[tuple[str, str]] = []
        self.text: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        values = {key: value or "" for key, value in attrs}
        self.all_attributes.extend(values.items())
        if tag == "article" and "data-chain-id" in values:
            self.current_chain = values["data-chain-id"]
            self.articles.append(values)
        elif tag == "a":
            self.anchors.append((self.current_chain, values))
        elif tag == "meta":
            self.meta.append(values)

    def handle_endtag(self, tag: str) -> None:
        if tag == "article":
            self.current_chain = None

    def handle_data(self, data: str) -> None:
        self.text.append(data)


def test_schemas_validate_and_cross_file_hash_bindings_are_exact() -> None:
    ledger_schema = _load(HUB / "status-ledger.schema.json")
    queue_schema = _load(HUB / "queue.schema.json")
    ledger = _ledger()
    queue = _queue()

    Draft202012Validator.check_schema(ledger_schema)
    Draft202012Validator(ledger_schema).validate(ledger)
    Draft202012Validator.check_schema(queue_schema)
    Draft202012Validator(queue_schema).validate(queue)

    assert ledger["schema_file_sha256"] == sha256_file(
        HUB / "status-ledger.schema.json"
    )
    assert queue["schema_file_sha256"] == sha256_file(HUB / "queue.schema.json")
    assert queue["status_ledger_file_sha256"] == sha256_file(HUB / "status-ledger.json")
    assert (
        ledger["generated_from_repository_head"]
        == queue["generated_from_repository_head"]
    )


def test_ledger_has_exact_separate_chains_and_no_unlocked_decision() -> None:
    ledger = _ledger()
    chain_ids = [row["chain_id"] for row in ledger["chains"]]

    assert len(chain_ids) == len(set(chain_ids)) == 13
    assert set(chain_ids) == EXPECTED_CHAIN_IDS
    assert all(row["decision_unlocked"] is False for row in ledger["chains"])
    assert all(
        row["promotion_status"]
        in {
            "prohibited",
            "blocked_pending_human_review",
            "blocked_prerequisite",
            "not_applicable",
        }
        for row in ledger["chains"]
    )
    assert ledger["review_policy"] == {
        "eligible_links_are_whitelisted": True,
        "failed_or_inconclusive_chains_are_ledger_only": True,
        "private_review_material_is_excluded": True,
        "human_review_cannot_enable_candidate_flags": True,
        "trial_data_combined_across_chains": False,
    }


def test_every_declared_supporting_file_and_principal_hash_is_bound() -> None:
    ledger = _ledger()

    for chain in ledger["chains"]:
        bound_hashes: set[str] = set()
        for binding in chain["supporting_files"]:
            _assert_frozen_binding_or_append_only_devlog(ledger, binding)
            bound_hashes.add(binding["sha256"])

        if chain["protocol_file_sha256"] is not None:
            assert chain["protocol_file_sha256"] in bound_hashes
        if chain["result_sha256"] is not None:
            assert chain["result_sha256"] in bound_hashes

        protocol_bindings = [
            row for row in chain["supporting_files"] if row["role"] == "protocol"
        ]
        if chain["protocol_sha256"] is not None:
            assert len(protocol_bindings) == 1
            protocol_path = ROOT / protocol_bindings[0]["path"]
            protocol = _load(protocol_path)
            embedded = protocol.get("protocol_sha256")
            if embedded is None:
                assert chain["protocol_sha256"] == sha256_file(protocol_path)
            else:
                assert chain["protocol_sha256"] == embedded

    g2p = _by_chain(ledger, "chains")["track-c-ptbr-g2p-characterization-v1"]
    characterization = _load(
        ROOT
        / "artifacts"
        / "portuguese"
        / "20260717-ptbr-g2p-coverage-characterization-v1"
        / "characterization.json"
    )
    assert g2p["result_semantic_sha256"] == characterization["characterization_sha256"]


def test_candidate_flags_are_exactly_false_and_hash_bound() -> None:
    ledger = _ledger()
    receipt = ledger["candidate_flag_receipt"]
    wrangler = _load(ROOT / receipt["path"])

    assert sha256_file(ROOT / receipt["path"]) == receipt["file_sha256"]
    assert receipt["all_candidate_flags_exactly_false"] is True
    assert set(receipt["flags"]) == {
        "KOKORO_ENGLISH_CANDIDATE_ENABLED",
        "PORTUGUESE_RENDERER_CANDIDATE_ENABLED",
        "RECIPROCAL_LISTENER_LENS_RESEARCH_ENABLED",
    }
    for name, record in receipt["flags"].items():
        assert wrangler["vars"][name] == "false"
        assert record == {"configured_value": "false", "enabled": False}

    for chain in ledger["chains"]:
        feature_flag = chain["feature_flag"]
        if feature_flag is not None:
            assert receipt["flags"][feature_flag]["enabled"] is False


def test_queue_order_eligibility_and_ledger_statuses_agree() -> None:
    ledger = _ledger()
    queue = _queue()
    chains = _by_chain(ledger, "chains")
    items = queue["items"]

    assert [row["order"] for row in items] == [1, 2, 3, 4]
    assert [row["chain_id"] for row in items] == QUEUE_CHAIN_IDS
    assert [row["queue_state"] for row in items[:3]] == [
        "eligible_now",
        "eligible_now",
        "blocked_prerequisite",
    ]

    eligible = [row for row in items if row["queue_state"] == "eligible_now"]
    assert queue["eligible_now_count"] == len(eligible)
    assert [row["chain_id"] for row in eligible[:2]] == QUEUE_CHAIN_IDS[:2]
    assert {row["chain_id"] for row in eligible} == {
        chain_id
        for chain_id, chain in chains.items()
        if chain["human_status"] == "pending_human_review"
    }

    for item in items:
        chain = chains[item["chain_id"]]
        if item["queue_state"] == "eligible_now":
            assert chain["review_page"] == item["review_page"]
            assert chain["response_filename"] == item["response_filename"]
            page = _assert_safe_repo_path(item["review_page"])
            assert page.is_file()
            href_target = (HUB / item["review_href"]).resolve()
            assert href_target == page.resolve()
            assert href_target.is_relative_to(ROOT.resolve())
        else:
            assert item["review_page"] is None
            assert item["review_href"] is None
            assert item["response_filename"] is None
            assert item["estimated_minutes"] is None

    renderer_qc = items[2]
    assert chains[renderer_qc["chain_id"]]["human_status"] == ("blocked_prerequisite")
    assert chains[renderer_qc["chain_id"]]["artifact_root"] is None
    assert not (
        ROOT / "artifacts" / "portuguese" / "20260717-ptbr-renderer-qc-v1"
    ).exists()


def test_track_d_final_inconclusive_branch_is_hash_bound_and_ledger_only() -> None:
    ledger = _ledger()
    queue = _queue()
    chain = _by_chain(ledger, "chains")["track-d-reciprocal-feasibility-v1"]
    item = _by_chain(queue, "items")["track-d-reciprocal-feasibility-v1"]

    assert ledger["status"] == "machine_work_complete_human_actions_pending"
    assert queue["status"] == "machine_work_complete_human_actions_pending"
    assert ledger["generated_from_repository_head"] == (
        "4588121fe69ddee4984675bdf1083f4f5e6bab3a"
    )
    assert queue["generated_from_repository_head"] == (
        "4588121fe69ddee4984675bdf1083f4f5e6bab3a"
    )
    assert TRACK_D_ROOT.is_dir()
    analysis = _load(TRACK_D_ROOT / "analysis.json")
    assert analysis["classification"] == "automatic_measurement_inconclusive"
    assert analysis["measurement_status"] == "inconclusive_measurement_error"
    assert analysis["automatic_acoustic_feasibility_pass"] is False
    assert analysis["claim"] == "no positive acoustic-feasibility claim"
    assert "retained=2/queried=2" in analysis["measurement_error"]
    assert "5500 Hz" in analysis["measurement_error"]

    assert chain["machine_status"] == "automatic_measurement_inconclusive"
    assert chain["human_status"] == "not_eligible_machine_inconclusive"
    assert chain["claim_tier"] == "no_positive_acoustic_feasibility_claim"
    assert chain["protocol_sha256"] == (
        "002fef936f04c293624046badc8d6f5c58b5bc3ab2858a24b3bee3bb68db2a69"
    )
    assert chain["protocol_file_sha256"] == (
        "9ba316338f511dcf4752275a28488883e873087538df39ce01beae82a7a02cc1"
    )
    assert chain["result_sha256"] == (
        "24c99df1a04087f84752c8420720638c281593aea27583a307037ea283c928e0"
    )
    assert chain["review_page"] is None
    assert chain["response_filename"] is None
    assert chain["estimated_minutes"] is None
    assert chain["decision_unlocked"] is False
    assert item["queue_state"] == "ledger_only"
    assert item["review_page"] is None
    assert item["response_filename"] is None
    assert item["estimated_minutes"] is None
    assert not (TRACK_D_ROOT / "public" / "review").exists()
    assert not (TRACK_D_ROOT / "private").exists()

    bindings = {row["role"]: row for row in chain["supporting_files"]}
    assert bindings["render_attempt"]["sha256"] == (
        "75175eaab5363cd5f2385b1bf0d682c4ed3431a792e4f04f1c5c5024ea5f7857"
    )
    assert bindings["render_records"]["sha256"] == (
        "3fa16b937fd7c772e2661f086809c9c387a381c822db6085422e722c21b41f20"
    )
    assert bindings["analysis"]["sha256"] == chain["result_sha256"]


def test_final_handoff_rejects_provisional_or_conditional_track_d_state() -> None:
    final_text = "\n".join(
        (HUB / filename).read_text(encoding="utf-8")
        for filename in (
            "status-ledger.schema.json",
            "status-ledger.json",
            "queue.schema.json",
            "queue.json",
            "index.html",
        )
    )
    assert "provisional_track_d_pending_machine_result" not in final_text
    assert "protocol_pending_independent_review_not_executed" not in final_text
    assert "conditional_on_machine_result" not in final_text
    assert "no_acoustic_feasibility_claim_yet" not in final_text


def test_track_g_binds_committed_docs_and_final_static_hub() -> None:
    ledger = _ledger()
    chain = _by_chain(ledger, "chains")["track-g-docs-review-hub-v1"]
    bindings = {row["path"]: row["sha256"] for row in chain["supporting_files"]}
    doc_paths = (
        "README.md",
        "DESIGN.md",
        "DEVLOG.md",
        "PROVENANCE.md",
        "THIRD_PARTY_NOTICES.md",
        "DEPLOYMENT.md",
    )

    assert chain["machine_status"] == "documentation_and_review_hub_complete"
    assert chain["prerequisites"] == []
    assert chain["result_sha256"] == sha256_file(HUB / "index.html")
    for path in doc_paths:
        _assert_frozen_binding_or_append_only_devlog(
            ledger, {"path": path, "sha256": bindings[path]}
        )
    assert chain["result_semantic_sha256"] == sha256_json(
        {path: bindings[path] for path in doc_paths}
    )


def test_eligible_voice_manifests_are_exact_and_identity_opaque() -> None:
    queue = _queue()
    ledger = _ledger()
    chains = _by_chain(ledger, "chains")
    inventory = _load(VOICE_ROOT / "inventory.json")
    voice_ids = {
        *inventory["english_screen_shortlist"],
        *inventory["portuguese_screen_voices"],
    }
    expected_clip_counts = {
        "track-b-en-voice-screen-v1": 18,
        "track-b-ptbr-voice-screen-v1": 9,
    }

    for item in queue["items"][:2]:
        chain = chains[item["chain_id"]]
        review_page = ROOT / item["review_page"]
        manifest_path = review_page.parent / "public-manifest.json"
        manifest = _load(manifest_path)
        public_text = manifest_path.read_text(encoding="utf-8") + (
            review_page.read_text(encoding="utf-8")
        )

        assert manifest["status"] == "pending-human-review"
        assert manifest["protocol_sha256"] == chain["protocol_sha256"]
        assert manifest["response_filename"] == item["response_filename"]
        assert len(manifest["clips"]) == expected_clip_counts[item["chain_id"]]
        assert all(voice_id not in public_text for voice_id in voice_ids)
        assert "voice_id" not in public_text
        assert "blind-key" not in public_text


def test_static_index_links_exactly_the_current_eligible_queue() -> None:
    queue = _queue()
    html = (HUB / "index.html").read_text(encoding="utf-8")
    parser = _HubHTMLParser()
    parser.feed(html)

    expected_items = {
        row["chain_id"]: row
        for row in queue["items"]
        if row["queue_state"] == "eligible_now"
    }
    actual_anchors = {chain_id: attrs["href"] for chain_id, attrs in parser.anchors}
    assert actual_anchors == {
        chain_id: item["review_href"] for chain_id, item in expected_items.items()
    }

    article_rows = {row["data-chain-id"]: row for row in parser.articles}
    assert set(article_rows) == set(QUEUE_CHAIN_IDS)
    for item in queue["items"]:
        article = article_rows[item["chain_id"]]
        assert article["data-order"] == str(item["order"])
        assert article["data-queue-state"] == item["queue_state"]

    csp = next(
        row["content"]
        for row in parser.meta
        if row.get("http-equiv") == "Content-Security-Policy"
    )
    assert "default-src 'none'" in csp
    assert "script-src 'none'" in csp
    assert "object-src 'none'" in csp
    assert "base-uri 'none'" in csp
    assert "<script" not in html.casefold()
    assert not any(
        name.casefold().startswith("on") for name, _ in parser.all_attributes
    )

    rendered_text = " ".join(parser.text)
    for item in expected_items.values():
        assert item["response_filename"] in rendered_text


def test_hub_excludes_sensitive_mappings_and_failed_review_links() -> None:
    ledger = _ledger()
    queue = _queue()
    chains = _by_chain(ledger, "chains")
    public_blob = "\n".join(
        (HUB / filename).read_text(encoding="utf-8")
        for filename in ("status-ledger.json", "queue.json", "index.html")
    )
    inventory = _load(VOICE_ROOT / "inventory.json")
    voice_ids = {
        *inventory["english_screen_shortlist"],
        *inventory["portuguese_screen_voices"],
    }

    for forbidden in (
        "/private/",
        "private/en-blind-key.json",
        "private/ptbr-blind-key.json",
        "blind-key",
        "condition-key",
    ):
        assert forbidden not in public_blob
    assert all(voice_id not in public_blob for voice_id in voice_ids)

    for sensitive_file in (VOICE_ROOT / "private").glob("*.json"):
        assert sha256_file(sensitive_file) not in public_blob

    assert "20260716-kokoro-typed-replication-v1/review.html" not in public_blob
    assert "kokoro-en-typed-confirmation-v1-response.json" not in public_blob
    assert "typed-confirmation-v1/review.html" not in public_blob
    assert chains["track-a-frozen-replication-v1"]["review_page"] is None
    assert chains["track-a-fresh-confirmation-v1"]["review_page"] is None
    assert not (
        ROOT
        / "artifacts"
        / "typed-engine"
        / "20260717-kokoro-typed-confirmation-v1"
        / "review.html"
    ).exists()

    eligible_paths = {
        row["review_page"]
        for row in queue["items"]
        if row["queue_state"] == "eligible_now"
    }
    assert eligible_paths == {
        row["review_page"]
        for row in ledger["chains"]
        if row["human_status"] == "pending_human_review"
    }


def test_all_repo_relative_paths_are_normalized_and_bounded() -> None:
    ledger = _ledger()
    queue = _queue()

    _assert_safe_repo_path(ledger["candidate_flag_receipt"]["path"])
    for chain in ledger["chains"]:
        for key in ("artifact_root", "review_page"):
            if chain[key] is not None:
                _assert_safe_repo_path(chain[key])
        for binding in chain["supporting_files"]:
            _assert_safe_repo_path(binding["path"])

    for item in queue["items"]:
        if item["review_page"] is not None:
            _assert_safe_repo_path(item["review_page"])
        if item["review_href"] is not None:
            assert re.fullmatch(
                r"\.\./\.\./(?:voice-screen|research)/[A-Za-z0-9._/-]+\.html",
                item["review_href"],
            )
            target = (HUB / item["review_href"]).resolve()
            assert target.is_relative_to(ROOT.resolve())
