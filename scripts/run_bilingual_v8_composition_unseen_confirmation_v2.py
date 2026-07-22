from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from earshift_bakeoff.bilingual_candidate_runtime import (
    BilingualCandidateRuntime,
    _count_rule_occurrences,
    _pcm_hash,
    _wav_bytes,
)
from earshift_bakeoff.bilingual_product_isolation import active_changed_rule_ids
from earshift_bakeoff.config import ROOT, stable_json
from earshift_bakeoff.util import atomic_write_json, sha256_file


PROTOCOL_PATH = (
    ROOT / "rules" / "bilingual-v8-composition-unseen-confirmation-v2.json"
)
RUN_DIR = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260718-bilingual-v8-composition-unseen-confirmation-v2"
)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def _semantic_hash(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("record_sha256", None)
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def _write_bytes_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to replace frozen artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _validate_protocol(protocol: dict[str, Any]) -> None:
    fixtures = protocol.get("fixture_groups")
    if (
        protocol.get("schema_version") != 1
        or protocol.get("protocol_version")
        != "bilingual-v8-composition-unseen-confirmation-v2"
        or protocol.get("production_enabled") is not False
        or protocol.get("api_calls_allowed") != 0
        or not isinstance(fixtures, list)
        or len(fixtures) != 3
    ):
        raise ValueError("unseen composition protocol contract drifted")
    for binding in protocol["bindings"].values():
        path = ROOT / binding["path"]
        if sha256_file(path) != binding["sha256"]:
            raise ValueError(f"unseen composition binding drifted: {binding['path']}")


def _selection_receipt(
    runtime: BilingualCandidateRuntime, fixture: dict[str, Any], text: str
) -> tuple[dict[str, Any], Any | None]:
    try:
        source_plan = runtime.base_planner.plan(text)
        changed_rule_ids = active_changed_rule_ids(source_plan)
        passing_cells = tuple(
            cell
            for rule_id in changed_rule_ids
            if (
                (cell := runtime.registry.cell(
                    fixture["profile_id"], fixture["voice_id"], rule_id
                ))
                is not None
                and cell.automatic_pass
            )
        )
        selected_rule_ids = tuple(sorted(cell.rule_id for cell in passing_cells))
        reasons: list[str] = []
        if not 2 <= len(passing_cells) <= 3:
            reasons.append("passing_rule_count_outside_two_to_three")
        if any(cell.candidate_rung != "v8" for cell in passing_cells):
            reasons.append("mixed_or_non_v8_candidate_rung")
        if reasons:
            return (
                {
                    "text": text,
                    "eligible": False,
                    "rejection_reasons": reasons,
                    "source_plan_sha256": source_plan.plan_sha256,
                    "changed_rule_ids": list(changed_rule_ids),
                    "passing_cells": [
                        {
                            "rule_id": cell.rule_id,
                            "candidate_rung": cell.candidate_rung,
                        }
                        for cell in passing_cells
                    ],
                },
                None,
            )
        isolated_plan = runtime._composition_planner(selected_rule_ids).plan(text)
        isolated_rule_ids = active_changed_rule_ids(isolated_plan)
        occurrences = {
            rule_id: _count_rule_occurrences(isolated_plan, rule_id)
            for rule_id in selected_rule_ids
        }
        source_occurrences = {
            rule_id: _count_rule_occurrences(source_plan, rule_id)
            for rule_id in selected_rule_ids
        }
        if isolated_rule_ids != selected_rule_ids:
            reasons.append("isolated_rule_identity_drift")
        if occurrences != source_occurrences or any(count <= 0 for count in occurrences.values()):
            reasons.append("selected_occurrence_count_drift")
        omitted_rule_ids = [
            rule_id for rule_id in changed_rule_ids if rule_id not in selected_rule_ids
        ]
        receipt = {
            "text": text,
            "eligible": not reasons,
            "rejection_reasons": reasons,
            "source_plan_sha256": source_plan.plan_sha256,
            "isolated_plan_sha256": isolated_plan.plan_sha256,
            "changed_rule_ids": list(changed_rule_ids),
            "selected_rule_ids": list(selected_rule_ids),
            "selected_rule_occurrences": occurrences,
            "omitted_rule_ids": omitted_rule_ids,
            "neutral_script": isolated_plan.neutral_script,
            "lens_script": isolated_plan.lens_script,
        }
        return receipt, isolated_plan if not reasons else None
    except Exception as exc:
        return (
            {
                "text": text,
                "eligible": False,
                "rejection_reasons": [f"planner_error:{type(exc).__name__}"],
            },
            None,
        )


def _selected_fixture(
    runtime: BilingualCandidateRuntime, fixture: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    receipts: list[dict[str, Any]] = []
    for text in fixture["candidate_inventory"]:
        receipt, plan = _selection_receipt(runtime, fixture, text)
        receipts.append(receipt)
        if plan is not None:
            expected = fixture["selected_fixture"]
            actual = {
                "text": receipt["text"],
                "source_plan_sha256": receipt["source_plan_sha256"],
                "isolated_plan_sha256": receipt["isolated_plan_sha256"],
                "neutral_script": receipt["neutral_script"],
                "lens_script": receipt["lens_script"],
                "selected_rule_occurrences": receipt["selected_rule_occurrences"],
                "omitted_rule_ids": receipt["omitted_rule_ids"],
            }
            if actual != expected:
                raise ValueError(
                    f"frozen first-eligible selection drifted for {fixture['fixture_id']}"
                )
            return receipt, receipts
    raise ValueError(f"no gate-clean fixture for {fixture['fixture_id']}")


def _audio_receipt(path: Path, pcm: Any) -> dict[str, Any]:
    wav = _wav_bytes(pcm)
    _write_bytes_once(path, wav)
    return {
        "relative_path": str(path.relative_to(RUN_DIR)),
        "wav_sha256": hashlib.sha256(wav).hexdigest(),
        "pcm_sha256": _pcm_hash(pcm),
        "sample_count": int(pcm.size),
        "duration_s": int(pcm.size) / 24_000,
    }


def main() -> int:
    if (RUN_DIR / "results.json").exists():
        raise FileExistsError("unseen composition confirmation already has a result")
    protocol = _load_object(PROTOCOL_PATH)
    _validate_protocol(protocol)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for fixture in protocol["fixture_groups"]:
        runtime = BilingualCandidateRuntime.load(
            fixture["profile_id"], fixture["voice_id"]
        )
        selected, selection_receipts = _selected_fixture(runtime, fixture)
        row: dict[str, Any] = {
            "fixture_id": fixture["fixture_id"],
            "fixture_role": fixture["fixture_role"],
            "profile_id": fixture["profile_id"],
            "voice_id": fixture["voice_id"],
            "selection_receipts": selection_receipts,
            "selected_text": selected["text"],
            "source_plan_sha256": selected["source_plan_sha256"],
            "isolated_plan_sha256": selected["isolated_plan_sha256"],
            "selected_rule_occurrences": selected["selected_rule_occurrences"],
            "omitted_rule_ids": selected["omitted_rule_ids"],
        }
        try:
            candidate = runtime.render_v8_composition_candidate(selected["text"])
            contract_pass = bool(
                candidate.isolated_plan.plan_sha256
                == selected["isolated_plan_sha256"]
                and candidate.isolated_plan.neutral_script
                == selected["neutral_script"]
                and candidate.isolated_plan.lens_script == selected["lens_script"]
                and {
                    cell.rule_id: _count_rule_occurrences(
                        candidate.isolated_plan, cell.rule_id
                    )
                    for cell in candidate.cells
                }
                == selected["selected_rule_occurrences"]
                and list(candidate.omitted_rule_ids) == selected["omitted_rule_ids"]
                and candidate.render.plan.plan_sha256
                == candidate.isolated_plan.plan_sha256
            )
            audio_dir = RUN_DIR / "audio"
            row.update(
                {
                    "contract_pass": contract_pass,
                    "render_integrity": asdict(candidate.render.verification),
                    "acoustic": candidate.acoustic,
                    "audio": {
                        "neutral": _audio_receipt(
                            audio_dir / f"{fixture['fixture_id']}__neutral.wav",
                            candidate.render.neutral_pcm,
                        ),
                        "lens": _audio_receipt(
                            audio_dir / f"{fixture['fixture_id']}__lens.wav",
                            candidate.render.lens_pcm,
                        ),
                    },
                    "execution_error": None,
                }
            )
            row["automatic_pass"] = bool(
                contract_pass
                and candidate.render.verification.integrity_pass
                and candidate.acoustic["pass"]
            )
        except Exception as exc:
            row.update(
                {
                    "contract_pass": False,
                    "render_integrity": None,
                    "acoustic": None,
                    "audio": None,
                    "execution_error": {
                        "type": type(exc).__name__,
                        "code": getattr(exc, "code", None),
                        "message": str(exc),
                    },
                    "automatic_pass": False,
                }
            )
        rows.append(row)

    all_pass = all(row["automatic_pass"] for row in rows)
    result = {
        "schema_version": 1,
        "run_id": RUN_DIR.name,
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "classification": (
            "unseen_v8_composition_automatic_pass_pending_blind_human_qc"
            if all_pass
            else "unseen_v8_composition_automatic_failed_preserve_exact_result"
        ),
        "production_enabled": False,
        "api_calls_made": 0,
        "fixture_count": len(rows),
        "automatic_pass_count": sum(row["automatic_pass"] for row in rows),
        "render_set_count": len(rows),
        "selected_rule_occurrence_count": sum(
            sum(row["selected_rule_occurrences"].values()) for row in rows
        ),
        "shared_natural_decoder_render_count": sum(
            row["acoustic"]["shared_natural_decoder_render_count"]
            for row in rows
            if row["acoustic"] is not None
        ),
        "elapsed_s": time.perf_counter() - started,
        "fixtures": rows,
        "interpretation_limit": protocol["outcomes"]["interpretation_limit"],
    }
    result["record_sha256"] = _semantic_hash(result)
    atomic_write_json(RUN_DIR / "results.json", result)
    print(
        json.dumps(
            {
                "classification": result["classification"],
                "automatic_pass_count": result["automatic_pass_count"],
                "fixture_count": result["fixture_count"],
                "api_calls_made": 0,
                "result_sha256": sha256_file(RUN_DIR / "results.json"),
                "record_sha256": result["record_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
