from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from .acoustic_calibration import run_acoustic_calibration
from .api import api_contract_fingerprint, require_api_key, smoke_render
from .audio_conformance import run_conformance, run_prosody_bakeoff
from .config import Paths, criteria_sha256, load_config, verify_criteria_hash
from .corpus import CodexCorpusGenerator
from .gates import EspeakPhonemizer, build_gate_database
from .pipeline import execute_run
from .release_gate import release_check
from .report import build_report
from .review import build_review
from .util import atomic_write_json, sha256_file
from .verifier import prepare_whisper


def doctor() -> int:
    config = load_config()
    checks: list[tuple[str, bool, str]] = []
    try:
        digest = verify_criteria_hash(config)
        checks.append(("criteria", True, digest))
    except Exception as exc:
        checks.append(("criteria", False, str(exc)))
    checks.append(
        ("python", sys.version_info[:3] == (3, 12, 12), sys.version.split()[0])
    )
    for command in ("ffmpeg", "ffprobe", "say", "espeak-ng"):
        path = shutil.which(command)
        checks.append((command, bool(path), path or "missing"))
    try:
        version = EspeakPhonemizer().version()
        checks.append(("espeak-version", "1.52.0" in version, version))
    except Exception as exc:
        checks.append(("espeak-version", False, str(exc)))
    free = shutil.disk_usage(Paths().root).free
    checks.append(("free-disk", free >= 8 * 1024**3, f"{free / 1024**3:.1f} GiB"))
    checks.append(
        (
            "api-key",
            bool(os.environ.get("OPENAI_API_KEY", "").strip()),
            "present"
            if os.environ.get("OPENAI_API_KEY", "").strip()
            else "absent (allowed before smoke)",
        )
    )
    checks.append(("gate-db", Paths().gate_db.is_file(), str(Paths().gate_db)))
    checks.append(
        (
            "prepare-receipt",
            Paths().prepare_receipt.is_file(),
            str(Paths().prepare_receipt),
        )
    )
    try:
        corpus_path = Paths().root / config["generator"]["corpus_path"]
        corpus_receipt = CodexCorpusGenerator(corpus_path).receipt()
        checks.append(
            (
                "codex-corpus",
                corpus_receipt["candidates"] == 120,
                f"{corpus_receipt['candidates']} candidates; {corpus_receipt['corpus_sha256']}",
            )
        )
    except Exception as exc:
        checks.append(("codex-corpus", False, str(exc)))
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'WAIT'}  {name}: {detail}")
    hard_fail = any(
        not ok
        for name, ok, _ in checks
        if name not in {"api-key", "gate-db", "prepare-receipt"}
    )
    return 4 if hard_fail else 0


def prepare() -> int:
    config = load_config()
    criteria = verify_criteria_hash(config)
    paths = Paths()
    gate_receipt_path = paths.gate_db.with_suffix(".receipt.json")
    if paths.gate_db.is_file() and gate_receipt_path.is_file():
        gate_receipt = json.loads(gate_receipt_path.read_text(encoding="utf-8"))
        gate_receipt["database_sha256"] = sha256_file(paths.gate_db)
        print(f"Reusing gate database: {paths.gate_db}")
    else:
        print(
            "Building pinned word/G2P gate; this is local setup outside T0.", flush=True
        )
        gate_receipt = build_gate_database(config=config)

    whisper_receipt_path = paths.whisper_cache / "preflight-receipt.json"
    if whisper_receipt_path.is_file():
        whisper_receipt = json.loads(whisper_receipt_path.read_text(encoding="utf-8"))
        model_path = Path(whisper_receipt["model_path"])
        if (
            not model_path.is_dir()
            or sha256_file(model_path / "weights.npz")
            != whisper_receipt["weights_sha256"]
        ):
            whisper_receipt = prepare_whisper(config)
        else:
            print(f"Reusing verified Whisper preflight: {whisper_receipt_path}")
    else:
        print(
            "Downloading and benchmarking local Whisper large-v3 outside T0.",
            flush=True,
        )
        whisper_receipt = prepare_whisper(config)

    receipt = {
        "schema_version": 1,
        "criteria_sha256": criteria,
        "gate": gate_receipt,
        "whisper": whisper_receipt,
    }
    atomic_write_json(paths.prepare_receipt, receipt)
    measurements = whisper_receipt["measurements"]
    summary = {
        "criteria_sha256": criteria,
        "gate_database_sha256": gate_receipt["database_sha256"],
        "written_union_count": gate_receipt["written_union_count"],
        "whisper_variant": whisper_receipt["variant"],
        "whisper_weights_sha256": whisper_receipt["weights_sha256"],
        "whisper_peak_rss_gib": whisper_receipt["peak_rss_gib"],
        "controls": {
            language: {
                "ok": result["ok"],
                "top_language": result["top_language"],
                "target_score": result["target_score"],
                "margin": result["margin"],
                "elapsed_s": result["elapsed_s"],
            }
            for language, result in measurements.items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _is_voice_specific_rejection(exc: Exception, voice: str) -> bool:
    status = getattr(exc, "status_code", None)
    message = str(exc).lower()
    return (
        isinstance(status, int)
        and 400 <= status < 500
        and "voice" in message
        and voice.lower() in message
    )


def smoke() -> int:
    config = load_config()
    criteria = verify_criteria_hash(config)
    if not Paths().prepare_receipt.is_file():
        raise RuntimeError("Run prepare successfully before smoke")
    require_api_key()
    requested = config["preferred_voice"]
    errors: list[str] = []
    try:
        result = smoke_render(requested)
        effective = requested
    except Exception as exc:
        if not _is_voice_specific_rejection(exc, requested):
            raise
        errors.append(str(exc))
        effective = config["fallback_voice"]
        result = smoke_render(effective)
    receipt = {
        "criteria_sha256": criteria,
        "requested_voice": requested,
        "effective_voice": effective,
        "api_contract_fingerprint": api_contract_fingerprint(effective),
        "voice_fallback_errors": errors,
        "smoke": result,
    }
    atomic_write_json(Paths().smoke_receipt, receipt)
    print(f"Smoke passed with shared voice: {effective}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="renderer-bakeoff")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor")
    subparsers.add_parser("prepare")
    subparsers.add_parser("smoke")
    corpus_parser = subparsers.add_parser("corpus-validate")
    corpus_parser.add_argument("--path")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--run-id", required=True)
    review_parser = subparsers.add_parser("review")
    review_parser.add_argument("--run-id", required=True)
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--run-id", required=True)
    conformance_parser = subparsers.add_parser("audio-conformance")
    conformance_parser.add_argument("--run-id", required=True)
    prosody_parser = subparsers.add_parser("audio-prosody")
    prosody_parser.add_argument("--run-id", required=True)
    prosody_parser.add_argument("--attempts", type=int, default=3)
    calibration_parser = subparsers.add_parser("acoustic-calibration")
    calibration_parser.add_argument("--run-id", required=True)
    reaudit_parser = subparsers.add_parser("acoustic-reaudit")
    reaudit_parser.add_argument("--run-id", required=True)
    v3_rescore_parser = subparsers.add_parser("acoustic-v3-rescore")
    v3_rescore_parser.add_argument("--run-id", required=True)
    v3_confirm_parser = subparsers.add_parser("acoustic-v3-confirm")
    v3_confirm_parser.add_argument("--run-id", required=True)
    curated_pair_parser = subparsers.add_parser("curated-matched-pair")
    curated_pair_parser.add_argument("--run-id", required=True)
    word_freeze_parser = subparsers.add_parser("same-take-word-freeze")
    word_freeze_parser.add_argument("--run-id", required=True)
    word_edit_parser = subparsers.add_parser("same-take-word-edit")
    word_edit_parser.add_argument("--run-id", required=True)
    release_parser = subparsers.add_parser("release-check")
    release_parser.add_argument("--strict", action="store_true")
    subparsers.add_parser("criteria-hash", help=argparse.SUPPRESS)

    args = parser.parse_args(argv)
    if args.command == "doctor":
        return doctor()
    if args.command == "prepare":
        return prepare()
    if args.command == "smoke":
        return smoke()
    if args.command == "corpus-validate":
        config = load_config()
        path = (
            Path(args.path)
            if args.path
            else Paths().root / config["generator"]["corpus_path"]
        )
        print(
            json.dumps(CodexCorpusGenerator(path).receipt(), indent=2, sort_keys=True)
        )
        return 0
    if args.command == "run":
        result = execute_run(args.run_id)
        print(f"Automated stages complete: {result}")
        print(f"Next: renderer-bakeoff review --run-id {args.run_id}")
        return 0
    if args.command == "review":
        destination = build_review(args.run_id)
        print(f"Open {destination}")
        return 0
    if args.command == "report":
        print(json.dumps(build_report(args.run_id), indent=2, sort_keys=True))
        return 0
    if args.command == "audio-conformance":
        print(json.dumps(run_conformance(args.run_id), indent=2, sort_keys=True))
        return 0
    if args.command == "audio-prosody":
        print(
            json.dumps(
                run_prosody_bakeoff(args.run_id, attempts=args.attempts),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "acoustic-calibration":
        print(
            json.dumps(
                run_acoustic_calibration(args.run_id),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "acoustic-reaudit":
        from .acoustic_reaudit import run_praat_reaudit

        print(json.dumps(run_praat_reaudit(args.run_id), indent=2, sort_keys=True))
        return 0
    if args.command == "acoustic-v3-rescore":
        from .acoustic_calibration_v3 import rescore_existing_v3

        print(json.dumps(rescore_existing_v3(args.run_id), indent=2, sort_keys=True))
        return 0
    if args.command == "acoustic-v3-confirm":
        from .acoustic_calibration_v3 import run_confirmation_v3

        print(json.dumps(run_confirmation_v3(args.run_id), indent=2, sort_keys=True))
        return 0
    if args.command == "curated-matched-pair":
        from .curated_matched_pair import run_curated_matched_pair

        print(
            json.dumps(
                run_curated_matched_pair(run_id=args.run_id),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "same-take-word-freeze":
        from .same_take_word import build_word_source_freeze

        print(
            json.dumps(
                build_word_source_freeze(run_id=args.run_id),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "same-take-word-edit":
        from .same_take_word import run_word_editor

        print(
            json.dumps(
                run_word_editor(run_id=args.run_id),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "release-check":
        return release_check(strict=args.strict)
    if args.command == "criteria-hash":
        print(criteria_sha256())
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
