from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from .config import DEVLOG_PATH, Paths
from .pipeline import RESULT_FIELDS
from .util import atomic_write_text, read_csv, write_csv


class ReportError(RuntimeError):
    pass


def _truth(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "yes", "1"}


def _float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_report(run_id: str) -> dict[str, Any]:
    run_dir = Paths().run_dir(run_id)
    results_path = run_dir / "results.csv"
    ratings_path = run_dir / "ratings.csv"
    if not ratings_path.is_file():
        raise ReportError(f"Blind rating sheet missing: {ratings_path}")
    rows = read_csv(results_path)
    ratings = {row["blind_id"]: row for row in read_csv(ratings_path)}
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))

    started = datetime.fromisoformat(manifest["started_at_utc"])
    elapsed = (datetime.now(timezone.utc) - started).total_seconds() - float(
        manifest.get("manual_pause_seconds", 0)
    )
    deadline_exceeded = elapsed > float(manifest["timebox_seconds"])

    for row in rows:
        rating = ratings.get(row["blind_id"], {})
        for field in (
            "human_fluent",
            "human_pace",
            "human_prosody",
            "human_coherence",
            "human_confidence",
            "human_glitch_or_spelling",
            "human_real_word_autocorrection",
            "human_notes",
            "g2p_judgment",
        ):
            row[field] = rating.get(field, "")
        completed = bool(row["human_fluent"])
        human_pass = (
            row["human_fluent"].lower() == "yes"
            and row["human_glitch_or_spelling"].lower() == "no"
            and row["human_real_word_autocorrection"].lower() == "no"
        )
        row["human_pass"] = human_pass
        row["clip_pass"] = _truth(row.get("machine_pass", "")) and human_pass
        if deadline_exceeded and not completed:
            row["failure_stage"] = "deadline"
            row["failure_code"] = "deadline_aborted"
            row["failure_detail"] = "Human review was incomplete at the active deadline"
    write_csv(results_path, rows, RESULT_FIELDS)

    cohorts: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        cohorts[(row["renderer_slug"], row["language"])].append(row)
    cohort_summary = {
        f"{renderer}:{language}": {
            "size": len(group),
            "passing_clips": sum(_truth(row["clip_pass"]) for row in group),
            "passed": any(_truth(row["clip_pass"]) for row in group),
        }
        for (renderer, language), group in cohorts.items()
    }
    renderer_names = sorted({renderer for renderer, _ in cohorts})
    viable = []
    for renderer in renderer_names:
        english = cohort_summary.get(f"{renderer}:en", {}).get("passed", False)
        other = any(
            cohort_summary.get(f"{renderer}:{language}", {}).get("passed", False)
            for language in ("es", "pt")
        )
        if english and other:
            viable.append(renderer)

    def renderer_rank(renderer: str) -> tuple:
        groups = [group for (name, _), group in cohorts.items() if name == renderer]
        flat = [row for group in groups for row in group]
        passed_cohorts = sum(
            bool(cohort_summary[f"{renderer}:{language}"]["passed"])
            for language in ("en", "es", "pt")
            if f"{renderer}:{language}" in cohort_summary
        )
        english_passes = sum(
            _truth(row["clip_pass"]) for row in cohorts.get((renderer, "en"), [])
        )
        total_passes = sum(_truth(row["clip_pass"]) for row in flat)
        diagnostic = [
            (_float(row["human_pace"]) or 0) + (_float(row["human_prosody"]) or 0)
            for row in flat
            if row["human_pace"] and row["human_prosody"]
        ]
        target_scores = [
            score
            for row in flat
            if (score := _float(row["whisper_target_score"])) is not None
        ]
        return (
            passed_cohorts,
            english_passes,
            total_passes,
            statistics.median(diagnostic) if diagnostic else 0,
            statistics.median(target_scores) if target_scores else 0,
        )

    ranked = sorted(viable, key=renderer_rank, reverse=True)
    winner = None
    exact_tie = False
    if ranked:
        winner = ranked[0]
        if len(ranked) > 1 and renderer_rank(ranked[0]) == renderer_rank(ranked[1]):
            winner = None
            exact_tie = True

    g2p_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        if row.get("g2p_judgment"):
            g2p_counts[row["renderer_slug"]][row["g2p_judgment"]] += 1
    escalation = False
    if winner:
        counts = g2p_counts[winner]
        judged = sum(counts.values())
        escalation = judged > 0 and (counts["match"] + counts["near"]) / judged < 0.60

    summary = {
        "verdict": "GO" if viable else "NO-GO",
        "viable_renderers": viable,
        "winner": winner,
        "exact_tie": exact_tie,
        "cohorts": cohort_summary,
        "g2p_counts": {renderer: dict(counts) for renderer, counts in g2p_counts.items()},
        "g2p_isolated_probe_escalation": escalation,
        "active_elapsed_seconds": elapsed,
        "deadline_exceeded": deadline_exceeded,
        "review_confidence_median": statistics.median(
            confidence_values
        ) if (confidence_values := [
            value
            for row in rows
            if (value := _float(row.get("human_confidence", ""))) is not None
        ]) else None,
    }
    manifest.update(
        {
            "status": "complete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
        }
    )
    from .util import atomic_write_json

    atomic_write_json(run_dir / "run.json", manifest)

    lines = [
        f"### Run `{run_id}`",
        "",
        f"- Verdict: **{summary['verdict']}**",
        f"- Viable renderers: {', '.join(viable) if viable else 'none'}",
        f"- Winner: {winner or ('exact tie' if exact_tie else 'none')}",
        f"- Active elapsed time: {elapsed:.1f} seconds",
        f"- Deadline exceeded: {'yes' if deadline_exceeded else 'no'}",
    ]
    for cohort, values in sorted(cohort_summary.items()):
        lines.append(
            f"- Cohort {cohort}: {values['passing_clips']}/{values['size']} joint passes "
            f"({'PASS' if values['passed'] else 'FAIL'})"
        )
    for renderer, counts in sorted(summary["g2p_counts"].items()):
        lines.append(f"- G2P observations for {renderer}: {json.dumps(counts, sort_keys=True)}")
    if escalation:
        lines.append("- G2P follow-up: run isolated-token probes before trusting a G2P-dependent component.")
    if not viable:
        lines.append(
            "- Next ladder: GPT-5.6 respelling loop → G2P convergence check → "
            "GPT Audio 1.5 → open-source phoneme TTS → reserve project."
        )

    devlog = DEVLOG_PATH.read_text(encoding="utf-8")
    pending = "## Verdict\n\nPending the timed run."
    replacement = "## Verdict\n\n" + "\n".join(lines)
    if pending in devlog:
        devlog = devlog.replace(pending, replacement)
    else:
        devlog += "\n\n" + replacement + "\n"
    atomic_write_text(DEVLOG_PATH, devlog)
    return summary
