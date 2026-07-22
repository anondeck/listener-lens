from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from .config import load_config
from .gates import CandidateGate


AMENDMENT_PREREGISTRATION_HEADING = (
    "## Calibration-v3 amended-confirmation preregistration — July 15, 2026"
)
CANDIDATE_INVENTORY_VERSION = "calibration-v3-amendment-candidates-v1"


@dataclass(frozen=True)
class ShellCandidate:
    rank: int
    shell: str
    ih: str
    ee: str
    a: str
    eh: str
    heuristic_tier: str

    @property
    def surfaces(self) -> tuple[str, str, str, str]:
        return (self.ih, self.ee, self.a, self.eh)


# This order is frozen before any gate evaluation. It prioritizes consonantal
# contexts expected to preserve periodic energy around the vowel: voiced
# singleton contexts first, then a postvocalic /v/ + /d/ cluster, then a
# non-sibilant continuous coda. These are candidate-selection heuristics, not a
# causal account of the k_V_sh exclusions.
CANDIDATE_INVENTORY: tuple[ShellCandidate, ...] = (
    ShellCandidate(1, "b_V_v", "bihv", "beev", "bav", "behv", "voiced_singletons"),
    ShellCandidate(2, "d_V_v", "dihv", "deev", "dav", "dehv", "voiced_singletons"),
    ShellCandidate(3, "v_V_d", "vihd", "veed", "vad", "vehd", "voiced_singletons"),
    ShellCandidate(4, "z_V_d", "zihd", "zeed", "zad", "zehd", "voiced_singletons"),
    ShellCandidate(5, "b_V_vd", "bihvd", "beevd", "bavd", "behvd", "voiced_coda_cluster"),
    ShellCandidate(6, "d_V_vd", "dihvd", "deevd", "davd", "dehvd", "voiced_coda_cluster"),
    ShellCandidate(7, "g_V_vd", "gihvd", "geevd", "gavd", "gehvd", "voiced_coda_cluster"),
    ShellCandidate(8, "z_V_vd", "zihvd", "zeevd", "zavd", "zehvd", "voiced_coda_cluster"),
    ShellCandidate(9, "b_V_th", "bihth", "beeth", "bath", "behth", "continuous_nonsibilant_coda"),
    ShellCandidate(10, "d_V_th", "dihth", "deeth", "dath", "dehth", "continuous_nonsibilant_coda"),
    ShellCandidate(11, "g_V_th", "gihth", "geeth", "gath", "gehth", "continuous_nonsibilant_coda"),
    ShellCandidate(12, "v_V_th", "vihth", "veeth", "vath", "vehth", "continuous_nonsibilant_coda"),
)


def evaluate_candidate_inventory(
    *,
    gate: CandidateGate | None = None,
    inventory: Sequence[ShellCandidate] = CANDIDATE_INVENTORY,
) -> dict[str, Any]:
    """Evaluate every frozen surface; select the first all-surface gate pass."""

    gate = gate or CandidateGate()
    language = "en"
    voice = gate.voices[language]
    records: list[dict[str, Any]] = []
    selected: ShellCandidate | None = None
    for candidate in inventory:
        ipa_values = gate.phonemizer.phonemize(candidate.surfaces, voice)
        surfaces: list[dict[str, Any]] = []
        for label, surface, ipa in zip(
            ("ih", "ee", "a", "eh"), candidate.surfaces, ipa_values
        ):
            written_match = gate.text_match(surface)
            predicted_homophone = gate.phone_match(language, ipa)
            surfaces.append(
                {
                    "label": label,
                    "surface": surface,
                    "predicted_ipa": ipa,
                    "written_word_match": written_match,
                    "predicted_homophone_match": predicted_homophone,
                    "passed": not written_match and not predicted_homophone,
                }
            )
        passed = all(item["passed"] for item in surfaces)
        records.append({**asdict(candidate), "passed": passed, "surfaces": surfaces})
        if selected is None and passed:
            selected = candidate

    return {
        "schema_version": 1,
        "inventory_version": CANDIDATE_INVENTORY_VERSION,
        "language": language,
        "espeak_voice": voice,
        "espeak_version": gate.phonemizer.version(),
        "gate_database": str(gate.database),
        "word_gate_config": load_config()["word_gate"],
        "candidates": records,
        "selected": asdict(selected) if selected is not None else None,
    }
