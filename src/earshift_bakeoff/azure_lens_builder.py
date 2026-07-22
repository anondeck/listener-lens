"""Azure lens lane v1 — planner-to-SSML pair builder.

Turns typed text into a neutral/lens SSML pair for Azure Speech: the pinned
Misaki source adapters produce per-word phones, the listener profile's
changed vowel and consonant rules select target words, and both sides of the
pair carry a per-word ``<phoneme alphabet="ipa">`` tag for every affected
word (the pair must differ only in swapped symbols, never in
tagged-versus-plain rendering). Unaffected words stay plain text. A rule that
is enabled but whose source segment or cited context is absent from the text
is reported as ``context_absent``; families this lane version does not render
yet stay in ``omitted``. Nothing is silently skipped.

The symbol transliteration is ``rules/azure-ipa-map-v1.json``, derived from
the frozen reachability inventories. Unmapped symbols fail closed. Nothing
here calls Azure; rendering lives in the CLI (`--render`) and requires the
caller's key in the environment or ``.env.local``.
"""

from __future__ import annotations

import json
import re
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape, quoteattr

from .config import ROOT

AZURE_IPA_MAP_PATH = ROOT / "rules" / "azure-ipa-map-v1.json"
AZURE_LANE_VERSION = "azure-lens-lane-v1"
OUTPUT_FORMAT = "riff-24khz-16bit-mono-pcm"

PROFILE_LOCALES = {
    "en-US-to-pt-BR-listener-v2": "en-US",
    "pt-BR-to-en-US-listener-v2": "pt-BR",
}
DEFAULT_VOICES = {
    "en-US": "en-US-AvaNeural",
    "pt-BR": "pt-BR-FranciscaNeural",
    "it-IT": "it-IT-ElsaNeural",
    "de-DE": "de-DE-KatjaNeural",
    "es-ES": "es-ES-ElviraNeural",
    "es-MX": "es-MX-DaliaNeural",
    "fr-FR": "fr-FR-DeniseNeural",
}

# One pinned voice per locale, used for the speaker track: the listener
# language's own voice reading the raw source text. That clip is production
# ("as their mouth would say it"), a different claim from the lens's
# perception, and it needs the listener's voice rather than the source's.
# Kept in runtime source because the deploy container cannot import the
# build-time scripts; a test asserts it agrees with the scripts-side map.
LISTENER_VOICES = {
    "en-US": "en-US-AvaNeural", "pt-BR": "pt-BR-FranciscaNeural",
    "pt-PT": "pt-PT-RaquelNeural", "es-ES": "es-ES-ElviraNeural",
    "es-MX": "es-MX-DaliaNeural", "fr-FR": "fr-FR-DeniseNeural",
    "it-IT": "it-IT-ElsaNeural", "de-DE": "de-DE-KatjaNeural",
    "el-GR": "el-GR-AthinaNeural", "ru-RU": "ru-RU-SvetlanaNeural",
    "hi-IN": "hi-IN-SwaraNeural", "nl-NL": "nl-NL-FennaNeural",
    "pl-PL": "pl-PL-AgnieszkaNeural", "tr-TR": "tr-TR-EmelNeural",
    "sv-SE": "sv-SE-SofieNeural", "uk-UA": "uk-UA-PolinaNeural",
    "id-ID": "id-ID-GadisNeural", "cs-CZ": "cs-CZ-VlastaNeural",
    "ro-RO": "ro-RO-AlinaNeural", "hu-HU": "hu-HU-NoemiNeural",
    "nb-NO": "nb-NO-PernilleNeural", "ca-ES": "ca-ES-JoanaNeural",
    "hr-HR": "hr-HR-GabrijelaNeural", "sk-SK": "sk-SK-ViktoriaNeural",
    "sl-SI": "sl-SI-PetraNeural", "bg-BG": "bg-BG-KalinaNeural",
    "ms-MY": "ms-MY-YasminNeural", "mr-IN": "mr-IN-AarohiNeural",
    "te-IN": "te-IN-ShrutiNeural", "gu-IN": "gu-IN-DhwaniNeural",
}

# Speaker-track overrides: locales whose pinned voice reads a foreign sentence
# too well for this clip to show anything.
#
# The track works by handing a voice raw orthography and letting that
# language's own letter-to-sound rules colour it. Four of the pinned voices
# defeat that — Katja, Dalia, Elsa and Fenna read English near-natively, so
# the clip came back sounding English and the whole track was cut for parity
# rather than ship a button that did nothing in five of thirty languages.
#
# Only one voice per locale had ever been auditioned. That pinning exists for
# the *lens*, where a per-symbol acceptance receipt is only evidence for the
# voice it was taken on. The speaker track sends no phonemes, holds no
# receipts, and is therefore free to use a different voice — so every Azure
# voice in those locales was auditioned reading one fixed English sentence,
# and each of the four has a replacement that keeps its accent. Swedish was
# a false alarm: Sofie carries an accent under a calibrated ear and stays.
#
# All four replacements are female, like the other twenty-six. The track
# already introduces a second voice, and changing gender on top of that would
# leave the listener unable to tell the accent from the speaker.
#
# Absent locales fall through to LISTENER_VOICES; the lens's own voices are
# untouched, so no receipt is invalidated by anything here.
C_TRACK_VOICE_OVERRIDES = {
    "de-DE": "de-DE-KlarissaNeural",
    "es-MX": "es-MX-RenataNeural",
    "it-IT": "it-IT-FabiolaNeural",
    "nl-NL": "nl-NL-ColetteNeural",
}


def speaker_voice_for(listener_locale: str) -> str:
    """The voice that reads the raw source text for the speaker track."""

    return C_TRACK_VOICE_OVERRIDES.get(
        listener_locale, LISTENER_VOICES[listener_locale]
    )


class AzureLensBuilderError(RuntimeError):
    pass


# Locales added on the Azure lane keep their profiles here, self-contained,
# so the Kokoro-era en/pt tables and their frozen evidence stay untouched.
AZURE_PROFILES_PATH = ROOT / "rules" / "azure-listener-lenses-v1.json"
AZURE_MATRIX_PATH = ROOT / "rules" / "azure-listener-lenses-v2.json"
RULE_AUDIBILITY_PATH = (
    ROOT / "artifacts" / "azure-rule-distinctness-v1" / "receipts.json"
)
STRESS_PROBE_PATH = ROOT / "artifacts" / "azure-stress-probe-v1" / "receipts.json"

# Both lexical stress operations. One filter constant, because filtering on a
# literal is how shift_primary_stress_to_final silently fell out of the rules
# list — never applied and never accounted — when it was added second.
_STRESS_OPERATIONS = (
    "swap_primary_and_initial_secondary_stress",
    "shift_primary_stress_to_final",
)


@lru_cache(maxsize=1)
def load_stress_honour() -> dict[str, str]:
    """Per-locale verdict: does this voice render IPA stress position?

    Measured by probe_stress_marks_v1 with a minimal pair (ˈama / aˈma) and
    validated against real stress-contrastive words (English protest, German
    Kaffee): every current voice returns byte-identical audio when only the
    stress mark moves. A stress rule on an ``ignored`` voice is therefore a
    renderer no-op and must be reported as such, never as applied.
    """

    if not STRESS_PROBE_PATH.is_file():
        return {}
    data = json.loads(STRESS_PROBE_PATH.read_text(encoding="utf-8"))
    return {loc: row["verdict"] for loc, row in data.get("locales", {}).items()}


@lru_cache(maxsize=1)
def load_rule_audibility() -> dict[str, str]:
    """Per-(locale, source, target) audibility verdicts.

    Consulted rather than stamped onto the rules because the three rule
    sources cannot all be edited: the frozen bilingual tables are hash-bound
    evidence. Looking the verdict up by phone pair covers the generated
    matrix, the curated registry, and the frozen pair with one mechanism.
    """

    if not RULE_AUDIBILITY_PATH.is_file():
        return {}
    data = json.loads(RULE_AUDIBILITY_PATH.read_text(encoding="utf-8"))
    return {key: row["verdict"] for key, row in data.get("rules", {}).items()}


def load_azure_profiles() -> dict[str, dict[str, Any]]:
    """Every listener direction the lane can build.

    The generated matrix is read first and the hand-curated registry second,
    so curation wins on collision. The two files share ids by design — a
    curated direction and its generated counterpart describe the same pair —
    and the curated one carries hand-authored rule families plus per-symbol
    Azure acceptance receipts that the generator cannot reproduce. Loading it
    last keeps that evidence authoritative instead of silently replacing it
    with a derived baseline.
    """

    profiles: dict[str, dict[str, Any]] = {}
    for path in (AZURE_MATRIX_PATH, AZURE_PROFILES_PATH):
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for profile in data.get("profiles", ()):
            profiles[profile["id"]] = profile
    return profiles


# A direction has to change enough for a listener to have something to hear.
# Below this many audible rules the two takes are near-identical and the pair
# reads as a bug rather than a lens, so the direction is not offered at all.
MIN_AUDIBLE_RULES = 3


def audible_rule_count(profile: dict[str, Any]) -> int:
    """How many of a direction's rules the renderer actually voices.

    Counts only what the distinctness receipts cover: segmental substitutions
    with an `audible` verdict, deletions, and insertions. Stress rules are
    deliberately excluded even though they are present on 461 directions and
    almost certainly audible — they have no receipts, and a direction whose
    only effect is a moved accent has not re-pronounced anything through
    another sound system. Counting them would let unreceipted rules rescue
    directions that the receipted evidence says are empty.
    """

    locale = profile.get("source_locale") or PROFILE_LOCALES.get(profile.get("id", ""))
    audibility = load_rule_audibility()
    count = 0
    for rule in list(profile.get("vowel_rules") or []) + list(
        profile.get("consonant_rules") or []
    ):
        if rule.get("operation") == "delete":
            count += 1
            continue
        source, target = rule.get("source"), rule.get("target")
        if not source or not target or source == target:
            continue
        verdict = rule.get("renderer_verdict") or audibility.get(
            f"{locale}|{source}|{target}"
        )
        if verdict == "audible":
            count += 1
    return count + len(profile.get("insertion_rules") or [])


@lru_cache(maxsize=1)
def suppressed_profile_ids() -> frozenset[str]:
    """Generated directions too thin to ship.

    Derived from the receipts rather than hand-listed, so a map fix that makes
    a substitution audible restores its direction on the next probe with no
    code change. The curated and frozen profiles all clear the threshold on
    their own; nothing here is protecting them by exception.
    """

    return frozenset(
        profile_id
        for profile_id, profile in load_azure_profiles().items()
        if audible_rule_count(profile) < MIN_AUDIBLE_RULES
    )


def supported_profile_ids() -> frozenset[str]:
    """Every profile id the lane can actually build a pair for.

    The single source of truth for request validation: the two bilingual
    v2 profiles plus every profile declared in the Azure listener-lens
    registry, less the directions the renderer cannot voice. Service and
    Worker allowlists must agree with this set.
    """

    return (
        frozenset(PROFILE_LOCALES) | frozenset(load_azure_profiles())
    ) - suppressed_profile_ids()


def _load_profile(profile_id: str) -> dict[str, Any]:
    azure_profiles = load_azure_profiles()
    if profile_id in azure_profiles:
        if profile_id in suppressed_profile_ids():
            # Fail closed here as well as at request validation: a suppressed
            # direction would render two near-identical takes, which is worse
            # than no audio because it looks like it worked.
            raise AzureLensBuilderError(
                f"direction suppressed as inaudible: {profile_id} "
                f"({audible_rule_count(azure_profiles[profile_id])} audible "
                f"rules, minimum {MIN_AUDIBLE_RULES})"
            )
        return azure_profiles[profile_id]
    from .bilingual_listener_engine import load_listener_profiles

    try:
        return load_listener_profiles()[profile_id]
    except KeyError as exc:
        raise AzureLensBuilderError(
            f"unsupported profile for Azure lane: {profile_id}"
        ) from exc


@lru_cache(maxsize=None)
def _adapter_for(locale: str, voice_id: str | None) -> Any:
    """Load one G2P engine per (locale, voice) and keep it.

    Building these is expensive — the English and Portuguese adapters pull in
    Misaki, and every espeak-backed locale dlopens libespeak — while analysis
    itself costs about a millisecond. Uncached, a single lens request spent
    over a second constructing an engine it then used once, and ranking a
    sentence across a source language's listeners paid that cost per listener.
    The adapters are read-only once loaded, so one instance serves every
    direction that shares a source locale.
    """

    if locale == "en-US":
        from .bilingual_vowel_engine import EnglishMisakiAdapter

        return EnglishMisakiAdapter.load()
    if locale == "pt-BR":
        from .bilingual_vowel_engine import PortugueseMisakiAdapter

        return PortugueseMisakiAdapter.load(voice_id=voice_id or "pf_dora")
    from .azure_source_adapters import EspeakSourceAdapter

    return EspeakSourceAdapter.load(locale)


def _load_adapter(locale: str, profile: dict[str, Any]) -> Any:
    """The pinned adapter for a source locale.

    English and Brazilian Portuguese keep their bespoke Kokoro-era G2P so the
    validated pair is byte-for-byte unchanged; every locale added on the Azure
    lane rides the generic espeak adapter.
    """

    return _adapter_for(locale, profile.get("voice_id") if locale == "pt-BR" else None)


def load_ipa_map(path: Path = AZURE_IPA_MAP_PATH) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("map_id") != "azure-ipa-map-v1":
        raise AzureLensBuilderError("unexpected Azure IPA map identity")
    return data["locales"]


# The map keys one phoneme per row, but the adapters NFD-normalize, which
# splits a phoneme that is a base codepoint plus a combining mark (espeak de
# emits the ich-laut as precomposed U+00E7; the adapter hands it on as
# c + U+0327). Locales with such a phoneme compose the phone back to the
# adapter's own raw form before the per-codepoint walk. Composition is
# locale-gated: the en/pt/it locales keep the per-codepoint NFD walk the
# validated pt-BR nasal-vowel path (vowel + combining tilde mapped as two
# independent codepoints) was built on, and their output is byte-identical.
# German nasal loan vowels (ɑ̃, œ̃) have no precomposed Unicode form, so NFC
# composes exactly the ich-laut and nothing else in the de-DE inventory.
_MAP_NORMALIZATION_BY_LOCALE = {"de-DE": "NFC"}


def _map_symbols(
    phone: str,
    table: dict[str, Any],
    *,
    context: str,
    normalization: str = "NFD",
) -> str:
    chunks: list[str] = []
    for symbol in unicodedata.normalize(normalization, phone):
        row = table.get(symbol)
        if row is None:
            composed = unicodedata.normalize("NFC", symbol)
            row = table.get(composed)
        if row is None:
            raise AzureLensBuilderError(
                f"unmapped adapter symbol {symbol!r} in {context}"
            )
        chunks.append(row["azure_ipa"])
    mapped = "".join(chunks)
    if not mapped:
        raise AzureLensBuilderError(f"mapping emptied the phone string in {context}")
    return mapped


# Supra-segmental markers that never count as phonemic neighbours; mirrors
# bilingual_vowel_engine._STRUCTURAL_SYMBOLS / _COMBINING_TILDE so consonant
# context detection and the tilde guard use one shared inventory.
_STRUCTURAL_SYMBOLS = frozenset("ˈˌːʰʲ")
_COMBINING_TILDE = "̃"
NASAL_TILDE = _COMBINING_TILDE

# Fallback consonant inventory for locales the Kokoro-era opacity tables never
# covered. It is used only to decide whether a segment is post-vocalic, so a
# broad IPA consonant set is sufficient and safe.
_GENERAL_CONSONANTS = frozenset(
    "pbtdkɡʔfvszʃʒθðçxɣhɦmnŋɲɱlʎʟrɾɽʁɹɻjwʧʤʦʣβɸʝ"
)


@dataclass(frozen=True)
class _Applied:
    rule_id: str
    family: str


@dataclass(frozen=True)
class WordPlan:
    word_index: int
    written: str
    source_phone: str
    lens_phone: str
    applied: tuple[_Applied, ...]
    mapped_neutral: str
    mapped_lens: str

    @property
    def applied_rule_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.rule_id for item in self.applied))

    @property
    def affected(self) -> bool:
        """True only when the rendered ph strings actually differ.

        A rule can be applied at the adapter level and then erased by an
        approximate transliteration (for example ə→ɐ collapsing back to ə in
        the en-US map). Such words are never tagged as shifted and their
        rules are reported as map-neutralized, not applied.
        """

        return bool(self.applied) and self.mapped_neutral != self.mapped_lens


def _phonemic_neighbor(phone: str, position: int, step: int) -> str | None:
    index = position
    while 0 <= index < len(phone):
        symbol = phone[index]
        if symbol not in _STRUCTURAL_SYMBOLS and symbol != _COMBINING_TILDE:
            return symbol
        index += step
    return None


def _consonant_context_ok(
    rule: dict[str, Any],
    phone: str,
    start: int,
    end: int,
    vowel_symbols: frozenset[str],
) -> bool:
    contexts = rule.get("contexts", ("any",))
    if "any" in contexts:
        return True
    before = _phonemic_neighbor(phone, start - 1, -1)
    after = _phonemic_neighbor(phone, end, 1)
    for context in contexts:
        if context == "word_initial" and before is None:
            return True
        if context == "word_final" and after is None:
            return True
        if (
            context == "intervocalic"
            and before in vowel_symbols
            and after in vowel_symbols
        ):
            return True
    return False


def _apply_segment_rules(
    phone: str,
    vowel_rules: list[dict[str, Any]],
    consonant_rules: list[dict[str, Any]],
    vowel_symbols: frozenset[str],
) -> tuple[str, tuple[_Applied, ...]]:
    """Single left-to-right pass over the source phones.

    A rule's target is emitted and never re-scanned, so one rule's output can
    never be consumed as another rule's source. Oral-vowel rules are guarded
    off nasal vowels by the combining tilde; consonant rules honour their
    cited contexts (``any`` / ``intervocalic`` / ``word_initial``).
    """

    vowel_by_source: dict[str, dict[str, Any]] = {}
    for rule in vowel_rules:
        source = rule["source"]
        if source and source not in vowel_by_source:
            vowel_by_source[source] = rule
    consonant_by_source: dict[str, list[dict[str, Any]]] = {}
    for rule in consonant_rules:
        source = rule["source"]
        if source:
            consonant_by_source.setdefault(source, []).append(rule)
    sources = sorted(
        set(vowel_by_source) | set(consonant_by_source), key=len, reverse=True
    )
    out: list[str] = []
    applied: list[_Applied] = []
    index = 0
    length = len(phone)
    while index < length:
        matched = False
        for source in sources:
            if not phone.startswith(source, index):
                continue
            end = index + len(source)
            for candidate in consonant_by_source.get(source, ()):
                if _consonant_context_ok(candidate, phone, index, end, vowel_symbols):
                    out.append(str(candidate["target"]))
                    applied.append(_Applied(candidate["id"], "consonant"))
                    index = end
                    matched = True
                    break
            if matched:
                break
            vowel_rule = vowel_by_source.get(source)
            if vowel_rule is not None:
                if (
                    end < length
                    and phone[end] == _COMBINING_TILDE
                    and _COMBINING_TILDE not in source
                ):
                    continue
                out.append(str(vowel_rule["target"]))
                applied.append(_Applied(vowel_rule["id"], "vowel"))
                index = end
                matched = True
                break
        if not matched:
            out.append(phone[index])
            index += 1
    return "".join(out), tuple(applied)


def _apply_epenthesis(
    source_phone: str,
    lens_phone: str,
    insertion_rules: list[dict[str, Any]],
    *,
    obstruents: frozenset[str],
    legal_codas: frozenset[str],
    following_consonants: dict[str, Any],
    locale_consonants: frozenset[str],
) -> tuple[str, tuple[_Applied, ...]]:
    """Insert the epenthetic vowel a BP listener perceives after illegal codas.

    Runs on the substituted lens phone. Every vowel/consonant rule is
    length-preserving, so the obstruent positions detected on the source
    phone line up with the lens phone; if a future rule breaks that parity
    the insertion is skipped rather than misaligned. The vowel is added only
    on the lens side — the neutral side keeps the licit source coda.

    A cluster obstruent only epenthesises when it is post-vocalic (a coda);
    a legal onset cluster such as /kl/ in "club" or /br/ in "brother" keeps
    its shape, so the perceptual /i/ never splits a native BP onset.
    """

    inserted: list[tuple[int, str, str]] = []
    for rule in insertion_rules:
        operation = rule.get("operation")
        if operation == "insert_after":
            positions = _insert_after_boundaries(
                source_phone,
                rule,
                obstruents=obstruents,
                legal_codas=legal_codas,
                following_consonants=following_consonants,
                locale_consonants=locale_consonants,
            )
        elif operation == "insert_before":
            positions = _insert_before_boundaries(source_phone, rule)
        else:
            continue
        inserted.extend(
            (position, str(rule["target"]), rule["id"]) for position in positions
        )
    if not inserted or len(lens_phone) != len(source_phone):
        return lens_phone, ()
    by_position: dict[int, list[tuple[str, str]]] = {}
    for position, target, rule_id in sorted(inserted):
        by_position.setdefault(position, []).append((target, rule_id))
    out: list[str] = []
    applied: list[_Applied] = []
    for position in range(len(lens_phone) + 1):
        for target, rule_id in by_position.get(position, ()):
            out.append(target)
            applied.append(_Applied(rule_id, "insertion"))
        if position < len(lens_phone):
            out.append(lens_phone[position])
    return "".join(out), tuple(applied)


def _apply_deletions(
    lens_phone: str,
    deletion_rules: list[dict[str, Any]],
) -> tuple[str, tuple[_Applied, ...]]:
    """Drop segments the listener language has no category for at all.

    Substitution answers "which of my categories is this?"; deletion answers
    the case where the honest answer is "none, and I do not hear it" — French
    listeners and English /h/ being the motivating pair (haricot is /aʁiko/,
    not /haʁiko/).

    This runs last, after insertion, on purpose. Insertion aligns positions
    detected on the source phone against the lens phone and bails out unless
    the two are the same length, so deleting earlier would silently disable
    every insertion rule in the profile. Running deletion afterwards keeps
    that parity intact and leaves every already-validated profile — none of
    which delete anything — byte-for-byte unchanged.
    """

    if not deletion_rules:
        return lens_phone, ()
    out: list[str] = []
    applied: list[_Applied] = []
    seen: set[str] = set()
    index = 0
    length = len(lens_phone)
    while index < length:
        dropped = False
        for rule in deletion_rules:
            source = str(rule["source"])
            if not lens_phone.startswith(source, index):
                continue
            if not _deletion_context_ok(rule, lens_phone, index, index + len(source)):
                continue
            if rule["id"] not in seen:
                seen.add(rule["id"])
                applied.append(_Applied(rule["id"], "deletion"))
            index += len(source)
            dropped = True
            break
        if not dropped:
            out.append(lens_phone[index])
            index += 1
    return "".join(out), tuple(applied)


def _deletion_context_ok(
    rule: dict[str, Any],
    phone: str,
    start: int,
    end: int,
) -> bool:
    """Gate a deletion on its cited context.

    ``any`` deletes every occurrence — the right shape for a listener whose
    inventory simply lacks the category outright (French and English /h/).
    ``word_initial_cluster`` deletes only the first element of an onset the
    listener may not begin a word with, which is how English resolves /ps/
    and /ts/ (psychology, Zeit) without touching those segments elsewhere.
    """

    contexts = set(rule.get("contexts", ("any",)))
    if "any" in contexts:
        return True
    if "word_initial_cluster" in contexts:
        preceding = phone[:start].strip("ˈˌ")
        if preceding:
            return False
        following = phone[end:end + 1]
        allowed = set(rule.get("followed_by", ()))
        return bool(following) and following in allowed
    return False


def _insert_after_boundaries(
    source_phone: str,
    rule: dict[str, Any],
    *,
    obstruents: frozenset[str],
    legal_codas: frozenset[str],
    following_consonants: dict[str, Any],
    locale_consonants: frozenset[str],
) -> set[int]:
    """Positions after an illegal coda obstruent (BP perceptual /i/, IT paragoge).

    ``any_word_final_consonant`` widens the trigger past the obstruent set for
    listener languages whose words simply may not end in a consonant at all
    (Italian), while the original obstruent-gated contexts are untouched.
    """

    contexts = set(rule.get("contexts", ()))
    boundaries: set[int] = set()
    for position, symbol in enumerate(source_phone):
        if symbol in _STRUCTURAL_SYMBOLS or symbol == _COMBINING_TILDE:
            continue
        following = _phonemic_neighbor(source_phone, position + 1, 1)
        is_consonant = symbol in locale_consonants or symbol in _GENERAL_CONSONANTS
        if (
            following is None
            and is_consonant
            and "any_word_final_consonant" in contexts
        ):
            boundaries.add(position + 1)
            continue
        if symbol not in obstruents or symbol in legal_codas:
            continue
        if following is None and "word_final_obstruent" in contexts:
            boundaries.add(position + 1)
        elif (
            following in following_consonants
            and "illegal_consonant_cluster" in contexts
        ):
            preceding = _phonemic_neighbor(source_phone, position - 1, -1)
            if preceding is not None and preceding not in locale_consonants:
                boundaries.add(position + 1)
    return boundaries


def _insert_before_boundaries(source_phone: str, rule: dict[str, Any]) -> set[int]:
    """Positions before a word-initial trigger (ES prothesis, DE glottal onset).

    ``word_initial_cluster`` fires when the word opens with one of the rule's
    ``onsets`` followed by another consonant — Spanish /s/+C, giving the
    'eschool' percept. ``word_initial_vowel`` fires when the word opens with a
    vowel, which is where German inserts its glottal onset.
    """

    contexts = set(rule.get("contexts", ()))
    first = _phonemic_neighbor(source_phone, 0, 1)
    if first is None:
        return set()
    start = source_phone.index(first)
    if "word_initial_vowel" in contexts:
        vowels = set(rule.get("vowels", ()))
        if first in vowels:
            return {start}
        return set()
    if "word_initial_cluster" in contexts:
        onsets = set(rule.get("onsets", ()))
        if first not in onsets:
            return set()
        following = _phonemic_neighbor(source_phone, start + 1, 1)
        blockers = set(rule.get("not_followed_by", ()))
        if following is not None and following not in blockers:
            return {start}
    return set()


def _apply_stress_prosody(
    source_phone: str,
    lens_phone: str,
    prosody_rules: list[dict[str, Any]],
    vowel_symbols: frozenset[str],
) -> tuple[str, tuple[_Applied, ...]]:
    """Swap primary and initial-secondary lexical stress (BP initial bias).

    Bounded to the cited structure: an initial secondary stress with no
    stressable vowel before it, competing with a later primary stress. The
    stress marks ride in the ph string, so the swap renders directly in Azure
    IPA. Runs before epenthesis while lens and source are still aligned.
    """

    rule = next(
        (
            item
            for item in prosody_rules
            if item.get("operation") == "swap_primary_and_initial_secondary_stress"
        ),
        None,
    )
    if rule is None or len(lens_phone) != len(source_phone):
        return lens_phone, ()
    secondary = source_phone.find("ˌ")
    primary = source_phone.find("ˈ")
    if secondary < 0 or primary <= secondary:
        return lens_phone, ()
    if any(symbol in vowel_symbols for symbol in source_phone[:secondary]):
        return lens_phone, ()
    values = list(lens_phone)
    if values[secondary] != "ˌ" or values[primary] != "ˈ":
        return lens_phone, ()
    values[secondary], values[primary] = "ˈ", "ˌ"
    return "".join(values), (_Applied(rule["id"], "prosody"),)


def _shift_primary_stress_to_final(
    source_phone: str,
    lens_phone: str,
    prosody_rules: list[dict[str, Any]],
    vowel_symbols: frozenset[str],
) -> tuple[str, tuple[_Applied, ...]]:
    """Move lexical stress onto the final syllable (French oxytone bias).

    French has no contrastive lexical stress: prominence falls at the end of
    the group regardless of where the source language put it, which is the
    most audible thing a French listener does to an English word (BAsic ->
    baSIC). Implemented by relocating the existing primary mark rather than
    adding one, so the string length is unchanged and any insertion rule that
    runs afterwards still sees source/lens parity.
    """

    rule = next(
        (
            item
            for item in prosody_rules
            if item.get("operation") == "shift_primary_stress_to_final"
        ),
        None,
    )
    if rule is None or len(lens_phone) != len(source_phone):
        return lens_phone, ()
    primary = lens_phone.find("ˈ")
    if primary < 0:
        return lens_phone, ()
    nucleus = max(
        (
            index
            for index, symbol in enumerate(lens_phone)
            if symbol in vowel_symbols
        ),
        default=-1,
    )
    if nucleus < 0:
        return lens_phone, ()
    # Walk back over the onset cluster so the mark lands before the syllable,
    # not before the nucleus.
    onset = nucleus
    while onset > 0 and lens_phone[onset - 1] not in vowel_symbols:
        if lens_phone[onset - 1] in _STRUCTURAL_SYMBOLS:
            break
        onset -= 1
    if onset <= primary:
        return lens_phone, ()
    values = [symbol for index, symbol in enumerate(lens_phone) if index != primary]
    values.insert(onset - 1, "ˈ")
    return "".join(values), (_Applied(rule["id"], "prosody"),)


# Interrogative openers that make a "?" a wh-question rather than a yes/no
# (polar) question. Keyed by the locale of the text being read, because the
# openers are language-specific: one shared English/Portuguese set meant
# "Perché corri?", "Warum läufst du?" and "¿Dónde está la casa?" were all
# classified polar and given the rise-to-fall contour, a claim the rule
# explicitly scopes to yes/no questions.
#
# Accented and unaccented spellings are both listed. Spanish interrogatives
# carry an accent that distinguishes them from their relative-pronoun twins
# (qué/que), and a typist may omit it; treating the unaccented form as a
# wh-opener errs toward not applying the contour, which is the safe side.
_WH_WORDS_BY_LOCALE: dict[str, frozenset[str]] = {
    "en-US": frozenset({
        "what", "where", "when", "why", "who", "how", "which", "whose", "whom",
    }),
    "pt-BR": frozenset({
        "que", "quê", "qual", "quais", "quem", "onde", "quando", "como",
        "quanto", "quanta", "quantos", "quantas", "cadê", "porque", "por",
    }),
    "pt-PT": frozenset({
        "que", "quê", "qual", "quais", "quem", "onde", "quando", "como",
        "quanto", "quanta", "quantos", "quantas", "porque", "por",
    }),
    "it-IT": frozenset({
        "che", "cosa", "chi", "dove", "quando", "come", "perché", "perche",
        "quale", "quali", "quanto", "quanta", "quanti", "quante", "qual",
    }),
    "de-DE": frozenset({
        "was", "wer", "wo", "wann", "warum", "wie", "welche", "welcher",
        "welches", "welchen", "welchem", "wem", "wen", "wessen", "wieso",
        "weshalb", "woher", "wohin", "womit", "wofür",
    }),
    "es-ES": frozenset({
        "qué", "que", "quién", "quiénes", "quien", "quienes", "dónde",
        "donde", "cuándo", "cuando", "cómo", "como", "cuál", "cuáles",
        "cual", "cuales", "cuánto", "cuánta", "cuántos", "cuántas",
        "cuanto", "cuanta", "cuantos", "cuantas", "por",
    }),
}
_WH_WORDS_BY_LOCALE["es-MX"] = _WH_WORDS_BY_LOCALE["es-ES"]


def _is_polar_question(text: str, locale: str | None = None) -> bool:
    """A yes/no question: ends with '?' and does not open with a wh-word.

    Fails closed on an unlisted locale. Without that locale's interrogative
    openers there is no way to tell a wh-question from a polar one, and
    guessing polar would apply a contour the rule does not license. Returning
    False costs a contour on genuine yes/no questions in languages nobody has
    listed; claiming True would assert something unverified.
    """

    stripped = text.strip()
    if not stripped.endswith("?"):
        return False
    wh_words = _WH_WORDS_BY_LOCALE.get(locale or "")
    if wh_words is None:
        return False
    tokens = "".join(
        char if (char.isalpha() or char.isspace()) else " " for char in stripped
    ).split()
    return bool(tokens) and tokens[0].lower() not in wh_words


# ---------------------------------------------------------------------------
# Input normalization for the two Misaki-pipeline locales.
#
# The 28 espeak locales expand digits natively, so "Ich habe 7 Katzen" always
# rendered. The English and Portuguese adapters carry a Kokoro-era honesty
# gate that rejects non-word tokens instead, so "I have 7 cats" failed closed.
# Spelling simple integers out routes them through the same verified G2P path
# as any other word — nothing is guessed. Anything harder than a bare integer
# ("3.14", "2nd", "1,000") still fails closed, now with a message that names
# the token.
# ---------------------------------------------------------------------------
_EN_ONES = ("zero one two three four five six seven eight nine ten eleven "
            "twelve thirteen fourteen fifteen sixteen seventeen eighteen "
            "nineteen").split()
_EN_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty",
            "seventy", "eighty", "ninety")
_PT_ONES = ("zero um dois três quatro cinco seis sete oito nove dez onze "
            "doze treze catorze quinze dezesseis dezessete dezoito "
            "dezenove").split()
_PT_TENS = ("", "", "vinte", "trinta", "quarenta", "cinquenta", "sessenta",
            "setenta", "oitenta", "noventa")
_PT_HUNDREDS = ("", "cento", "duzentos", "trezentos", "quatrocentos",
                "quinhentos", "seiscentos", "setecentos", "oitocentos",
                "novecentos")


def _en_number_words(n: int) -> str:
    if n < 20:
        return _EN_ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _EN_TENS[tens] + (f" {_EN_ONES[ones]}" if ones else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        head = f"{_EN_ONES[hundreds]} hundred"
        return head + (f" {_en_number_words(rest)}" if rest else "")
    thousands, rest = divmod(n, 1000)
    head = f"{_en_number_words(thousands)} thousand"
    return head + (f" {_en_number_words(rest)}" if rest else "")


def _pt_number_words(n: int) -> str:
    if n < 20:
        return _PT_ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _PT_TENS[tens] + (f" e {_PT_ONES[ones]}" if ones else "")
    if n == 100:
        return "cem"
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        head = _PT_HUNDREDS[hundreds]
        return head + (f" e {_pt_number_words(rest)}" if rest else "")
    thousands, rest = divmod(n, 1000)
    head = "mil" if thousands == 1 else f"{_pt_number_words(thousands)} mil"
    if not rest:
        return head
    joiner = " e " if rest < 100 or rest % 100 == 0 else " "
    return head + joiner + _pt_number_words(rest)


_NUMBER_WORDS = {"en-US": _en_number_words, "pt-BR": _pt_number_words}
# A bare integer with no adjacent digit punctuation or letters: "7" and
# "2024" qualify; "3.14", "1,000" and "2nd" deliberately do not.
_BARE_INT_RE = re.compile(r"(?<![\w.,])\d{1,6}(?![\w.,])")


def _normalize_source_text(locale: str, text: str) -> str:
    if locale == "en-US":
        # English diacritics are decoration, not phonology: naïve -> naive.
        # Never applied to any other locale, where marks are contrastive.
        text = unicodedata.normalize("NFC", "".join(
            ch for ch in unicodedata.normalize("NFD", text)
            if unicodedata.category(ch) != "Mn"
        ))
    speller = _NUMBER_WORDS.get(locale)
    if speller:
        text = _BARE_INT_RE.sub(lambda m: speller(int(m.group())), text)
    return text


def build_pair(
    text: str,
    profile_id: str,
    *,
    voice: str | None = None,
    source_analysis: Any | None = None,
) -> dict[str, Any]:
    """Build the neutral/lens/speaker SSML set for one text in one direction.

    ``source_analysis`` replaces the G2P step with a caller-supplied
    ``SourceAnalysis``. Everything downstream — rule matching, epenthesis,
    deletion, stress, contour, receipts, the fail-closed refusal — reads the
    analysis and not the text, so a prepared one passes through untouched.
    Gibberish mode uses it to hand the lens a sentence of pseudowords; the
    typed text still supplies the sentence type and the on-screen wording.
    """

    from .bilingual_vowel_engine import (
        _BP_LEGAL_CODA_CATEGORIES,
        _COMMON_OPACITY_SYMBOLS,
        _CONSONANT_CLASS_BY_SYMBOL,
        _EPENTHESIS_OBSTRUENTS,
    )

    profile = _load_profile(profile_id)
    locale = profile.get("source_locale") or PROFILE_LOCALES.get(profile_id)
    if locale is None:
        raise AzureLensBuilderError(f"unsupported profile for Azure lane: {profile_id}")
    try:
        table = load_ipa_map()[locale]
    except KeyError as exc:
        raise AzureLensBuilderError(f"no Azure IPA map for locale: {locale}") from exc
    # Phones the caller prepared are authoritative: it built them for a
    # reason the text cannot express, so the G2P is skipped rather than run
    # and discarded, and the written form drops to a display label.
    phones_supplied = source_analysis is not None
    if phones_supplied:
        analysis = source_analysis
        normalized_input = analysis.normalized_text
    else:
        adapter = _load_adapter(locale, profile)
        normalized_input = _normalize_source_text(locale, text.strip())
        try:
            analysis = adapter.analyze(normalized_input)
        except Exception as exc:
            # The frozen English engine refuses non-word tokens with a generic
            # message, and its bytes are hash-bound evidence that must not
            # change. Naming the offending token happens here instead: after
            # digit spelling, any remaining digit-bearing token is the
            # refusal's cause ("3.14", "1,000", "2nd").
            if getattr(exc, "code", None) == "unsupported_nonword_token":
                token = next(
                    (t.strip(".,;:!?") for t in normalized_input.split()
                     if any(ch.isdigit() for ch in t)), None)
                if token:
                    raise AzureLensBuilderError(
                        f"The token '{token}' is not a word this lane can "
                        "verify; write numbers and symbols out as words."
                    ) from exc
            raise

    changed_vowel_rules = [
        rule for rule in profile["vowel_rules"] if rule["source"] != rule["target"]
    ]
    changed_consonant_rules = [
        rule
        for rule in profile.get("consonant_rules", ())
        if rule.get("operation") != "delete"
        and rule.get("source") != rule.get("target")
    ]
    deletion_rules = [
        rule
        for rule in profile.get("consonant_rules", ())
        if rule.get("operation") == "delete"
    ]
    insertion_rules = [
        rule
        for rule in profile.get("insertion_rules", ())
        if rule.get("operation") in ("insert_after", "insert_before")
    ]
    stress_rules = [
        rule
        for rule in profile.get("prosody_rules", ())
        if rule.get("operation") in _STRESS_OPERATIONS
    ]
    # The render happens on the source locale's voice, so that voice's stress
    # receipt decides whether moving a mark can be heard at all. An unprobed
    # locale is treated as honouring, matching how unprobed segment pairs are
    # treated; every probed voice currently ignores stress position.
    stress_honoured = load_stress_honour().get(locale) != "ignored"
    contour_rule = next(
        (
            rule
            for rule in profile.get("prosody_rules", ())
            if rule.get("operation") == "map_final_rise_fall_to_fall"
        ),
        None,
    )
    # The text is in the source language, so its interrogative openers are the
    # ones that decide whether this "?" is polar.
    polar_question = _is_polar_question(analysis.normalized_text, locale)
    contour_applied = contour_rule is not None and polar_question
    # Which codas the listener's language tolerates is language-specific, so an
    # insertion rule may carry its own set; the Brazilian table stays the
    # default for the pair that was authored against it.
    legal_codas = (
        frozenset(insertion_rules[0]["legal_codas"])
        if insertion_rules and "legal_codas" in insertion_rules[0]
        else _BP_LEGAL_CODA_CATEGORIES
    )
    locale_consonants = _COMMON_OPACITY_SYMBOLS.get(locale) or _GENERAL_CONSONANTS
    map_normalization = _MAP_NORMALIZATION_BY_LOCALE.get(locale, "NFD")
    vowel_symbols = frozenset(
        symbol
        for rule in changed_vowel_rules
        for symbol in rule["source"]
        if symbol != _COMBINING_TILDE
    )
    words: list[WordPlan] = []
    for word in analysis.words:
        lens_phone, applied = _apply_segment_rules(
            word.phone, changed_vowel_rules, changed_consonant_rules, vowel_symbols
        )
        # Stress ops run for their match bookkeeping either way; the moved
        # mark only reaches the ph string when this voice can render it.
        # Shipping a mark the voice provably ignores would make the SSML
        # differ while the audio does not — the fail-open shape again.
        stressed_phone, stressed = _apply_stress_prosody(
            word.phone, lens_phone, stress_rules, vowel_symbols
        )
        stressed_phone, oxytone = _shift_primary_stress_to_final(
            word.phone, stressed_phone, stress_rules, vowel_symbols
        )
        stressed = stressed + oxytone
        if stress_honoured:
            lens_phone = stressed_phone
        lens_phone, inserted = _apply_epenthesis(
            word.phone,
            lens_phone,
            insertion_rules,
            obstruents=_EPENTHESIS_OBSTRUENTS,
            legal_codas=legal_codas,
            following_consonants=_CONSONANT_CLASS_BY_SYMBOL,
            locale_consonants=locale_consonants,
        )
        lens_phone, deleted = _apply_deletions(lens_phone, deletion_rules)
        applied = applied + stressed + inserted + deleted
        context = f"word {word.source!r}"
        words.append(
            WordPlan(
                word_index=word.word_index,
                written=word.source,
                source_phone=word.phone,
                lens_phone=lens_phone,
                applied=applied,
                mapped_neutral=_map_symbols(
                    word.phone, table, context=context, normalization=map_normalization
                ),
                mapped_lens=_map_symbols(
                    lens_phone, table, context=context, normalization=map_normalization
                ),
            )
        )
    # A polar question carries an audible neutral/lens difference in its final
    # contour alone, so it renders even when no segment rule survives mapping.
    if not any(row.affected for row in words) and not contour_applied:
        raise AzureLensBuilderError(
            "no supported changed rule survives mapping in this text"
        )

    # Generated matrix profiles carry their own voice; the curated ones fall
    # back to the per-locale default. A receipt is only evidence for the voice
    # it was taken on, so the profile's choice wins.
    selected_voice = voice or profile.get("voice") or DEFAULT_VOICES.get(locale)
    if not selected_voice:
        raise AzureLensBuilderError(f"no Azure voice registered for {locale}")
    changed_by_id = {
        rule["id"]: rule
        for rule in (
            *changed_vowel_rules,
            *changed_consonant_rules,
            *deletion_rules,
            *insertion_rules,
            *stress_rules,
            *([contour_rule] if contour_rule else []),
        )
    }

    def _rule_survives_map(rule: dict[str, Any]) -> bool:
        """True when the map keeps the rule's source and target distinct.

        Mapping is per-symbol and context-free, so a substitution that
        collapses to the same Azure IPA on both sides (for example ə→ɐ→ə in
        the en map) renders no audible difference and is reported as
        map-neutralized, even when it rode inside a word another rule already
        changed. Insertions add a whole vowel token, so they always survive.
        """

        if "source" not in rule:
            return True
        # Removing a segment always changes the ph string; there is no target
        # for the map to collapse the source onto.
        if rule.get("operation") == "delete":
            return True
        try:
            return _map_symbols(
                rule["source"],
                table,
                context="rule-source",
                normalization=map_normalization,
            ) != _map_symbols(
                str(rule["target"]),
                table,
                context="rule-target",
                normalization=map_normalization,
            )
        except AzureLensBuilderError:
            return True

    audibility = load_rule_audibility()

    def _renderer_collapses(rule: dict[str, Any]) -> bool:
        """True when this voice renders the rule's two phones identically.

        The map can keep a pair distinct while the voice still collapses it:
        Azure's Hindi voice accepts ʋ and v and then returns byte-identical
        audio. That is the fail-open case — the lens ran, the SSML differed,
        and nothing was heard — so an audibility receipt from
        probe_rule_distinctness_v1 rides on the rule and demotes it here
        rather than letting it count as applied.
        """

        # Stress operations carry no segment pair; their receipt is the
        # per-voice stress probe. Every probed voice currently ignores
        # stress position, so a matched stress rule reports as inaudible
        # rather than applied.
        if rule.get("operation") in _STRESS_OPERATIONS:
            return not stress_honoured
        verdict = rule.get("renderer_verdict")
        if verdict is None:
            source, target = rule.get("source"), rule.get("target")
            if source and target:
                verdict = audibility.get(f"{locale}|{source}|{target}")
        return verdict == "inaudible"

    matched_rule_ids = {rule_id for row in words for rule_id in row.applied_rule_ids}
    if contour_applied:
        matched_rule_ids.add(contour_rule["id"])
    applied_rule_ids = sorted(
        rule_id
        for rule_id in matched_rule_ids
        if _rule_survives_map(changed_by_id[rule_id])
        and not _renderer_collapses(changed_by_id[rule_id])
    )
    map_neutralized_rule_ids = sorted(
        rule_id
        for rule_id in matched_rule_ids
        if not _rule_survives_map(changed_by_id[rule_id])
    )
    # Matched, distinct in the map, and still inaudible on this voice.
    renderer_inaudible_rule_ids = sorted(
        rule_id
        for rule_id in matched_rule_ids
        if _rule_survives_map(changed_by_id[rule_id])
        and _renderer_collapses(changed_by_id[rule_id])
    )
    # Implemented rules whose source segment / cited context is simply not
    # present in this sentence — a "not triggered here", never a coverage gap.
    context_absent_rule_ids = sorted(set(changed_by_id) - matched_rule_ids)
    # The guard above is ph-based, so it still passes when every matched rule
    # is one this voice renders identically. That combination produced a
    # payload with differing SSML, zero affected words and an empty applied
    # list — which no consumer can accept, and which reached the page as a
    # bare Worker contract error. Refuse here instead, naming the reason, and
    # save the two Azure calls that would have rendered two identical takes.
    if not applied_rule_ids and not contour_applied:
        collapsed = len(renderer_inaudible_rule_ids)
        raise AzureLensBuilderError(
            "no audible listener-lens shift in this text: "
            + (
                f"{collapsed} matched rule{'s' if collapsed != 1 else ''} "
                "land on sound pairs this voice renders identically"
                if collapsed
                else "no rule matched"
            )
        )
    # Every profile rule family now renders; nothing is silently skipped.
    omitted_rule_ids: list[str] = []

    def ssml(side: str) -> str:
        pieces: list[str] = []
        for row in words:
            # An untouched word normally renders as text, so the voice reads
            # it naturally instead of being steered toward a transcription of
            # its own pronunciation. That is the wrong default when the caller
            # supplied the phones: there the written form is a label for a
            # word that was never meant to be said, and leaving one untagged
            # would drop a real word into the middle of the pseudo-sentence.
            if row.affected or phones_supplied:
                mapped = row.mapped_neutral if side == "neutral" else row.mapped_lens
                if " " in mapped:
                    raise AzureLensBuilderError(
                        f"ph attribute contains whitespace for {row.written!r}"
                    )
                pieces.append(
                    f"<phoneme alphabet=\"ipa\" ph={quoteattr(mapped)}>"
                    f"{escape(row.written)}</phoneme>"
                )
            else:
                pieces.append(escape(row.written))
        # The final punctuation is the deterministic contour control. The
        # neutral side keeps the typed sentence type; a polar question's lens
        # side renders the cited rise-fall-to-fall mapping as a final fall.
        final_punctuation = (
            "?" if analysis.normalized_text.strip().endswith("?") else "."
        )
        if side == "lens" and contour_applied:
            final_punctuation = "."
        body = " ".join(pieces) + final_punctuation
        return (
            '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            f'xml:lang="{locale}"><voice name="{selected_voice}">{body}</voice></speak>'
        )

    # The speaker track: the listener language's own voice reading the raw
    # source text. Production, not perception — a German mouth attempting the
    # sentence, everything colored at once — so it involves no rules, no
    # G2P and no phoneme tags, and it is labeled as a different claim.
    listener_locale = profile.get("listener_locale") or (
        "pt-BR" if locale == "en-US" else "en-US"
    )
    speaker_voice = speaker_voice_for(listener_locale)
    ssml_speaker = (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        f'xml:lang="{listener_locale}"><voice name="{speaker_voice}">'
        f"{escape(analysis.normalized_text)}</voice></speak>"
    )

    return {
        "lane_version": AZURE_LANE_VERSION,
        "profile_id": profile_id,
        "locale": locale,
        "voice": selected_voice,
        "listener_locale": listener_locale,
        "speaker_voice": speaker_voice,
        "normalized_text": analysis.normalized_text,
        "ssml_neutral": ssml("neutral"),
        "ssml_lens": ssml("lens"),
        "ssml_speaker": ssml_speaker,
        "words": [
            {
                "word_index": row.word_index,
                "written": row.written,
                "source_phone": row.source_phone,
                "lens_phone": row.lens_phone,
                "applied_rule_ids": list(row.applied_rule_ids),
            }
            for row in words
        ],
        "applied_rule_ids": applied_rule_ids,
        "map_neutralized_rule_ids": map_neutralized_rule_ids,
        "context_absent_rule_ids": context_absent_rule_ids,
        "renderer_inaudible_rule_ids": renderer_inaudible_rule_ids,
        # A word counts as shifted only when at least one rule on it is
        # audible on this voice. A word carrying nothing but collapsed pairs
        # has a different ph string and identical audio, so counting it would
        # promise a change the listener cannot hear.
        "affected_word_count": sum(
            1
            for row in words
            if row.affected and set(row.applied_rule_ids) & set(applied_rule_ids)
        ),
        "omitted_rule_ids": omitted_rule_ids,
        "prosody": {
            "polar_question": polar_question,
            "contour_applied": contour_applied,
        },
        "api_calls_made": 0,
    }


def load_local_env(path: Path = ROOT / ".env.local") -> dict[str, str]:
    values: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip()
    return values


def render_ssml_bytes(ssml: str, *, key: str, region: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1",
        data=ssml.encode("utf-8"),
        headers={
            "Ocp-Apim-Subscription-Key": key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": OUTPUT_FORMAT,
            "User-Agent": "build-week-azure-lens-lane/1",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:400]
        return {"http_status": error.code, "rendered": False, "error_body": detail}
    return {"http_status": 200, "rendered": True, "wav_bytes": payload}


def render_ssml(ssml: str, destination: Path, *, key: str, region: str) -> dict[str, Any]:
    result = render_ssml_bytes(ssml, key=key, region=region)
    if not result["rendered"]:
        return result
    payload = result.pop("wav_bytes")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return {**result, "bytes": len(payload)}
