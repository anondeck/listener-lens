#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
URL = os.environ.get("PRODUCT_URL", "http://127.0.0.1:8789/")


def main() -> None:
    errors: list[str] = []
    activity_requests: list[dict] = []
    listener_requests: list[dict] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1050})
        page.on(
            "console",
            lambda message: (
                errors.append(message.text) if message.type == "error" else None
            ),
        )
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.route(
            "**/api/voices",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "schema_version": 1,
                        "registry_version": "kokoro-product-voices-v1",
                        "renderer": "kokoro",
                        "same_voice_pair_required": True,
                        "production_enabled": False,
                        "languages": [
                            {
                                "language_id": "en-US",
                                "display_name": "American English",
                                "default_voice_id": "af_heart",
                                "voices": [
                                    {"voice_id": "af_heart", "display_name": "Heart", "gender": "female", "current_narrow_runtime_available": True},
                                    {"voice_id": "am_michael", "display_name": "Michael", "gender": "male", "current_narrow_runtime_available": False},
                                ],
                            },
                            {
                                "language_id": "pt-BR",
                                "display_name": "Brazilian Portuguese",
                                "default_voice_id": "pm_alex",
                                "voices": [
                                    {"voice_id": "pm_alex", "display_name": "Alex", "gender": "male", "current_narrow_runtime_available": False},
                                    {"voice_id": "pf_dora", "display_name": "Dora", "gender": "female", "current_narrow_runtime_available": False},
                                ],
                            },
                        ],
                    }
                ),
            ),
        )

        def fulfill_listener_lens(route) -> None:
            listener_requests.append(route.request.post_data_json)
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "status": "ready",
                        "cache_hit": False,
                        "api_calls_made": 0,
                        "transform": {
                            "profile_id": "en-to-pt-BR-vowel-lens",
                            "voice_id": "af_heart",
                            "original_text": "What a great day it is to catch some sun.",
                            "neutral_script": "drayl dayk droh voht tohn playm prayr bavd brayn truhk.",
                            "lens_script": "drayl dayk droh voht tohn playm prayr behvd brayn truhk.",
                            "slots": [
                                {
                                    "word_index": 7,
                                    "neutral_character_span": [1, 2],
                                    "lens_character_span": [1, 3],
                                    "source_ipa": "æ",
                                    "target_ipa": "ɛ",
                                }
                            ],
                        },
                        "audio": {
                            "neutral": {"base64": "UklGRg==", "sha256": "neutral"},
                            "lens": {"base64": "UklGRg==", "sha256": "lens"},
                        },
                    }
                ),
            )

        page.route("**/api/listener-lens", fulfill_listener_lens)

        def fulfill_activity(route) -> None:
            activity_requests.append(route.request.post_data_json)
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "source": "cached_fallback",
                        "activity": {
                            "title": "Hear the mechanism, bound the claim",
                            "objective": "Use a meaning-opaque A/B to explore one cited sound-category substitution.",
                            "warmup": [
                                "Compare cat and bet.",
                                "Discuss what approximation means.",
                            ],
                            "listen_for": [
                                "Hear the highlighted slots.",
                                "Notice delivery differences.",
                            ],
                            "practice_steps": [
                                {
                                    "minutes": 4,
                                    "instruction": "Play A and B.",
                                    "teacher_note": "Focus on the highlighted slot.",
                                },
                                {
                                    "minutes": 8,
                                    "instruction": "Connect the rule card.",
                                    "teacher_note": "Keep claims bounded.",
                                },
                                {
                                    "minutes": 8,
                                    "instruction": "Rewrite one claim.",
                                    "teacher_note": "Do not claim private perception.",
                                },
                            ],
                            "exit_ticket": "Distinguish an approximation from private experience.",
                            "evidence_note": "The product implements a selected cited mechanism; it does not certify subjective fidelity.",
                        },
                    }
                ),
            )

        page.route("**/api/activity", fulfill_activity)
        page.goto(URL, wait_until="networkidle")

        assert "education prototype" in page.title()
        assert page.get_by_text("Typed listener-lens").count() == 1
        assert (
            page.get_by_text(
                "an evidence-informed, language-shaped approximation", exact=False
            ).count()
            >= 1
        )
        assert page.locator(".evidence-card").count() == 3
        assert page.locator("#source-voice").input_value() == "af_heart"
        assert page.locator("#source-voice option").count() == 2
        assert page.locator("#source-voice option[value='am_michael']").is_disabled()
        assert page.locator("#voice-roster strong").count() == 4
        assert page.get_by_text("Heart · American English", exact=True).count() == 1
        assert page.get_by_text("Alex · Brazilian Portuguese", exact=True).count() == 1

        page.get_by_role("button", name="Create comparison").click()
        page.locator("#lens-result").wait_for(state="visible")
        assert page.locator("#neutral-carrier mark").text_content() == "a"
        assert page.locator("#lens-carrier mark").text_content() == "eh"
        assert page.locator("#runtime-player").is_visible()
        assert "Controlled Kokoro synthesis passed" in (
            page.locator("#lens-status").text_content() or ""
        )
        assert listener_requests == [{
            "text": "What a great day it is to catch some sun.",
            "profile_id": "en-to-pt-BR-vowel-lens",
            "voice_id": "af_heart",
        }]
        assert "Heart · Kokoro AI-generated voice" in (
            page.locator("#runtime-voice-label").text_content() or ""
        )

        static_player = page.locator("[data-static-player]")
        page.locator(".static-fallback summary").click()
        altered = static_player.get_by_role("button", name="B Altered carrier")
        altered.click()
        assert altered.get_attribute("aria-pressed") == "true"
        assert (
            static_player.locator(".now-playing").text_content()
            == "B · static altered carrier"
        )
        assert (
            static_player.locator("audio").get_attribute("src")
            == "/audio/altered-carrier.wav"
        )

        page.locator("#grade-band").select_option("middle")
        page.get_by_role("button", name="Generate teaching activity").click()
        page.locator(".activity-result").wait_for(state="visible")
        result_source = page.locator(".result-source").text_content() or ""
        assert result_source in {"Generated with GPT-5.6", "Curated cached fallback"}
        evidence_note = (
            page.locator(".activity-result .note").text_content() or ""
        ).casefold()
        assert "selected cited mechanism" in evidence_note
        assert "subjective fidelity" in evidence_note
        assert len(activity_requests) == 1
        result_metadata = activity_requests[0]["result_metadata"]
        assert result_metadata == {
            "profile_id": "en-to-pt-BR-vowel-lens",
            "rule_ids": [],
            "changed_slot_count": 1,
            "comparison_status": "ready",
            "renderer_verification": "automatic_acoustic_and_pcm_gates_checked",
        }
        assert "original_text" not in json.dumps(activity_requests[0])

        desktop = ROOT / "artifacts" / "prototypes" / "product-desktop.png"
        desktop.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(desktop), full_page=True)

        page.set_viewport_size({"width": 390, "height": 844})
        page.reload(wait_until="networkidle")
        overflow = page.evaluate(
            "document.documentElement.scrollWidth - window.innerWidth"
        )
        assert overflow <= 1
        mobile = ROOT / "artifacts" / "prototypes" / "product-mobile.png"
        page.screenshot(path=str(mobile), full_page=True)

        injected = '<img src=x onerror="window.__listenerLensInjected=true">'
        page.unroute("**/api/listener-lens")
        page.route(
            "**/api/listener-lens",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "status": "no_supported_sounds",
                        "message": injected,
                        "api_calls_made": 0,
                    }
                ),
            ),
        )
        page.get_by_role("button", name="Create comparison").click()
        page.locator("#lens-status").filter(has_text=injected).wait_for(
            state="visible"
        )
        assert injected in (page.locator("#lens-status").text_content() or "")
        assert page.locator("#lens-status img").count() == 0
        assert page.evaluate("Boolean(window.__listenerLensInjected)") is False
        assert errors == [], errors
        browser.close()

    print(
        json.dumps(
            {
                "typed_listener_lens": True,
                "isolated_evidence_cards": 3,
                "activity_result": True,
                "activity_result_source": result_source,
                "mobile_overflow_px": overflow,
                "browser_errors": errors,
                "desktop_screenshot": str(desktop),
                "mobile_screenshot": str(mobile),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
