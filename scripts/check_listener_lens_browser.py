#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    errors: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1050})
        page.on(
            "console",
            lambda message: errors.append(message.text)
            if message.type == "error"
            else None,
        )
        page.on("pageerror", lambda error: errors.append(str(error)))

        page.goto("http://127.0.0.1:8788/", wait_until="networkidle")
        assert "Listener Lens Lab" in page.title()
        source = "The black cat sat on the wooden bench."
        page.locator("#input").fill(source)
        page.get_by_role("button", name="Build listening versions").click()
        page.locator("#results").wait_for(state="visible")

        assert page.locator("#original").text_content() == source
        neutral = page.locator("#neutral").text_content() or ""
        lens = page.locator("#lens").text_content() or ""
        assert len(neutral) > 10
        assert len(neutral) == len(lens)
        assert neutral != lens
        neutral_marks = page.locator("#neutral mark").count()
        assert neutral_marks > 0
        assert neutral_marks == page.locator("#lens mark").count()
        assert page.locator(".take").count() == 3
        assert page.locator(".rule").count() == 2
        assert "Dictionary + G2P" in (page.locator("#gate").text_content() or "")
        assert not page.locator(".carrier-play").first.is_disabled()
        assert errors == []
        page.wait_for_timeout(700)

        screenshot = ROOT / "artifacts" / "prototypes" / "listener-lens.png"
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(screenshot), full_page=True)

        page.locator("#input").fill("The day today.")
        page.get_by_role("button", name="Build listening versions").click()
        page.wait_for_function(
            "document.querySelector('#neutral').textContent === "
            "document.querySelector('#lens').textContent"
        )
        assert page.locator("#neutral").text_content() == page.locator(
            "#lens"
        ).text_content()
        assert page.locator(".carrier-play").first.is_disabled()
        assert "playback is disabled" in (
            page.locator("#rules").text_content() or ""
        )

        page.set_viewport_size({"width": 390, "height": 844})
        page.reload(wait_until="networkidle")
        page.locator("#input").fill(source)
        page.get_by_role("button", name="Build listening versions").click()
        page.locator("#results").wait_for(state="visible")
        page.wait_for_timeout(700)
        mobile_overflow = page.evaluate(
            "document.documentElement.scrollWidth - window.innerWidth"
        )
        assert mobile_overflow <= 1
        mobile_screenshot = (
            ROOT / "artifacts" / "prototypes" / "listener-lens-mobile.png"
        )
        page.screenshot(path=str(mobile_screenshot), full_page=True)
        assert errors == []
        browser.close()

    print(
        json.dumps(
            {
                "cards": 3,
                "highlighted_slots": neutral_marks,
                "no_rule_playback_disabled": True,
                "mobile_overflow_px": mobile_overflow,
                "browser_errors": errors,
                "screenshot": str(screenshot),
                "mobile_screenshot": str(mobile_screenshot),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
