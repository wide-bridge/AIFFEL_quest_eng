"""
Capture BabyCoach Chat UI screenshots (3 states).

Why: This repo's UI is static HTML served by FastAPI, so a small browser
automation script is the simplest way to capture screenshots for the UX checks.

Usage (run locally):
  1) Start the server (e.g. uvicorn) on http://127.0.0.1:8002/
  2) Install Playwright:
       pip install playwright
       playwright install chromium
  3) Run:
       python scripts/capture_chat_ui_screenshots.py

Outputs:
  - chat_screenshots/1_chat_initial_*.png
  - chat_screenshots/2_chat_stacked_*.png
  - chat_screenshots/3_chat_input_fixed_*.png
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class ScreenshotResult:
    path: Path
    description: str


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


async def _ensure_chat_ready(page, base_url: str) -> None:
    # Load page
    await page.goto(base_url, wait_until="networkidle")
    await asyncio.sleep(1)

    try:
        # Empty chat screen shows the one-line coaching when `final_output` exists.
        # We'll generate a recommendation if it's missing.
        has_today_nudge = (await page.locator(".todayNudge").count()) > 0
        if has_today_nudge:
            await page.locator("#tabBtnChat").click()
            await asyncio.sleep(0.8)
            return

        # Generate recommendation via Spoon UI if possible.
        await page.locator("#tabBtnSpoon").click()
        await asyncio.sleep(0.5)

        # The Spoon tab UI differs by versions; we try common button text.
        btn = page.locator("button:has-text('Spoon 추천 받기')")
        if await btn.count() > 0:
            await btn.first.click()
            try:
                await page.wait_for_selector("#spoonStatus:has-text('완료')", timeout=40000)
            except Exception:
                await asyncio.sleep(4)
    except Exception:
        # If generation fails, we'll still capture screenshots of the chat layout.
        pass

    # Ensure chat tab active
    await page.locator("#tabBtnChat").click()
    await asyncio.sleep(0.8)

    # Wait a bit for the one-line coaching pill to render.
    try:
        if (await page.locator(".todayNudge").count()) == 0:
            await page.wait_for_selector(".todayNudge", timeout=8000)
    except Exception:
        pass


async def _scroll_to_bottom(chat_msgs) -> None:
    await chat_msgs.evaluate("el => el.scrollTop = el.scrollHeight")
    await asyncio.sleep(0.3)


async def main() -> None:
    from playwright.async_api import async_playwright

    base_url = "http://127.0.0.1:8002/"
    out_dir = Path("chat_screenshots")
    out_dir.mkdir(exist_ok=True)
    ts = _timestamp()

    results: list[ScreenshotResult] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page(viewport={"width": 420, "height": 900})

        await _ensure_chat_ready(page, base_url)

        # Screenshot 1: initial / empty-ish state
        await page.locator("#tabBtnChat").click()
        await asyncio.sleep(0.6)
        ss1 = out_dir / f"1_chat_initial_{ts}.png"
        await page.screenshot(path=str(ss1), full_page=False)
        results.append(ScreenshotResult(path=ss1, description="Header/tabs/chat area/input (initial)"))

        # Screenshot 2: click example question (start conversation)
        await page.locator("button:has-text('밥을 잘 안 먹어요')").click()
        await asyncio.sleep(2.8)
        ss2 = out_dir / f"2_chat_after_example_{ts}.png"
        await page.screenshot(path=str(ss2), full_page=False)
        results.append(ScreenshotResult(path=ss2, description="After clicking example question"))

        # Screenshot 3: stacked messages with scroll visible
        # (Send two more messages to ensure scrolling bubbles appear)
        messages = ["요즘 놀이를 금방 싫어해요", "상호작용을 더 늘리려면?"]
        chat_input = page.locator("#chatUserMessage")
        send_btn = page.locator("#chatSendBtn")

        for msg in messages:
            await chat_input.fill(msg)
            await asyncio.sleep(0.2)
            await send_btn.click()
            await asyncio.sleep(2.5)

        chat_msgs = page.locator("#chatMsgs")
        await _scroll_to_bottom(chat_msgs)
        ss3 = out_dir / f"3_chat_stacked_{ts}.png"
        await page.screenshot(path=str(ss3), full_page=False)
        results.append(ScreenshotResult(path=ss3, description="Stacked bubbles + chat scroll"))

        await browser.close()

    print("\nCaptured screenshots:")
    for r in results:
        print(f"- {r.path} ({r.description})")


if __name__ == "__main__":
    asyncio.run(main())

