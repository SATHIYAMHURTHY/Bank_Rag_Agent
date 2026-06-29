"""
scraper.py
----------
Hybrid async scraper with TTL cache.

Two fetch strategies, chosen per source via the js_render flag in sources.py:
  - aiohttp   (fast, no browser overhead) for static HTML pages
  - Playwright (real Chromium browser)    for JS-rendered pages

Fixes applied:
  - ICICI: large CSP headers  -> increased aiohttp header size limit
  - SBI:   self-signed SSL    -> SSL verification disabled for sbi.co.in
  - BOB/PNB/Axis/Kotak etc:  -> Playwright handles JS rendering
"""

import json
import asyncio
import aiohttp
import aiofiles
import re

from pathlib import Path
from datetime import datetime, timezone
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

from src.scraper.sources import SOURCES

RAW_DATA_DIR    = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

TTL_HOURS       = 24
MAX_CONCURRENT  = 3
REQUEST_TIMEOUT = 30
PLAYWRIGHT_TIMEOUT = 20000   # ms — Playwright uses milliseconds

NOISE_TAGS    = ["script", "style", "nav", "footer", "header", "aside", "form"]
NOISE_PHRASES = ["cookie", "javascript", "subscribe", "newsletter", "follow us"]

# ── Skip-until markers (disclaimer before real content) ───────────────────────
SKIP_UNTIL_MARKERS = [
    "i agree",
    "i agree\n",
    "link for providing suggestions",
]

# ── Block-start markers (drop everything after this point) ────────────────────
BLOCK_START_MARKERS = [
    "quick links",
    "you may also like",
    "other products",
    "related products",
    "explore more",
    "was this helpful",
    "rate this page",
    "is typing",
    "characters remaining",
    "your feedback matters",
    "thank you for your feedback",
    "goto previous card",
    "close disclaimer modal",
    "stay on this site",
    "see products",
]

# ── Individual line noise patterns ────────────────────────────────────────────
NOISE_LINE_PATTERNS = [
    "apply now", "know more", "click here", "read more",
    "learn more", "view all", "show more", "show less",
    "more information", "imp. note", "get a call back",
    "t&c apply", "t & c apply", "*t&c", "last updated on",
    "all rights reserved", "terms and conditions apply",
    "cookie policy", "privacy policy",
    "back to top", "skip to main content",
    "follow us on", "download the app",
    "scan the qr code",
    "w.e.f",
    "tools & calculators",
    "unauthorized digital transaction",
    "credila financial services limited",
    "goto previous", "goto next", "proceed",
]

# ── Exact line matches ────────────────────────────────────────────────────────
PURE_UI_LINES = {
    "apply now", "know more", "compare", "submit",
    "features", "eligibility", "criteria",
    "next", "previous", "back", "close",
    "poor", "average", "good", "very good", "outstanding",
    "disclaimer",
}

# ── Structural regex patterns ─────────────────────────────────────────────────
GENERIC_UI_PATTERNS = [
    re.compile(r"^\s*[\d,]+\s*$"),
    re.compile(r"^\s*[\d.]+\s*%\s*$"),
    re.compile(r"^w\.e\.f\b", re.IGNORECASE),
    re.compile(r"^\s*\*+\s*$"),
    re.compile(r"^p\.a\.\*?$", re.IGNORECASE),
    re.compile(r"^start from$", re.IGNORECASE),
    re.compile(r"^starts from$", re.IGNORECASE),
]

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


# ── TTL cache check ───────────────────────────────────────────────────────────

def is_data_fresh(source: dict, ttl_hours: int = TTL_HOURS) -> bool:
    filename = f"{source['bank']}_{source['scheme_name']}.json".replace(" ", "_")
    filepath = RAW_DATA_DIR / filename
    if not filepath.exists():
        return False
    try:
        data       = json.loads(filepath.read_text(encoding="utf-8"))
        scraped_at = datetime.fromisoformat(data["scraped_at"])
        age        = datetime.now(timezone.utc) - scraped_at
        return age.total_seconds() < ttl_hours * 3600
    except Exception:
        return False


# ── HTML cleaning ─────────────────────────────────────────────────────────────

def clean_html_to_text(html: str) -> str:
    """Stage 1: Strip HTML noise tags, extract text, dedupe consecutive lines."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(NOISE_TAGS):
        tag.decompose()
    raw_text = soup.get_text(separator="\n")
    lines = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if any(phrase in line.lower() for phrase in NOISE_PHRASES):
            continue
        lines.append(line)
    deduped = []
    for line in lines:
        if not deduped or line != deduped[-1]:
            deduped.append(line)
    return "\n".join(deduped)


def remove_repeated_sections(text: str, min_length: int = 200) -> str:
    """Stage 4: Remove verbatim repeated blocks (e.g. SBI rate widget x2)."""
    for start in range(0, len(text) - min_length, 50):
        chunk  = text[start:start + min_length]
        second = text.find(chunk, start + min_length)
        if second != -1:
            return text[:second].rstrip()
    return text


def clean_bank_specific_noise(text: str, bank: str) -> str:
    """
    Stage 2+3: Generic noise removal across any bank.

    Strategy 1 - Skip-until:  skip top-of-page disclaimers (SBI, IndianBank)
    Strategy 2 - Block removal: drop sidebars/widgets after real content ends
    Strategy 3 - Line filtering: drop individual noise lines
    Strategy 4 - Section dedup: remove repeated blocks
    """
    lines          = text.split("\n")
    result         = []
    all_text_lower = text.lower()

    has_skip_until  = any(m in all_text_lower for m in SKIP_UNTIL_MARKERS)
    skipping_prefix = has_skip_until
    skip_rest       = False

    for line in lines:
        stripped   = line.strip()
        line_lower = stripped.lower()

        # Strategy 1: skip-until
        if skipping_prefix:
            if any(marker in line_lower for marker in SKIP_UNTIL_MARKERS):
                skipping_prefix = False
            continue

        # Strategy 2: block removal
        if not skip_rest:
            if any(marker in line_lower for marker in BLOCK_START_MARKERS):
                skip_rest = True
        if skip_rest:
            continue

        # Strategy 3a: structural regex
        if any(p.match(stripped) for p in GENERIC_UI_PATTERNS):
            continue

        # Strategy 3b: noise substrings
        if any(pattern in line_lower for pattern in NOISE_LINE_PATTERNS):
            continue

        # Strategy 3c: exact UI labels
        if line_lower in PURE_UI_LINES:
            continue

        # Strategy 3d: too short with no letters
        if len(stripped) < 6 and not any(c.isalpha() for c in stripped):
            continue

        result.append(line)

    cleaned = "\n".join(result)
    cleaned = remove_repeated_sections(cleaned)
    return cleaned


# ── Fetch strategies ──────────────────────────────────────────────────────────

async def fetch_aiohttp(
    session: aiohttp.ClientSession,
    url: str,
    semaphore: asyncio.Semaphore,
) -> str:
    """Fast static HTML fetcher — for pages that don't need JS execution."""
    async with semaphore:
        async with session.get(
            url,
            headers=BROWSER_HEADERS,
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
        ) as response:
            response.raise_for_status()
            return await response.text()


async def fetch_playwright(
    url: str,
    semaphore: asyncio.Semaphore,
    browser,
    extra_wait_ms: int = 1500,
    wait_until: str = "networkidle",
) -> str:
    """
    Real browser fetcher for JS-rendered pages.
    Launches a new browser context per request (clean slate, no cookie leakage).
    Waits for networkidle so JS content finishes loading before extracting HTML.
    """
    async with semaphore:
        context = await browser.new_context(
            user_agent=BROWSER_HEADERS["User-Agent"],
            ignore_https_errors=True,   # handles self-signed certs (SBI, some others)
        )
        page = await context.new_page()
        try:
            await page.goto(
                url,
                wait_until="networkidle",   # wait for JS to finish
                timeout=PLAYWRIGHT_TIMEOUT,
            )
            # Extra wait for pages with lazy-loading or delayed renders
            await page.wait_for_timeout(1500)
            html = await page.content()
            return html
        finally:
            await context.close()


# ── Save ──────────────────────────────────────────────────────────────────────

async def save_record_async(record: dict) -> None:
    filename = f"{record['bank']}_{record['scheme_name']}.json".replace(" ", "_")
    out_path = RAW_DATA_DIR / filename
    async with aiofiles.open(out_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(record, indent=2, ensure_ascii=False))


# ── Per-source scrape orchestration ──────────────────────────────────────────

async def scrape_source_async(
    source: dict,
    index: int,
    total: int,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    browser,
) -> str:
    label = f"[{index}/{total}] {source['bank']} - {source['scheme_name']}"
    js_render = source.get("js_render", False)

    try:
        if js_render:
            bank = source["bank"]
            wait_until = source.get("playwright_wait", "networkidle")
            extra_wait = (
                5000 if bank == "IndianBank"
                else 3000 if bank == "Canara"
                else 1500
            )

            html = await fetch_playwright(
                source["url"],
                semaphore,
                browser,
                extra_wait_ms=extra_wait,
                wait_until=wait_until,
            )
        else:
            html = await fetch_aiohttp(
                session,
                source["url"],
                semaphore,
            )

        text = clean_html_to_text(html)
        text = clean_bank_specific_noise(text, source["bank"])

        record = {
            "url": source["url"],
            "bank": source["bank"],
            "scheme_type": source["scheme_type"],
            "scheme_name": source["scheme_name"],
            "doc_type": source.get("doc_type", "unknown"),
            "js_render": js_render,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "content": text,
        }

        await save_record_async(record)

        return f"[OK] {label} - {len(text)} chars"

    except Exception as e:
        return f"[FAIL] {label} - {e}"


# ── Main orchestrator ─────────────────────────────────────────────────────────

async def scrape_all_async(ttl_hours: int = TTL_HOURS) -> dict:
    fresh_sources = []
    stale_sources = []

    for source in SOURCES:
        if is_data_fresh(source, ttl_hours):
            fresh_sources.append(source)
            print(f"[FRESH] {source['bank']} - {source['scheme_name']}")
        else:
            stale_sources.append(source)
            print(f"[STALE] {source['bank']} - {source['scheme_name']}")

    results = {"skipped": fresh_sources, "scraped": [], "failed": []}

    if not stale_sources:
        print(f"\nAll {len(fresh_sources)} sources fresh - skipping scraping!")
        return results

    has_playwright = any(s.get("js_render", False) for s in stale_sources)
    print(f"\nScraping {len(stale_sources)} stale sources...")
    print(f"Strategy: aiohttp for static, Playwright for JS-rendered pages\n")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    connector = aiohttp.TCPConnector(ssl=False)

    async with async_playwright() as pw:
        # Launch ONE shared browser instance — much faster than launching per-page
        browser = await pw.chromium.launch(headless=True) if has_playwright else None

        async with aiohttp.ClientSession(
            connector=connector,
            connector_owner=True,
            max_line_size=32768,
            max_field_size=32768,
        ) as session:
            tasks = [
                scrape_source_async(
                    source, i + 1, len(stale_sources),
                    session, semaphore, browser
                )
                for i, source in enumerate(stale_sources)
            ]
            task_results = await asyncio.gather(*tasks)

        if browser:
            await browser.close()

    for result in task_results:
        print(result)
        if result.startswith("[OK]"):
            results["scraped"].append(result)
        else:
            results["failed"].append(result)

    print(f"\nDone: {len(fresh_sources)} skipped, "
          f"{len(results['scraped'])} scraped, "
          f"{len(results['failed'])} failed")

    return results


def scrape_all(ttl_hours: int = TTL_HOURS) -> dict:
    return asyncio.run(scrape_all_async(ttl_hours))


if __name__ == "__main__":
    import time
    start   = time.time()
    results = scrape_all()
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")