# playwright_browser.py (FINAL PRODUCTION VERSION)

import logging
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import time

# --- CONFIGURATION BASED ON SUCCESSFUL DIAGNOSTICS ---
# This is the mode that successfully bypassed Cloudflare.
HEADLESS = False
# This is the argument that keeps the window from appearing on-screen and stealing focus.
LAUNCH_ARGS = ["--window-position=-2000,0"]

# --- INTELLIGENT PARSING & SCRAPING LOGIC ---

def _scrape_page_in_context(page, url: str) -> dict:
    """
    Helper function to scrape content from an already-open page (tab).
    This contains the intelligent parsing logic.
    """
    logging.info(f"    - Scraping and parsing content from: {url}")
    try:
        # The page is already navigated, so we get the content.
        html_content = page.content()
        soup = BeautifulSoup(html_content, "html.parser")

        # 1. Remove all high-level junk tags first.
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
            tag.decompose()

        # 2. Intelligently find the main content block.
        main_content = soup.find("main") or soup.find("article") or soup.find("div", id="content")

        if main_content:
            # If a main block is found, use it for the cleanest text.
            text = main_content.get_text(separator=" ", strip=True)
        else:
            # Fallback to the whole body if no main block is found.
            text = soup.get_text(separator=" ", strip=True)

        # 3. Final cleanup of whitespace.
        cleaned_text = " ".join(text.split())
        return {"url": url, "content": cleaned_text}

    except Exception as e:
        logging.error(f"    - Failed to parse content for {url}: {e}")
        return {"url": url, "error": f"Failed to parse content: {e}"}

def scrape_all(urls: list[str], debug: bool = False) -> list[dict]:
    """
    Launches ONE non-headless browser, off-screen. It opens each URL in a new tab,
    scrapes it, and closes the tab, reusing the single browser for maximum
    efficiency and minimal user interruption.
    """
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS, args=LAUNCH_ARGS)
        context = browser.new_context()
        logging.info("Single browser instance launched off-screen.")
        try:
            for url in urls:
                page = context.new_page()
                try:
                    page.goto(url, timeout=30000, wait_until="networkidle")
                    results.append(_scrape_page_in_context(page, url))
                except Exception as e:
                    logging.error(f"  -> Failed to navigate to {url}: {e}")
                    results.append({"url": url, "error": str(e)})
                finally:
                    page.close()
        finally:
            browser.close()
            logging.info("Single browser instance has been closed.")
    return results