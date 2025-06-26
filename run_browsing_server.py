# run_browsing_server.py (FINAL FLASK IMPLEMENTATION)

import sys
import logging
from flask import Flask, request, jsonify
from waitress import serve

# Ensure other project modules can be imported
sys.path.append('.')

# Import the tools that were proven to work in our diagnostics
from duckduckgo_search import DDGS
from tts_engine.playwright_browser import scrape_all

# Set up logging for visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# This endpoint replaces the Serper API call locally.
@app.route('/search', methods=['POST'])
def search():
    """Receives a search query, uses DuckDuckGo to get links, and returns them."""
    data = request.get_json()
    query = data.get('q')
    logging.info(f"Flask Server: Received search for '{query}'")

    with DDGS() as ddgs:
        # Get a few more results to increase the chance of finding scrapable content
        results = list(ddgs.text(query, max_results=5))

    # Format results to be perfectly compatible with web_tools.py
    formatted_results = {
        "organic": [
            {"link": r.get("href"), "title": r.get("title"), "snippet": r.get("body")}
            for r in results if r.get("href")
        ]
    }
    logging.info(f"Flask Server: Found {len(results)} links via DuckDuckGo.")
    return jsonify(formatted_results)


# This endpoint uses our robust Playwright function.
@app.route('/browse', methods=['POST'])
def browse():
    """Receives URLs, scrapes them with Playwright, and returns the content."""
    data = request.get_json()
    urls = data.get('urls')
    logging.info(f"Flask Server: Received browse request for {len(urls)} URLs.")

    # Run the synchronous Playwright function directly.
    # Waitress (the server) will handle running this in a worker thread.
    results = scrape_all(urls)

    logging.info(f"Flask Server: Finished browsing.")
    return jsonify(results)


if __name__ == '__main__':
    print("--- Starting Flask Tool Server (Waitress) ---")
    # Use Waitress, a production-quality server for Flask apps
    serve(app, host='127.0.0.1', port=5007)