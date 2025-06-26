# browsing.py (FINAL - SIMPLE & SELF-CONTAINED)

import asyncio
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from tts_engine.playwright_browser import get_search_links, scrape_all
from concurrent.futures import ThreadPoolExecutor

# --- THIS IS THE FIX ---
# Create a single, global thread pool executor.
# This is simple, robust, and avoids all the FastAPI state/lifespan complexity.
executor = ThreadPoolExecutor(max_workers=5)
# ----------------------

app = FastAPI(
    title="AI Assistant Local Tool Server",
    description="A server providing local search and browse capabilities.",
)

# --- Models (Unchanged) ---
class SearchRequest(BaseModel):
    q: str = Field(..., description="The search query.")
class BrowseRequest(BaseModel):
    urls: list[str] = Field(..., description="A list of URLs to browse and scrape.")
class BrowseResult(BaseModel):
    url: str
    content: str | None = None
    error: str | None = None

# --- Endpoints (Now simplified) ---
@app.post("/search")
async def local_search_for_links(search_request: SearchRequest):
    loop = asyncio.get_running_loop()
    # Use our reliable global executor
    links = await loop.run_in_executor(
        executor, get_search_links, search_request.q
    )
    formatted_results = {"organic": [{"link": link} for link in links]}
    return formatted_results

@app.post("/browse", response_model=list[BrowseResult])
async def perform_browse(browse_request: BrowseRequest):
    loop = asyncio.get_running_loop()
    # Use our reliable global executor
    results = await loop.run_in_executor(
        executor, scrape_all, browse_request.urls
    )
    return results