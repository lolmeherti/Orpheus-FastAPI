# Save this code as main.py
import os
import httpx
import asyncio
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

# --- 1. Configuration ---
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPER_API_URL = "https://google.serper.dev/search"

# --- 2. Create the FastAPI App ---
app = FastAPI(
    title="AI Assistant Backend",
    description="A secure server providing search and browse capabilities.",
)

# --- 3. Define Data Models ---
class SerperSearchRequest(BaseModel):
    q: str = Field(..., description="The search query.")

class BrowseRequest(BaseModel):
    urls: list[str] = Field(..., description="A list of URLs to browse and scrape.")

class BrowseResult(BaseModel):
    url: str
    content: str | None = None
    error: str | None = None


# --- 4. The /search Endpoint (Unchanged) ---
@app.post("/search")
async def perform_search(search_request: SerperSearchRequest):
    if not SERPER_API_KEY:
        raise HTTPException(status_code=500, detail="SERPER_API_KEY is not configured.")
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = search_request.model_dump()
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(SERPER_API_URL, headers=headers, json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Error from Serper API: {e.response.text}")
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Could not connect to Serper API: {e}")
    return response.json()


# --- 5. The NEW /browse Endpoint ---
async def fetch_and_parse_url(client: httpx.AsyncClient, url: str) -> BrowseResult:
    """Fetches a single URL, parses it, and returns a structured result."""
    try:
        # Some websites block default user agents, so we pretend to be a browser.
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = await client.get(url, headers=headers, follow_redirects=True, timeout=15.0)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        # Use BeautifulSoup to parse the HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # Clean up the text: remove script/style tags, extra whitespace
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose() # Remove these tags from the soup

        text = soup.get_text(separator=' ', strip=True)
        # Optional: A more aggressive cleaning with regex can be useful
        # text = re.sub(r'\s+', ' ', text).strip()

        return BrowseResult(url=url, content=text)

    except httpx.HTTPStatusError as e:
        return BrowseResult(url=url, error=f"HTTP error {e.response.status_code} while fetching page.")
    except Exception as e:
        return BrowseResult(url=url, error=f"An error occurred: {str(e)}")


@app.post("/browse", response_model=list[BrowseResult])
async def perform_browse(browse_request: BrowseRequest):
    """
    Accepts a list of URLs, scrapes their content, and returns the cleaned text.
    """
    async with httpx.AsyncClient() as client:
        # Create a list of tasks to run concurrently
        tasks = [fetch_and_parse_url(client, url) for url in browse_request.urls]
        # Run all browsing tasks in parallel and wait for them to complete
        results = await asyncio.gather(*tasks)

    return results