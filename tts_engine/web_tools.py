# web_tools.py
# This module provides functions for initiating and executing web searches.

import logging
import re
import requests
import config

KEYWORD_TRIGGERS = [
    "search the web for", "search online for", "search for",
    "find online", "look up", "web search for", "web search", "can you look up", "please look up",
    "search online", "use web search", "use internet", "use your search"
]

def check_for_search_keyword(user_input: str) -> str | None:
    """
    Checks if the user input starts with a keyword trigger.
    If yes, returns the extracted search query. This is the ONLY way a search can be initiated.
    If no, returns None.
    """
    user_input_lower = user_input.lower()
    for phrase in KEYWORD_TRIGGERS:
        if user_input_lower.startswith(phrase + " "):
            # Extract the actual query part of the user's input
            search_query = user_input[len(phrase):].strip()
            logging.info(f"WEB_TOOLS: Keyword trigger '{phrase}' detected. Proposing search: '{search_query}'")
            return search_query
    return None # No keyword was found

def _call_tool_server(endpoint: str, payload: dict) -> dict | None:
    """A private helper function to call the tool server."""
    url = f"{config.TOOL_SERVER_URL}{endpoint}"
    headers = {"Content-Type": "application/json"}
    try:
        logging.info(f"WEB_TOOLS -> TOOL_SERVER: Calling {url}")
        response = requests.post(url, headers=headers, json=payload, timeout=(10, 60))
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logging.error(f"WEB_TOOLS -> TOOL_SERVER_ERROR: {e}")
        return None

def execute_search_and_summarize(user_input: str, search_query: str, personas: dict) -> list:
    """
    Executes the full search->browse->summarize toolchain.
    This function is only called AFTER the user has confirmed the search.
    It returns a list of messages ready to be sent to the main LLM for the final answer.
    """
    logging.info(f"WEB_TOOLS: Executing confirmed search for: '{search_query}'")

    search_results = _call_tool_server("/search", {"q": search_query})

    if not search_results or 'organic' not in search_results or not search_results['organic']:
        logging.error("WEB_TOOLS: Search tool failed or returned no organic results.")
        return [{"role": "system", "content": personas['default']}, {"role": "user", "content": "Please apologize that your internal web search tool failed to find any relevant information."}]

    urls_to_browse = [item['link'] for item in search_results['organic'][:3]]
    browsed_data = _call_tool_server("/browse", {"urls": urls_to_browse})

    context = f"Based on a web search for '{search_query}', here is the collected information:\n\n"
    if browsed_data:
        for item in browsed_data:
            content = item.get('content', 'No content found.')
            error = item.get('error')
            context += f"--- Content from {item.get('url', 'unknown URL')} ---\n{error or content}\n\n"
    else:
        context += "Unfortunately, the browsing tool failed to retrieve content from the web pages."

    # We use your 'summarizer' persona here to process the raw data.
    user_prompt_for_summarizer = (
        f"USER'S ORIGINAL QUESTION: \"{user_input}\"\n\n"
        f"--- COLLECTED WEB DATA ---\n{context}"
    )
    return [{"role": "system", "content": personas['summarizer']}, {"role": "user", "content": user_prompt_for_summarizer}]