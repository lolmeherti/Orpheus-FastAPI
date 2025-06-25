# search_logger.py (Complete Final Version)

import json
import logging
from datetime import datetime, timezone
import os

LOG_FILE = "search_history.jsonl"

def log_search(original_question: str, executed_query: str, summary_answer: str):
    try:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "original_question": original_question,
            "executed_query": executed_query,
            "summary_answer": summary_answer,
        }
        log_line = json.dumps(log_entry)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")
        logging.info(f"SEARCH_LOGGER: Successfully logged search for query: '{executed_query}'")
    except Exception as e:
        logging.error(f"SEARCH_LOGGER: Failed to log search event. Error: {e}", exc_info=True)

def read_recent_searches(count: int = 5) -> list[dict]:
    if not os.path.exists(LOG_FILE):
        logging.warning("SEARCH_LOGGER: read_recent_searches called, but log file does not exist.")
        return []
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            recent_lines = f.readlines()[-count:]
        recent_searches = []
        for line in reversed(recent_lines):
            try:
                recent_searches.append(json.loads(line))
            except json.JSONDecodeError:
                logging.warning(f"SEARCH_LOGGER: Skipping malformed JSON line: {line.strip()}")
        logging.info(f"SEARCH_LOGGER: Successfully read {len(recent_searches)} search events from log.")
        return recent_searches
    except Exception as e:
        logging.error(f"SEARCH_LOGGER: Failed to read search log file. Error: {e}", exc_info=True)
        return []