# search_logger.py
# This module handles the persistent logging of search events.

import json
import logging
from datetime import datetime, timezone

# The name of the log file. It will be created in the same directory.
LOG_FILE = "search_history.jsonl"

def log_search(original_question: str, executed_query: str, summary_answer: str):
    """
    Appends a new search event to the search history log file.

    Each log entry is a single line containing a JSON object.
    This is the "JSON Lines" format (.jsonl).
    """
    try:
        # Create a dictionary to hold the structured log data.
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "original_question": original_question,
            "executed_query": executed_query,
            "summary_answer": summary_answer,
        }

        # Convert the dictionary to a JSON string.
        log_line = json.dumps(log_entry)

        # Open the log file in "append" mode ('a').
        # The 'with' statement ensures the file is properly closed.
        # 'encoding="utf-8"' is crucial for handling all text characters.
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_line + "\n") # Write the JSON string followed by a newline.

        logging.info(f"SEARCH_LOGGER: Successfully logged search for query: '{executed_query}'")

    except Exception as e:
        # Log any errors that occur during the file writing process.
        logging.error(f"SEARCH_LOGGER: Failed to log search event. Error: {e}", exc_info=True)