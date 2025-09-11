#!/usr/bin/env python3
"""
watchdog_csv_uploader.py

Watches a “drop” folder for new CSVs and triggers the
Airflow DAG via your /airflow/trigger_dag endpoint as soon
as a new file appears.
"""

import time
import os
import requests
import logging
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import constants

# ─── Logging configuration ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────────

# Folder where you drop raw CSVs for ingestion
WATCHED_DIR = constants.COMPARISON_FOLDER_PATH

# Flask endpoint that triggers the Airflow DAG
TRIGGER_API_URL = f"{constants.REST_API_BASE_URL}airflow/trigger_dag"

# The DAG ID you want to trigger
DAG_ID = "monitor_csv_folder_for_training"

# ─── Event Handler ────────────────────────────────────────────────────────────────

class CsvUploaderHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self._processed = set()
        logger.info(f"Handler initialized, watching: {WATCHED_DIR}")

    def on_created(self, event):
        # Only handle files, not directories
        if event.is_directory:
            return

        filepath = Path(event.src_path)
        # Only CSVs
        if filepath.suffix.lower() != ".csv":
            logger.debug(f"Ignoring non-CSV file: {filepath.name}")
            return

        fullpath = str(filepath.resolve())
        # Skip if already processed
        if fullpath in self._processed:
            logger.debug(f"Already processed, skipping: {filepath.name}")
            return

        # Wait to ensure write is complete
        time.sleep(1)

        logger.info(f"Detected new CSV: {fullpath}")

        # Trigger the Airflow DAG with the filename
        try:
            payload = {"dag_id": DAG_ID, "filename": filepath.name}
            logger.info(f"Sending trigger to {TRIGGER_API_URL} with payload {payload}")
            resp = requests.post(TRIGGER_API_URL, json=payload)
            resp.raise_for_status()
            logger.info(f"Triggered DAG successfully, response: {resp.json()}")
            # Mark as processed only after successful trigger
            self._processed.add(fullpath)
        except Exception as e:
            logger.error(f"ERROR triggering DAG for {filepath.name}: {e}", exc_info=True)

# ─── Main ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting watchdog_csv_uploader")
    os.makedirs(WATCHED_DIR, exist_ok=True)
    if not os.path.isdir(WATCHED_DIR):
        logger.error(f"Watched dir does not exist: {WATCHED_DIR}")
        exit(1)

    handler = CsvUploaderHandler()
    observer = Observer()
    observer.schedule(handler, WATCHED_DIR, recursive=False)

    logger.info(f"Watching directory: {WATCHED_DIR} -> TRIGGER at {TRIGGER_API_URL}")
    observer.start()

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("Interrupted by user, stopping watchdog...")
        observer.stop()
    observer.join()
    logger.info("Watchdog stopped.")
