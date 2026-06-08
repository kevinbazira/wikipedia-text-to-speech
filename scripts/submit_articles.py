#!/usr/bin/env python3
"""Submit Featured Articles to the TTS /generate endpoint with dynamic throttling.

Polls Redis queue depth before each batch submission.  If the queue exceeds
the configured threshold, the script backs off until it drops.  This
auto-regulates throughput based on actual worker capacity.

Usage::

    python3 scripts/submit_articles.py [--dry-run] [--api-url URL] [--input FILE]

Environment variables::

    TTS_API_URL   Override the default API base URL.
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Allow project imports regardless of CWD
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import redis

from wiki_tts.config import CELERY_TASK_DEFAULT_QUEUE, REDIS_URL

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "featured_articles.json"
CHECKPOINT_FILE = PROJECT_ROOT / "submission_progress.json"
FAILED_FILE = PROJECT_ROOT / "failed_articles.json"

# ── Constants ──────────────────────────────────────────────────────────────────

if os.path.exists("/data/project/wiki-tts"):
    DEFAULT_API_URL = "https://wiki-tts.toolforge.org"
else:
    DEFAULT_API_URL = "http://localhost:8000"

BATCH_SIZE = 25
QUEUE_THRESHOLD = 200
COOLDOWN_SECONDS = 60
POLL_INTERVAL = 5  # seconds between queue-depth checks during cooldown
MAX_RETRIES = 3
RETRY_BASE_DELAY = 5  # seconds, multiplied by attempt number

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("submit_articles")

# ── Shutdown flag (set by signal handlers) ─────────────────────────────────────

_shutdown_requested = False


def handle_shutdown(signum: int, frame: object) -> None:
    global _shutdown_requested
    sig_name = signal.Signals(signum).name
    logger.info("Received %s — shutting down after current batch...", sig_name)
    _shutdown_requested = True


signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)


# ── Helpers ────────────────────────────────────────────────────────────────────


def load_checkpoint() -> int:
    """Return the last *successfully* submitted index, or -1 if no checkpoint."""
    if not CHECKPOINT_FILE.exists():
        return -1
    try:
        data = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
        last_index = int(data.get("last_index", -1))
        logger.info("Resuming from checkpoint at index %d (%d already processed)", last_index, last_index + 1)
        return last_index
    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        logger.warning("Could not parse checkpoint file (%s), starting from beginning", exc)
        return -1


def save_checkpoint(last_index: int) -> None:
    """Atomically write the checkpoint file."""
    tmp_path = CHECKPOINT_FILE.with_suffix(CHECKPOINT_FILE.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps({"last_index": last_index, "articles_processed": last_index + 1}),
        encoding="utf-8",
    )
    tmp_path.rename(CHECKPOINT_FILE)


def append_failed(titles: list[str]) -> None:
    """Append failed article titles to the failed-articles file as a JSON array."""
    existing: list[str] = []
    if FAILED_FILE.exists():
        try:
            existing = json.loads(FAILED_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            existing = []

    seen = set(existing)
    new_entries = [t for t in titles if t not in seen]
    if not new_entries:
        return

    existing.extend(new_entries)
    tmp_path = FAILED_FILE.with_suffix(FAILED_FILE.suffix + ".tmp")
    tmp_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.rename(FAILED_FILE)
    logger.info("Appended %d titles to %s", len(new_entries), FAILED_FILE)


def read_queue_depth(redis_client: redis.Redis) -> int:
    """Return the current length of the Celery task queue."""
    return redis_client.llen(CELERY_TASK_DEFAULT_QUEUE)


def wait_for_queue(redis_client: redis.Redis) -> None:
    """Block until the queue depth falls below the threshold or shutdown is signalled."""
    while not _shutdown_requested:
        depth = read_queue_depth(redis_client)
        if depth <= QUEUE_THRESHOLD:
            return
        logger.info(
            "Queue depth %d > %d — backing off for %ds...",
            depth,
            QUEUE_THRESHOLD,
            COOLDOWN_SECONDS,
        )
        # Sleep in short intervals so we can respond to shutdown signals
        slept = 0
        while slept < COOLDOWN_SECONDS and not _shutdown_requested:
            time.sleep(min(POLL_INTERVAL, COOLDOWN_SECONDS - slept))
            slept += POLL_INTERVAL


def submit_batch(
    api_url: str,
    titles: list[str],
) -> tuple[bool, int]:
    """POST a pipe-separated batch of article titles to the /generate endpoint.

    Returns ``(success, sections_queued)``.
    """
    articles = "|".join(titles)
    url = f"{api_url}?articles={urllib.parse.quote(articles, safe='')}"

    req = urllib.request.Request(  # noqa: S310
        url,
        data=b"",
        headers={
            "User-Agent": "Wiki-TTS-Batch/1.0 (WMF ML Team; Toolforge)",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
        body = json.loads(resp.read().decode())
        # Count sections queued across all articles in the response
        sections = 0
        for detail in body.get("details", []):
            sections += detail.get("sections_queued", 0)
        return True, sections


def submit_batch_with_retry(api_url: str, titles: list[str]) -> tuple[bool, int]:
    """Submit a batch with retries on transient errors.

    Returns ``(success, sections_queued)``.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            ok, queued = submit_batch(api_url, titles)
            return ok, queued
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            if attempt == MAX_RETRIES:
                logger.error(
                    "Batch failed after %d attempts: %s — articles: %s",
                    MAX_RETRIES,
                    exc,
                    ", ".join(titles[:5]) + ("..." if len(titles) > 5 else ""),
                )
                return False, 0
            delay = RETRY_BASE_DELAY * attempt
            logger.warning("Attempt %d/%d failed (%s), retrying in %ds...", attempt, MAX_RETRIES, exc, delay)
            time.sleep(delay)

    return False, 0


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit Featured Articles to the TTS /generate endpoint with dynamic throttling"
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get("TTS_API_URL", DEFAULT_API_URL),
        help=f"Base URL of the TTS API (default: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"JSON file with article titles (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Articles per batch (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--queue-threshold",
        type=int,
        default=QUEUE_THRESHOLD,
        help=f"Max queue depth before backoff (default: {QUEUE_THRESHOLD})",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=COOLDOWN_SECONDS,
        help=f"Seconds to wait when queue is above threshold (default: {COOLDOWN_SECONDS})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate Redis connectivity and print plan without queueing any tasks",
    )
    args = parser.parse_args()

    api_url = args.api_url.rstrip("/") + "/generate"

    # ── Load articles ──────────────────────────────────────────────────────
    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        logger.info("Run 'scripts/extract_featured_articles.py' first to create it.")
        sys.exit(1)

    all_titles = json.loads(args.input.read_text(encoding="utf-8"))
    logger.info("Loaded %d articles from %s", len(all_titles), args.input)

    # ── Resume from checkpoint ──────────────────────────────────────────────
    last_index = load_checkpoint()
    start_index = last_index + 1

    if start_index >= len(all_titles):
        logger.info("All %d articles already processed. Nothing to do.", len(all_titles))
        return

    pending_titles = all_titles[start_index:]
    logger.info(
        "%d articles remaining to submit (starting at index %d)",
        len(pending_titles),
        start_index,
    )

    # ── Redis connection ────────────────────────────────────────────────────
    logger.info("Connecting to Redis at %s...", REDIS_URL)
    try:
        redis_client = redis.from_url(REDIS_URL, socket_timeout=10)
        redis_client.ping()
        logger.info("Redis connection OK")
    except redis.RedisError as exc:
        logger.error("Cannot connect to Redis: %s", exc)
        sys.exit(1)

    if args.dry_run:
        depth = read_queue_depth(redis_client)
        logger.info("Dry-run mode — queue depth is %d, would submit to %s", depth, api_url)
        logger.info(
            "Would process %d articles in %d batches of %d",
            len(pending_titles),
            (len(pending_titles) + args.batch_size - 1) // args.batch_size,
            args.batch_size,
        )
        return

    # ── Main submission loop ────────────────────────────────────────────────
    total_queued = 0
    total_failed: list[str] = []
    current_idx = start_index
    total_batches = (len(pending_titles) + args.batch_size - 1) // args.batch_size
    batch_num = 0

    while current_idx < len(all_titles) and not _shutdown_requested:
        batch_titles = all_titles[current_idx : current_idx + args.batch_size]
        batch_num += 1

        # Throttle based on queue depth
        wait_for_queue(redis_client)

        if _shutdown_requested:
            break

        logger.info(
            "Batch %d/%d: submitting %d articles [%d–%d]...",
            batch_num,
            total_batches,
            len(batch_titles),
            current_idx,
            current_idx + len(batch_titles) - 1,
        )

        ok, sections = submit_batch_with_retry(api_url, batch_titles)

        if ok:
            total_queued += sections
            current_idx += len(batch_titles)
            save_checkpoint(current_idx - 1)
            logger.info(
                "  OK — %d sections queued (total: %d, progress: %d/%d)",
                sections,
                total_queued,
                current_idx,
                len(all_titles),
            )
        else:
            append_failed(batch_titles)
            total_failed.extend(batch_titles)
            current_idx += len(batch_titles)
            save_checkpoint(current_idx - 1)
            logger.warning(
                "  FAILED — %d articles written to %s, continuing...",
                len(batch_titles),
                FAILED_FILE,
            )

    if _shutdown_requested:
        save_checkpoint(current_idx - 1)
        logger.info(
            "Graceful shutdown at index %d. Progress saved; re-run to resume.",
            current_idx,
        )

    logger.info(
        "Done. %d articles submitted, %d sections queued, %d failed.",
        current_idx,
        total_queued,
        len(total_failed),
    )

    if total_failed:
        logger.info("Failed articles written to %s — re-run with that file to retry.", FAILED_FILE)


if __name__ == "__main__":
    main()
