#!/usr/bin/env python3
"""Extract all Featured Article titles from Wikipedia via the categorymembers API.

Writes a sorted, deduplicated JSON array to ``featured_articles.json`` in the
project root.  No ML imports — runs on the Toolforge bastion in seconds.

Usage::

    python3 scripts/extract_featured_articles.py [--output featured_articles.json]
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API_BASE = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "Wiki-TTS-Batch/1.0 (WMF ML Team | LiftWing)"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "featured_articles.json"

BASE_PARAMS = {
    "action": "query",
    "list": "categorymembers",
    "cmtitle": "Category:Featured_articles",
    "cmnamespace": "0",
    "cmlimit": "max",
    "format": "json",
}

MAX_RETRIES = 5
RETRY_DELAY = 5  # seconds


def fetch_page(cmcontinue: str | None = None) -> dict:
    """Fetch one page of categorymembers results."""
    params = dict(BASE_PARAMS)
    if cmcontinue:
        params["cmcontinue"] = cmcontinue

    query_string = urllib.parse.urlencode(params)
    url = f"{API_BASE}?{query_string}"

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def extract_all() -> list[str]:
    """Paginate through all category members and return sorted, deduplicated titles."""
    titles: set[str] = set()
    cmcontinue: str | None = None
    page = 0

    while True:
        page += 1

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                data = fetch_page(cmcontinue)
                break
            except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
                if attempt == MAX_RETRIES:
                    print(
                        f"FATAL: API request failed after {MAX_RETRIES} attempts on "
                        f"page {page}: {exc}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                print(
                    f"  Retry {attempt}/{MAX_RETRIES} after error: {exc}",
                    file=sys.stderr,
                )
                time.sleep(RETRY_DELAY * attempt)

        if "error" in data:
            error_info = data["error"].get("info", str(data["error"]))
            print(f"FATAL: API error on page {page}: {error_info}", file=sys.stderr)
            sys.exit(1)

        members = data.get("query", {}).get("categorymembers", [])
        for member in members:
            titles.add(member["title"])

        print(f"  Page {page}: {len(members)} titles (total unique: {len(titles)})")

        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break

    return sorted(titles)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract Wikipedia Featured Article titles via the categorymembers API"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON file path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    print("Extracting Featured Article titles from Wikipedia categorymembers API...")
    print(f"  Output: {args.output}")
    print()

    titles = extract_all()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: write to temp file then rename
    tmp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(titles, f, indent=2, ensure_ascii=False)
    tmp_path.rename(args.output)

    print(f"\nDone. {len(titles)} unique titles written to {args.output}")
