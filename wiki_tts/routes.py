import os

import wikipediaapi
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from wiki_tts.config import AUDIO_OUTPUT_DIR, MIN_TEXT_LENGTH
from wiki_tts.locking import acquire_lock, release_lock
from wiki_tts.text import clean_spoken_text
from wiki_tts.wikipedia_utils import find_section_by_title, get_valid_sections
from wiki_tts.worker import generate_section_audio

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Wikipedia TTS Prototype | WMF ML Team")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Wikipedia API client ──────────────────────────────────────────────────────
user_agent = "WMF ML Team TTS model-server (LiftWing)"
wiki = wikipediaapi.Wikipedia(user_agent, "en")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _audio_path(article: str, section: str) -> str:
    safe_article = article.replace(" ", "_").replace("/", "-")
    safe_section = section.replace(" ", "_").replace("/", "-")
    return f"{AUDIO_OUTPUT_DIR}/{safe_article}/{safe_section}.mp3"


def _audio_exists(article: str, section: str) -> bool:
    return os.path.exists(_audio_path(article, section))


def _queue_missing_sections(article_title: str) -> dict:
    """
    Fetches the article, checks local storage + Redis locks, and only queues
    Celery tasks for sections that are missing on disk and not already locked.
    """
    page = wiki.page(article_title)
    if not page.exists():
        return {"article": article_title, "status": "error", "message": "Article not found"}

    safe_article = page.title.replace(" ", "_").replace("/", "-")
    sections_queued = 0
    section_names = []
    all_sections = []

    def try_queue_task(sec_title: str, raw_text: str) -> None:
        nonlocal sections_queued
        all_sections.append({"name": sec_title, "status": "queued"})
        section_names.append(sec_title)

        safe_sec = sec_title.replace(" ", "_").replace("/", "-")
        if acquire_lock(safe_article, safe_sec):
            cleaned = clean_spoken_text(raw_text)
            if len(cleaned) > MIN_TEXT_LENGTH:
                generate_section_audio.delay(article=page.title, section=sec_title, text=raw_text)
                sections_queued += 1
            else:
                release_lock(safe_article, safe_sec)

    # Lead section
    if _audio_exists(page.title, "Lead"):
        all_sections.append({"name": "Lead", "status": "exists"})
    else:
        try_queue_task("Lead", page.summary)

    # Remaining sections (recursive: h2, h3, h4...)
    for section in get_valid_sections(page.sections):
        if _audio_exists(page.title, section.title):
            all_sections.append({"name": section.title, "status": "exists"})
        else:
            try_queue_task(section.title, section.text)

    return {
        "article": page.title,
        "status": "queued" if sections_queued > 0 else "already_exists",
        "sections_queued": sections_queued,
        "section_names": section_names,
        "sections": all_sections,
    }


def _fetch_single_section_text(article_title: str, section_title: str) -> tuple[str | None, str | None]:
    """Fetch and clean text for a single section. Returns (text, None) or (None, error)."""
    page = wiki.page(article_title)
    if not page.exists():
        return None, f"Article '{article_title}' not found"

    if section_title == "Lead":
        text = page.summary
    else:
        section = find_section_by_title(page.sections, section_title)
        if section is None:
            return None, f"Section '{section_title}' not found"
        text = section.text

    cleaned = clean_spoken_text(text)
    if len(cleaned) <= MIN_TEXT_LENGTH:
        return None, "Section text too short for TTS generation"

    return text, None  # Return raw — worker's NeMo normalizes from scratch


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serve the frontend."""
    import wiki_tts

    pkg_dir = os.path.dirname(wiki_tts.__file__)
    index_path = os.path.join(pkg_dir, "static", "index.html")
    with open(index_path, encoding="utf-8") as f:
        return f.read()


@app.post("/generate")
def generate_audio(articles: str):
    """
    Accepts a single article (e.g "Earth") or a pipe-separated list (e.g "Earth|Mars").
    Idempotent: only queues sections missing on disk.
    """
    if not articles:
        raise HTTPException(status_code=400, detail="Articles parameter cannot be empty.")

    article_titles = [a.strip() for a in articles.split("|") if a.strip()]
    article_titles = list(set(article_titles))

    results = []
    total_sections_queued = 0

    for article_title in article_titles:
        res = _queue_missing_sections(article_title)
        results.append(res)
        if res.get("status") == "queued":
            total_sections_queued += res.get("sections_queued", 0)

    return {
        "message": f"Processing complete. Queued {total_sections_queued} new sections.",
        "details": results,
    }


@app.get("/audio")
def get_audio_dynamic(article: str, section: str):
    """
    If the MP3 exists, serve it directly (HTTP 200).
    Otherwise, atomically check the Redis lock and queue generation (HTTP 202).
    Returns 404 if the article/section doesn't exist.
    """
    safe_article = article.replace(" ", "_").replace("/", "-")
    safe_section = section.replace(" ", "_").replace("/", "-")
    file_path = _audio_path(safe_article, safe_section)

    if os.path.exists(file_path):
        return FileResponse(
            path=file_path,
            media_type="audio/mpeg",
            filename=f"{safe_section}.mp3",
        )

    # Atomically check Redis lock before queueing
    if acquire_lock(safe_article, safe_section):
        text, error = _fetch_single_section_text(article, section)
        if text is None:
            release_lock(safe_article, safe_section)
            raise HTTPException(status_code=404, detail=error)
        generate_section_audio.delay(article=article, section=section, text=text)

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "article": article,
            "section": section,
            "status": "processing",
            "message": f"Audio for '{article}' - '{section}' is generating.",
        },
    )


@app.get("/audio/{article:path}/{section}.mp3")
def get_audio_static(article: str, section: str):
    """
    Pure static URL for native OS audio players (iOS AVPlayer, Android ExoPlayer, etc).
    Returns HTTP 200 (MP3 byte-stream) if the file exists, or HTTP 404 if it doesn't.
    NB: Does not trigger generation.
    """
    file_path = _audio_path(article, section)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found.")

    return FileResponse(
        path=file_path,
        media_type="audio/mpeg",
        filename=f"{section}.mp3",
    )
