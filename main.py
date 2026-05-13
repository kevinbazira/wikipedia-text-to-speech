import os
import re
import wikipediaapi
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from worker import generate_section_audio

# Initialize FastAPI
app = FastAPI(title="Wikipedia TTS Prototype | WMF ML Team")

# Allow frontend clients to call this API without CORS errors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Wikipedia API
user_agent = "WMF ML Team TTS model-server (LiftWing)"
wiki = wikipediaapi.Wikipedia(user_agent, "en")

# ── Number-to-words helpers ──────────────────────────────────────────────────

_WORDS = (
    "zero one two three four five six seven eight nine ten "
    "eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen"
).split()

_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()

_SCALES = ["", "thousand", "million", "billion"]


def _int_to_words(n: int) -> str:
    """Convert an integer (0 – 999 999 999 999) to English words."""
    if n == 0:
        return "zero"

    def _hundreds(n: int) -> str:
        """Handle 0 – 999."""
        if n == 0:
            return ""
        parts = []
        if n >= 100:
            parts.append(_WORDS[n // 100] + " hundred")
            n %= 100
        if n >= 20:
            t, o = divmod(n, 10)
            chunk = _TENS[t - 2]
            if o:
                chunk += "-" + _WORDS[o]
            parts.append(chunk)
        elif n > 0:
            parts.append(_WORDS[n])
        return " ".join(parts)

    result = []
    scale_idx = 0
    while n > 0:
        chunk = n % 1000
        if chunk:
            label = _hundreds(chunk)
            if scale := _SCALES[scale_idx]:
                label += " " + scale
            result.append(label)
        n //= 1000
        scale_idx += 1
    return " ".join(reversed(result))


def _norm_numbers(text: str) -> str:
    """
    Convert numeric tokens to their spoken form so the TTS model's G2P
    doesn't have to guess how to pronounce them.

    "70.8%"  → "seventy point eight percent"
    "3.14"   → "three point one four"
    "50%"    → "fifty percent"
    "42"     → "forty-two"
    """
    def _replace_decimal(m: re.Match) -> str:
        integer_word = _int_to_words(int(m.group(1)))
        decimal_digits = " ".join(_WORDS[int(d)] for d in m.group(2))
        suffix = " percent" if m.group(3) else ""
        return f"{integer_word} point {decimal_digits}{suffix}"

    def _replace_int_percent(m: re.Match) -> str:
        return f"{_int_to_words(int(m.group(1)))} percent"

    def _replace_int(m: re.Match) -> str:
        return _int_to_words(int(m.group(0)))

    # Order: decimal + optional percent first, then integer + percent, then bare integer
    text = re.sub(r'(\d+)\.(\d+)(%)?', _replace_decimal, text)
    text = re.sub(r'(?<!\d)(\d+)%', _replace_int_percent, text)
    # Convert bare integers that are standalone tokens (surrounded by word boundaries)
    text = re.sub(r'(?<!\d)(\d+)(?!\.\d)', _replace_int, text)

    return text


# ── Main text cleaner ────────────────────────────────────────────────────────

def clean_spoken_text(text: str) -> str:
    """
    Normalizes Wikipedia text for text-to-speech models.
    Removes citations, phonetic guides, normalizes numbers, extra whitespace.
    """
    if not text:
        return ""

    # Remove citation brackets e.g [1], [23]
    text = re.sub(r'\[\d+\]', '', text)
    # Remove "edit" tags if any leaked through
    text = re.sub(r'\[edit\]', '', text, flags=re.IGNORECASE)
    # Remove basic parenthetical pronunciations often found in lead sections
    # e.g (English: /fəˈnɛtɪk/) - This is a simple regex, can be expanded later
    text = re.sub(r'\(/.*?/\)', '', text)
    # Normalize numbers to spoken form
    text = _norm_numbers(text)
    # Clean up multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# ── Core Logic Helpers ────────────────────────────────────────────────────────

def _queue_missing_sections(article_title: str) -> dict:
    """
    For /generate (Bulk queueing):
    Fetches the article, checks local storage, and ONLY queues Celery tasks
    for sections that do not already exist on disk. Makes the endpoint idempotent!
    """
    page = wiki.page(article_title)
    if not page.exists():
        return {"article": article_title, "status": "error", "message": "Article not found"}

    safe_article = page.title.replace(" ", "_").replace("/", "-")

    def audio_exists(sec_title: str) -> bool:
        safe_sec = sec_title.replace(" ", "_").replace("/", "-")
        return os.path.exists(f"./audio_output/{safe_article}/{safe_sec}.mp3")

    sections_queued = 0
    section_names = []

    # Check Lead
    if not audio_exists("Lead"):
        lead_text = clean_spoken_text(page.summary)
        if len(lead_text) > 50:
            generate_section_audio.delay(article=page.title, section="Lead", text=lead_text)
            sections_queued += 1
            section_names.append("Lead")

    # Check remaining sections
    for section in page.sections:
        if section.title.lower() in ['see also', 'references', 'external links', 'further reading', 'notes']:
            continue

        if not audio_exists(section.title):
            cleaned_text = clean_spoken_text(section.text)
            if len(cleaned_text) > 50:
                generate_section_audio.delay(article=page.title, section=section.title, text=cleaned_text)
                sections_queued += 1
                section_names.append(section.title)

    return {
        "article": page.title,
        "status": "queued" if sections_queued > 0 else "already_exists",
        "sections_queued": sections_queued,
        "section_names": section_names
    }

def _fetch_single_section_text(article_title: str, section_title: str) -> tuple[str | None, str | None]:
    """
    For /audio (Laser-focused fetch):
    Fetches and cleans text for a single specific article section.
    Returns (cleaned_text, None) on success, or (None, error_message) on failure.
    """
    page = wiki.page(article_title)
    if not page.exists():
        return None, f"Article '{article_title}' not found"

    if section_title == "Lead":
        text = page.summary
    else:
        section = next((s for s in page.sections if s.title == section_title), None)
        if section is None:
            return None, f"Section '{section_title}' not found"
        text = section.text

    cleaned = clean_spoken_text(text)
    if len(cleaned) <= 50:
        return None, "Section text too short for TTS generation"

    return cleaned, None


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """
    Serves the index.html front-end.
    Will likely not be required in the LiftWing production setup as Wikipedia will be the front-end.
    It's convenient for this prototype to keep it all in one place that's easy to demo.
    """
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/generate")
def generate_audio(articles: str):
    """
    Accepts a single article (e.g "Earth") or a pipe-separated list of articles (e.g "Earth|Mars")
    just like the MediaWiki action API: https://en.wikipedia.org/w/api.php?action=query&prop=info&titles=Earth|Mars
    Idempotent endpoint: Fetches the article(s), checks local storage, and ONLY queues
    Celery tasks for sections that do not already exist on disk.
    """
    if not articles:
        raise HTTPException(status_code=400, detail="Articles parameter cannot be empty.")

    article_titles = [a.strip() for a in articles.split('|') if a.strip()]
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
        "details": results
    }

@app.get("/audio")
def get_audio(article: str, section: str):
    """
    Checks if the audio file exists and serves it to the client.
    If it doesn't exist, fetches the article text, queues generation, and returns HTTP 202.
    """
    # Sanitize inputs exactly as the Celery worker does
    safe_article = article.replace(" ", "_").replace("/", "-")
    safe_section = section.replace(" ", "_").replace("/", "-")

    file_path = f"./audio_output/{safe_article}/{safe_section}.mp3"

    if os.path.exists(file_path):
        # FileResponse automatically handles streaming/byte-range requests!
        return FileResponse(
            path=file_path,
            media_type="audio/mpeg",
            filename=f"{safe_section}.mp3"
        )

    # File doesn't exist — fetch text and queue async generation
    text, error = _fetch_single_section_text(article, section)
    if text is None:
        raise HTTPException(status_code=404, detail=error)

    generate_section_audio.delay(article=article, section=section, text=text)

    return JSONResponse(
        status_code=202,
        content={
            "article": article,
            "section": section,
            "status": "queued",
            "message": f"Audio generation queued for {article} - {section}."
        }
    )