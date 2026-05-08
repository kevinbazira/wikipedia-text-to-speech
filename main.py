import os
import re
import wikipediaapi
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
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
    Fetches the article(s), cleans the text, and queues Celery tasks 
    to generate the audio asynchronously.
    """
    if not articles:
        raise HTTPException(status_code=400, detail="Articles parameter cannot be empty.")
        
    # Split the pipe-separated string into a list and strip whitespace
    article_titles =[a.strip() for a in articles.split('|') if a.strip()]
    
    # Remove duplicates in case a user sends "Earth|Earth"
    article_titles = list(set(article_titles))
    
    results =[]
    total_sections_queued = 0

    for article_title in article_titles:
        page = wiki.page(article_title)
        
        # If an article is not found, record the error and continue to the next one
        if not page.exists():
            results.append({
                "article": article_title,
                "status": "error",
                "message": "Article not found"
            })
            continue

        sections_queued = 0
        section_names =[]

        # Queue Lead Section
        lead_text = clean_spoken_text(page.summary)
        if len(lead_text) > 50:
            generate_section_audio.delay(article=page.title, section="Lead", text=lead_text)
            sections_queued += 1
            section_names.append("Lead")

        # Queue remaining sections
        for section in page.sections:
            if section.title.lower() in['see also', 'references', 'external links', 'further reading', 'notes']:
                continue
                
            cleaned_text = clean_spoken_text(section.text)
            if len(cleaned_text) > 50:
                generate_section_audio.delay(article=page.title, section=section.title, text=cleaned_text)
                sections_queued += 1
                section_names.append(section.title)

        results.append({
            "article": page.title,
            "status": "queued",
            "sections_queued": sections_queued,
            "section_names": section_names
        })
        total_sections_queued += sections_queued

    return {
        "message": f"Processing complete. Queued {total_sections_queued} total sections.",
        "details": results
    }

@app.get("/audio")
def get_audio(article: str, section: str):
    """
    Checks if the audio file exists and serves it to the client.
    """
    # Sanitize inputs exactly as the Celery worker does
    safe_article = article.replace(" ", "_").replace("/", "-")
    safe_section = section.replace(" ", "_").replace("/", "-")
    
    file_path = f"./audio_output/{safe_article}/{safe_section}.mp3"
    
    if not os.path.exists(file_path):
        # Return a 404 (or 202) so the UI knows it's still processing
        raise HTTPException(
            status_code=404, 
            detail=f"Audio not found or still processing for {article} - {section}."
        )

    # FileResponse automatically handles streaming/byte-range requests!
    return FileResponse(
        path=file_path, 
        media_type="audio/mpeg", 
        filename=f"{safe_section}.mp3"
    )