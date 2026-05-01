import os
import re
import wikipediaapi
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from worker import generate_section_audio

# Initialize FastAPI
app = FastAPI(title="Wikipedia TTS Audio Service | WMF ML Team Prototype")

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

def clean_spoken_text(text: str) -> str:
    """
    Normalizes Wikipedia text for text-to-speech models.
    Removes citations, phonetic guides, and extra whitespace.
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
    # Clean up multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

@app.post("/generate")
def generate_audio(article: str):
    """
    Fetches the article, cleans the text, and queues Celery tasks 
    to generate the audio asynchronously.
    """
    page = wiki.page(article)
    
    if not page.exists():
        raise HTTPException(status_code=404, detail="Article not found")

    sections_queued = 0

    # Queue Lead Section
    lead_text = clean_spoken_text(page.summary)
    if len(lead_text) > 50:
        generate_section_audio.delay(article=page.title, section="Lead", text=lead_text)
        sections_queued += 1

    # Queue remaining sections
    for section in page.sections:
        if section.title.lower() in['see also', 'references', 'external links', 'further reading', 'notes']:
            continue
            
        cleaned_text = clean_spoken_text(section.text)
        if len(cleaned_text) > 50:
            generate_section_audio.delay(article=page.title, section=section.title, text=cleaned_text)
            sections_queued += 1

    return {
        "status": "queued",
        "article": page.title,
        "message": f"Successfully queued {sections_queued} sections for asynchronous audio generation."
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