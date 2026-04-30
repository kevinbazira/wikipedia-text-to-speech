import wikipediaapi
import re

# Initialize the Wikipedia API client (WMF requires a proper User-Agent)
user_agent = "WMF ML Team TTS model-server (LiftWing)"
wiki = wikipediaapi.Wikipedia(user_agent, "en")

def clean_spoken_text(text: str) -> str:
    """
    Normalizes Wikipedia text for text-to-speech models.
    Removes citations, phonetic guides, and extra whitespace.
    """
    if not text:
        return ""
        
    # 1. Remove citation brackets e.g., [1], [23]
    text = re.sub(r'\[\d+\]', '', text)
    
    # 2. Remove "edit" tags if any leaked through
    text = re.sub(r'\[edit\]', '', text, flags=re.IGNORECASE)
    
    # 3. (Optional) Remove basic parenthetical pronunciations often found in lead sections
    # e.g., (English: /fəˈnɛtɪk/) - This is a simple regex, can be expanded later
    text = re.sub(r'\(/.*?/\)', '', text)
    
    # 4. Clean up multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def process_article(page_title: str):
    """
    Fetches an article, splits it by section, cleans the text, 
    and prepares it for the ML queue.
    """
    page = wiki.page(page_title)
    
    if not page.exists():
        print(f"Article '{page_title}' does not exist.")
        return[]

    print(f"\nProcessing Article: {page.title}")
    
    sections_to_queue =[]
    
    # 1. Process the "Lead" (Summary) section first
    lead_text = clean_spoken_text(page.summary)
    if lead_text:
        sections_to_queue.append({
            "article": page.title,
            "section": "Lead",
            "text": lead_text
        })
        print(f"  -> Extracted: Lead ({len(lead_text)} chars)")

    # 2. Process all other top-level sections
    for section in page.sections:
        # Skip sections that shouldn't be read out loud
        if section.title.lower() in ['see also', 'references', 'external links', 'further reading', 'notes']:
            continue
            
        cleaned_text = clean_spoken_text(section.text)
        
        # Only queue sections that actually have text (skipping image-only or empty sections)
        if len(cleaned_text) > 50: 
            sections_to_queue.append({
                "article": page.title,
                "section": section.title,
                "text": cleaned_text
            })
            print(f"  -> Extracted: {section.title} ({len(cleaned_text)} chars)")
            
    return sections_to_queue

if __name__ == "__main__":
    # For the PoC, we will start with a small sample of highly visited articles
    test_articles =[
        "Python_(programming_language)",
        "Earth",
        "Artificial_intelligence"
    ]
    
    all_jobs =[]
    for article in test_articles:
        jobs = process_article(article)
        all_jobs.extend(jobs)
        
    print(f"\nTotal sections prepared for the TTS queue: {len(all_jobs)}")
    
    # In Step 3, we will push 'all_jobs' into Redis/Celery!