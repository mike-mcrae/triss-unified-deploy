import pandas as pd
from seleniumbase import SB
from bs4 import BeautifulSoup
import time
import os
import random
from openai import OpenAI
from pathlib import Path

# Path configuration (env-overridable, no machine-specific absolutes)
_PIPELINE_ROOT = Path(
    os.environ.get(
        "TRISS_SOURCE_PIPELINE_DIR",
        str(next((p for p in Path(__file__).resolve().parents if p.name == "pipeline"), Path(__file__).resolve().parents[2]))
    )
).expanduser().resolve()
_PROJECT_ROOT = Path(
    os.environ.get("TRISS_SOURCE_PROJECT_DIR", str(_PIPELINE_ROOT.parent))
).expanduser().resolve()
_SHARED_DATA_ROOT = Path(
    os.environ.get("TRISS_SOURCE_SHARED_DATA_DIR", str(_PROJECT_ROOT / "1. data"))
).expanduser().resolve()
_REPORT_ROOT = Path(
    os.environ.get("TRISS_SOURCE_REPORT_DIR", str(_PROJECT_ROOT / "triss-report-app-v3"))
).expanduser().resolve()

# Configuration
BASE_DIR = str(_PIPELINE_ROOT / "1. data/1. raw/0. profiles/v3")
INPUT_CSV = os.path.join(BASE_DIR, "2e.all_listed_publications_2019_journal_like_no_abstract.csv")
BATCH_SIZE = 5 
OPENAI_MODEL = "gpt-4o-mini"
MAX_BODY_CHARS = 6000 # Capture enough text to ensure abstract is included

# Initialize OpenAI Client
# Assumes OPENAI_API_KEY is in environment variables.
client = OpenAI()

def clean_text(text):
    if not text: return ""
    return text.strip().replace('\n', ' ').replace('\r', ' ')

def get_abstract_from_llm(raw_text):
    """
    Sends raw text to GPT-4o-mini to extract and clean the abstract.
    """
    try:
        if len(raw_text) < 200:
            return None # Not enough text to be useful

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful academic assistant skilled at processing web-scraped text."},
                {"role": "user", "content": f"""
Please identify and extract the **Abstract** from the following raw text which was scraped from an academic publication page.

**Rules:**
1. Return **ONLY** the abstract text. Do not include 'Abstract' as a header.
2. Clean up any scraping artifacts, weird formatting, or merged words.
3. If the text clearly does **not** contain an abstract (e.g. it's just a login page, cookie banner, or reference list), return exactly: `NO_ABSTRACT_FOUND`.
4. Do NOT attempt to write your own abstract if one is not present. Only extract.

**Raw Text:**
{raw_text[:MAX_BODY_CHARS]}
"""}
            ],
            max_tokens=500,
            temperature=0.1 # Low temperature for extraction reliability
        )
        content = response.choices[0].message.content.strip()
        
        if "NO_ABSTRACT_FOUND" in content:
            return None
        return content
        
    except Exception as e:
        print(f"    ! OpenAI Error: {e}")
        return None

def scrape_and_extract(sb, url):
    """Navigate, grab broad text, and use LLM to extract abstract."""
    try:
        print(f"    Navigating: {url}")
        sb.uc_open_with_reconnect(url, reconnect_time=4)
        
        # Handle Cloudflare/CAPTCHA
        try:
             if sb.is_element_visible('iframe[src*="cloudflare"]') or sb.is_text_visible("Just a moment"):
                 sb.uc_gui_click_captcha()
                 time.sleep(3)
        except: 
            pass
            
        # Wait for content
        time.sleep(random.uniform(4, 7))
        
        # Check for block
        if "Just a moment" in sb.get_title():
            print("    ! Still blocked/loading...")
            time.sleep(5)
            
        html = sb.get_page_source()
        soup = BeautifulSoup(html, 'html.parser')
        
        # --- Broad Text Capture ---
        # Prioritize content-heavy areas to avoid header/footer noise if possible,
        # but fall back to body to be safe.
        
        body_text = ""
        
        # 1. Try common article containers first
        article = soup.find('article') or \
                  soup.find('div', class_='article-body') or \
                  soup.find('div', role='main') or \
                  soup.find('main')
                  
        if article:
            body_text = article.get_text(separator=' ', strip=True)
        else:
            # 2. Fallback to full body
            if soup.body:
                body_text = soup.body.get_text(separator=' ', strip=True)
        
        # Clean up whitespace
        body_text = clean_text(body_text)
        
        if len(body_text) > 200:
             print(f"    -> Extracted {len(body_text)} chars of raw text. Sending to LLM...")
             abstract = get_abstract_from_llm(body_text)
             
             if abstract:
                 return abstract, "gpt-4o-mini_extracted"
             else:
                 print("    -> LLM returned NO_ABSTRACT_FOUND.")
        else:
            print("    -> Page text too short/empty.")
        
    except Exception as e:
        print(f"    ! Scrape Error: {e}")
    
    return None, None

def main():
    print(f"Loading {INPUT_CSV}...")
    if not os.path.exists(INPUT_CSV):
        print("File not found.")
        return

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    
    # Init columns if missing
    for col in ['url_found', 'scraped_status', 'abstract', 'abstract_source']:
        if col not in df.columns:
            df[col] = ""
            
    # Force string type to avoid float inference on empty columns
    df['abstract'] = df['abstract'].fillna("").astype(str)
    df['abstract_source'] = df['abstract_source'].fillna("").astype(str)
    df['scraped_status'] = df['scraped_status'].fillna("").astype(str)
    df['url_found'] = df['url_found'].fillna("").astype(str)
    
    # Filter rows to process: Only where abstract is empty or too short
    def needs_abstract(val):
        return len(str(val)) < 20 # Strict empty check

    mask_process = df['abstract'].apply(needs_abstract)
    indices = df[mask_process].index.tolist()
    
    print(f"Found {len(indices)} rows to scrape. Starting SeleniumBase (Headless)...")

    # Safety Check for API Key
    if not os.environ.get("OPENAI_API_KEY"):
         print("ERROR: OPENAI_API_KEY not found in environment variables.")
         return

    with SB(uc=True, test=True, headless=True) as sb:
        save_counter = 0
        
        for idx in indices:
            doi = str(df.loc[idx, 'crossref_doi']).strip()
            
            url = None
            if doi and doi != "nan":
                 url = f"https://doi.org/{doi}"
            
            if url:
                print(f"[{idx}] Scraping DOI: {doi}")
                abstract, source = scrape_and_extract(sb, url)
                
                if abstract:
                    print(f"    -> Found Abstract ({len(abstract)} chars).")
                    df.at[idx, 'abstract'] = abstract
                    df.at[idx, 'abstract_source'] = source
                    df.at[idx, 'scraped_status'] = 'success'
                    df.at[idx, 'url_found'] = sb.get_current_url()
                    df.at[idx, 'abstract_retrieved'] = 1
                    df.at[idx, 'abstract_char_count'] = len(abstract)
                else:
                    df.at[idx, 'scraped_status'] = 'failed'
            
            save_counter += 1
            if save_counter >= BATCH_SIZE:
                print("Saving batch...")
                df.to_csv(INPUT_CSV, index=False)
                save_counter = 0
                
            time.sleep(random.uniform(2, 5))

    print("Final save...")
    df.to_csv(INPUT_CSV, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
