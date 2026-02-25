import pandas as pd
import os
import time
import random
from seleniumbase import SB
from openai import OpenAI
from bs4 import BeautifulSoup
import traceback
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

# Paths
BASE_DIR = str(_PIPELINE_ROOT / "1. data/1. raw/0. profiles/v3")
TARGET_CSV = os.path.join(BASE_DIR, "2e.all_listed_publications_2019_books_other_abstracts.csv")

OPENAI_MODEL = "gpt-4o-mini"

def clean_html(html_content):
    """
    Reduces HTML to mostly text and structural tags to save tokens.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Remove scripts, styles, etc.
    for element in soup(["script", "style", "noscript", "svg", "path", "footer", "nav"]):
        element.decompose()
        
    # Get text with some structure
    # text = soup.get_text(separator=" ", strip=True)
    # Using simple text might lose too much context (headers vs body)
    # Let's keep it simple for now -> get body text
    text = soup.get_text(separator="\n", strip=True)
    return text[:15000] # Limit to ~15k chars to fit in context easily

def get_description_from_llm(client, title, page_text):
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a research assistant helping to catalog books and chapters."},
                {"role": "user", "content": f"""
I visited the URL for this publication: "**{title}**".
Here is the text content extracted from the page:

--- START PAGE CONTENT ---
{page_text}
--- END PAGE CONTENT ---

**Task**:
1. Identify if this page contains a description, summary, or abstract of the specific book or chapter named "**{title}**".
2. If found, extract it and rewrite it as a clear, standalone abstract-length description (150-300 words).
3. If the page is a login wall, captcha, or unrelated (e.g. generic publisher home page), return `NO_CONTENT`.

**Output**:
Return ONLY the description text or `NO_CONTENT`.
"""}
            ],
            temperature=0.0,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM Error: {e}")
        return None

def main():
    print(f"Loading {os.path.basename(TARGET_CSV)}...")
    try:
        df = pd.read_csv(TARGET_CSV, low_memory=False, encoding='utf-8')
    except:
        df = pd.read_csv(TARGET_CSV, low_memory=False, encoding='latin1')
        
    if 'abstract' not in df.columns:
        df['abstract'] = ''
    if 'abstract_source' not in df.columns:
        df['abstract_source'] = ''

    # Filter for work
    # conditions: abstract is empty AND main_url is present
    # also valid main_url
    
    # We will iterate row by row to save progress
    
    client = OpenAI()
    
    # Start SB
    # Using context manager for the whole loop might be unstable if it runs for hours,
    # but for ~600 rows (subset of 1128 missing) it might correspond to 30 mins.
    # Reconnecting driver for each might be safer but slower. 
    # Let's try one driver session.
    
    processed_count = 0
    
    print("Starting SeleniumBase...")
    with SB(uc=True, test=True, headless=False) as sb: # Headless=False often better for bypassing
        for idx, row in df.iterrows():
            current_abstract = str(row.get('abstract', ''))
            main_url = str(row.get('main_url', ''))
            title = str(row.get('Title', ''))
            
            # Skip if already has abstract or marked as no content
            if current_abstract and (len(current_abstract) > 50 or current_abstract == 'NO_CONTENT'):
                continue
                
            # Skip if no URL
            if not main_url or main_url.lower() in ['nan', '-', '']:
                continue
                
            print(f"[{idx}] Probing: {title[:40]}... -> {main_url}")
            
            try:
                # Open URL
                sb.uc_open_with_reconnect(main_url, reconnect_time=4)
                
                # Check for simplistic bot detection/captcha presence?
                time.sleep(3 + random.random() * 2) 
                
                # Get Source
                page_source = sb.get_page_source()
                
                # Clean
                clean_text = clean_html(page_source)
                
                # LLM
                desc = get_description_from_llm(client, title, clean_text)
                
                if desc and desc != "NO_CONTENT" and "NO_CONTENT" not in desc:
                    print(f"    -> Scraped Description ({len(desc)} chars).")
                    df.at[idx, 'abstract'] = desc
                    df.at[idx, 'abstract_source'] = 'selenium_llm_scrape'
                else:
                    print("    -> No content found (LLM rejected).")
                    df.at[idx, 'abstract'] = 'NO_CONTENT'
                    df.at[idx, 'abstract_source'] = 'selenium_llm_scrape_rejected'
                
                processed_count += 1
                
                # Periodic Save (Every 5 rows processed, regardless of outcome)
                if processed_count % 5 == 0:
                    df.to_csv(TARGET_CSV, index=False)
                    print(f"    (Saved - {processed_count} processed)")
                    
            except Exception as e:
                print(f"    -> Error scraping: {e}")
                # continue
                
    print(f"Done. Processed {processed_count} new abstracts.")
    df.to_csv(TARGET_CSV, index=False)

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY missing.")
    else:
        main()
