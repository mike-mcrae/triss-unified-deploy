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
OPENAI_MODEL = "gpt-4o-mini"
MAX_BODY_CHARS = 12000

def clean_text(text):
    if not text: return ""
    return text.strip().replace('\n', ' ').replace('\r', ' ')

def get_abstract_from_llm(client, raw_text):
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful academic assistant."},
                {"role": "user", "content": f"""
Please identify and extract the **Abstract** from the following text.

**Instructions:**
1. **Extraction**: If there is a clear "Abstract" section, extract it word-for-word.
2. **Paraphrasing (Fallback)**: If NO formal abstract exists, but the text contains an Introduction, Summary, or Conclusion that clearly describes the study's **Aims, Methods, and Results**, write a concise paragraph (150-250 words) summarizing the paper. 
    - Prefix paraphrased abstracts with: `[Paraphrased]: `
3. **Failure**: If the text is unrelated (e.g., login page, cookie banner, random footer text) or does not contain enough info to summarize the paper, return exactly: `NO_ABSTRACT_FOUND`.

**Raw Text:**
{raw_text[:MAX_BODY_CHARS]}
"""}
            ],
            max_tokens=800,
            temperature=0.3
        )
        content = response.choices[0].message.content.strip()
        if "NO_ABSTRACT_FOUND" in content:
            return None
        return content
    except Exception as e:
        print(f"    ! OpenAI Error: {e}")
        return None

def main():
    print(f"Loading {INPUT_CSV}...")
    if not os.path.exists(INPUT_CSV):
        print("File not found.")
        return

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    
    # Ensure columns exist
    for col in ['url_found', 'scraped_status', 'abstract', 'abstract_source']:
        if col not in df.columns:
            df[col] = ""

    # Filter for rows that meet criteria:
    # 1. Missing Abstract
    # 2. Have Url OR Tara Handle
    # 3. Not already scraped (scraped_status is empty)
    missing_abstract = df['abstract'].fillna("").astype(str).apply(lambda x: len(x) < 20)
    has_url = df['Url'].fillna("").astype(str).str.startswith("http")
    has_tara = df['Tara Handle'].fillna("").astype(str).str.startswith("http")
    not_scraped = df['scraped_status'].fillna("").astype(str) == ""
    
    target_mask = missing_abstract & (has_url | has_tara) & not_scraped
    target_indices = df[target_mask].index.tolist()
    
    print(f"Found {len(target_indices)} rows to scrape (Missing Abstract + Has URL/Tara).")
    
    if not target_indices:
        print("Nothing to do.")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: No OpenAI API Key found.")
        return

    client = OpenAI()

    # Single Browser Context (Headless=False)
    with SB(uc=True, test=True, headless=False) as sb:
        print("Browser Started (Non-Headless).")
        
        counter = 0
        for idx in target_indices:
            counter += 1
            row = df.loc[idx]
            title = str(row.get('Title', '')).strip()
            url_col = str(row.get('Url', '')).strip()
            tara = str(row.get('Tara Handle', '')).strip()
            
            # Prioritize Url, then Tara
            target_url = url_col if url_col.startswith("http") else tara
            
            if not target_url.startswith("http"):
                print(f"[{counter}/{len(target_indices)}] Skip {idx}: Invalid URL")
                continue
                
            print(f"\n[{counter}/{len(target_indices)}] Processing [{idx}]: {title[:50]}...")
            print(f"    -> URL: {target_url}")
            
            try:
                sb.uc_open_with_reconnect(target_url, reconnect_time=5)
                
                # Check blocks
                if "Sorry" in sb.get_title() or sb.is_element_visible('iframe[src*="recaptcha"]'):
                    print("    ! CAPTCHA detected. Please solve manually if needed...")
                    try: sb.uc_gui_click_captcha()
                    except: pass
                    time.sleep(5)
                
                time.sleep(random.uniform(3, 5))
                
                html = sb.get_page_source()
                soup = BeautifulSoup(html, 'html.parser')
                
                body_text = ""
                article = soup.find('article') or soup.find('div', role='main') or soup.find('main')
                if article:
                    body_text = article.get_text(separator=' ', strip=True)
                elif soup.body:
                    body_text = soup.body.get_text(separator=' ', strip=True)
                
                body_text = clean_text(body_text)
                
                if len(body_text) > 300:
                    print(f"    -> Extracted {len(body_text)} chars. Sending to LLM...")
                    abstract = get_abstract_from_llm(client, body_text)
                    
                    if abstract:
                        print("    -> SUCCESS: Abstract Found/Paraphrased.")
                        df.at[idx, 'abstract'] = abstract
                        df.at[idx, 'abstract_source'] = "url_tara_scrape_llm"
                        df.at[idx, 'url_found'] = target_url
                        df.at[idx, 'scraped_status'] = "success"
                    else:
                        print("    -> Failed: LLM could not find abstract.")
                        df.at[idx, 'scraped_status'] = "no_abstract_in_text"
                else:
                    print(f"    -> Failed: Text too short ({len(body_text)} chars).")
                    df.at[idx, 'scraped_status'] = "text_too_short"
                    
            except Exception as e:
                print(f"    ! Error: {e}")
                df.at[idx, 'scraped_status'] = f"error: {str(e)}"
            
            # Save every 5 rows
            if counter % 5 == 0:
                df.to_csv(INPUT_CSV, index=False)
                print("    (Saved Progress)")

    # Final Save
    df.to_csv(INPUT_CSV, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
