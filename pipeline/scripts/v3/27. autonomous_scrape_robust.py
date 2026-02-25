import pandas as pd
from seleniumbase import SB
from bs4 import BeautifulSoup
import time
import os
import random
import urllib.parse
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
INPUT_CSV = os.path.join(BASE_DIR, "2e.all_listed_publications_2019_journal_like_no_abstract copy.csv")
OPENAI_MODEL = "gpt-4o-mini"
MAX_BODY_CHARS = 15000

def clean_text(text):
    if not text: return ""
    return text.strip().replace('\n', ' ').replace('\r', ' ')

def get_abstract_or_classification(client, raw_text, title, author):
    """
    Uses LLM to:
    1. Extract Abstract.
    2. Paraphrase if missing but inferable.
    3. Classify if it's NOT a journal article (e.g., Book Review, Obituary).
    """
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful academic assistant."},
                {"role": "user", "content": f"""
Analyze the following text to find the Abstract for the paper:
**Title**: {title}
**Author**: {author}

**Instructions:**
1. **Extraction**: If a clear Abstract exists, extract it word-for-word.
2. **Paraphrase**: If NO formal abstract, but the text allows you to summarize the study's Aims, Methods, and Results, write a 150-250 word summary. Prefix with `[Paraphrased]:`.
3. **Classification**: If the text indicates this is NOT a standard research paper, return one of these exact phrases:
    - `TYPE: BOOK_REVIEW`
    - `TYPE: OBITUARY`
    - `TYPE: EDITORIAL`
    - `TYPE: ERRATUM`
    - `TYPE: TOC` (Table of Contents)
4. **Failure**: If the text is unrelated (login page, cookies) or insufficient, return `NO_ABSTRACT_FOUND`.

**Raw Text:**
{raw_text[:MAX_BODY_CHARS]}
"""}
            ],
            max_tokens=800,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"    ! OpenAI Error: {e}")
        return None

def google_search_urls(sb, query):
    links = []
    try:
        search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
        sb.uc_open_with_reconnect(search_url, reconnect_time=4)
        
        # Check title/content for blocks
        if "Sorry" in sb.get_title() or sb.is_element_visible('iframe[src*="recaptcha"]'):
             print("    ! Google CAPTCHA. Waiting 10s...")
             try: sb.uc_gui_click_captcha()
             except: pass
             time.sleep(10)
             
        if sb.is_element_visible('button#L2AGLb') or sb.is_element_visible('button#W0wltc'):
             try: sb.click('button#L2AGLb')
             except: sb.click('button#W0wltc')
             time.sleep(2)
        
        time.sleep(2)
        
        selectors = ["div#search a h3", "a > h3"]
        for sel in selectors:
            try:
                elements = sb.find_elements(sel)
                for el in elements:
                    try:
                        parent = el.find_element("./..")
                        href = parent.get_attribute("href")
                        if href and "google.com" not in href and href.startswith("http"):
                             links.append(href)
                    except: continue
                    if len(links) >= 2: break
            except: pass
            if links: break
    except: pass
    return links

def ddg_search_urls(sb, query):
    urls = []
    try:
        ddg_url = f"https://duckduckgo.com/?q={urllib.parse.quote(query)}"
        sb.uc_open_with_reconnect(ddg_url, reconnect_time=4)
        time.sleep(3)
        elements = sb.find_elements("a[data-testid='result-title-a']") or sb.find_elements("h2 a")
        for el in elements:
             href = el.get_attribute("href")
             if href: urls.append(href)
             if len(urls) >= 2: break
    except: pass
    return urls

def scrape_text(sb, url):
    try:
        sb.uc_open_with_reconnect(url, reconnect_time=5)
        # Check blocks
        if "Sorry" in sb.get_title() or sb.is_element_visible('iframe[src*="recaptcha"]'):
            time.sleep(5)
            try: sb.uc_gui_click_captcha()
            except: pass
        
        time.sleep(random.uniform(3, 6))
        
        html = sb.get_page_source()
        soup = BeautifulSoup(html, 'html.parser')
        
        body_text = ""
        article = soup.find('article') or soup.find('div', role='main') or soup.find('main')
        if article:
            body_text = article.get_text(separator=' ', strip=True)
        elif soup.body:
            body_text = soup.body.get_text(separator=' ', strip=True)
        
        return clean_text(body_text)
    except Exception as e:
        print(f"    ! Scrape Error {url}: {e}")
        return ""

def main():
    print(f"Loading {INPUT_CSV}...")
    if not os.path.exists(INPUT_CSV):
        print("File not found.")
        return

    try:
        df = pd.read_csv(INPUT_CSV, low_memory=False, encoding='utf-8')
    except UnicodeDecodeError:
        print("    ! UTF-8 failed. Trying 'latin1'...")
        df = pd.read_csv(INPUT_CSV, low_memory=False, encoding='latin1')
    
    # Init columns
    for col in ['url_found', 'scraped_status', 'abstract', 'abstract_source', 'failure_reason']:
        if col not in df.columns:
            df[col] = ""

    # Filter: Missing Abstract AND Not successfully Scraped
    # We re-try failures to "give it a good crack" unless it's a known TYPE mismatch
    mask = (df['abstract'].fillna("").astype(str).apply(lambda x: len(x) < 20)) & \
           (~df['failure_reason'].astype(str).str.startswith("TYPE:"))
           
    indices = df[mask].index.tolist()
    print(f"Found {len(indices)} rows to process.")
    
    if not indices:
        print("All done!")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: No OpenAI API Key found.")
        return

    client = OpenAI()

    with SB(uc=True, test=True, headless=False) as sb:
        print("Browser Active. Starting Autonomous Run...")
        
        counter = 0
        for idx in indices:
            counter += 1
            row = df.loc[idx]
            title = str(row.get('Title', '')).strip()
            author = str(row.get('Author', '')).strip()
            doi = str(row.get('crossref_doi', '')).strip()
            url_col = str(row.get('Url', '')).strip()
            tara = str(row.get('Tara Handle', '')).strip()
            
            print(f"\n[{counter}/{len(indices)}] Processing [{idx}]: {title[:40]}...")
            
            # --- Strategy 1: Direct Link ---
            target_urls = []
            if doi and doi != "nan": target_urls.append(f"https://doi.org/{doi}")
            if url_col.startswith("http"): target_urls.append(url_col)
            if tara.startswith("http"): target_urls.append(tara)
            
            # --- Strategy 2 & 3: Search (if needed) ---
            if not target_urls:
                query = f"{title} {author}"
                print(f"    -> Searching Google: {query}")
                g_links = google_search_urls(sb, query)
                if g_links:
                    target_urls.extend(g_links)
                else:
                    print("    -> Google failed. Searching DDG...")
                    d_links = ddg_search_urls(sb, query)
                    target_urls.extend(d_links)

            # --- Try URLs ---
            success = False
            for url in target_urls:
                if not url: continue
                print(f"    -> Scraping: {url}")
                text = scrape_text(sb, url)
                
                if len(text) > 300:
                    result = get_abstract_or_classification(client, text, title, author)
                    
                    if result:
                        if result.startswith("TYPE:"):
                            print(f"    -> Classified: {result}")
                            df.at[idx, 'failure_reason'] = result
                            df.at[idx, 'scraped_status'] = "classified_type_mismatch"
                            success = True
                            break
                        elif "NO_ABSTRACT_FOUND" not in result:
                            print("    -> SUCCESS: Abstract Found.")
                            df.at[idx, 'abstract'] = result
                            df.at[idx, 'abstract_source'] = "robust_scrape_llm"
                            df.at[idx, 'url_found'] = url
                            df.at[idx, 'scraped_status'] = "success"
                            success = True
                            break
                        else:
                            print("    -> LLM: No abstract in text.")
            
            if not success:
                print("    -> FAILED (All methods tried).")
                df.at[idx, 'scraped_status'] = "failed_all_attempts"
            
            # Save frequently
            if counter % 5 == 0:
                df.to_csv(INPUT_CSV, index=False)
                print("    (Saved)")

    df.to_csv(INPUT_CSV, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
