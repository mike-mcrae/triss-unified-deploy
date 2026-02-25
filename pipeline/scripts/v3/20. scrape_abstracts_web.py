import pandas as pd
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import time
import os
import re
import random
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
# Input file: Check if 2e.2d... or 2e... exists. Prefer 2e... based on recent ls
INPUT_CSV = os.path.join(BASE_DIR, "2e.all_listed_publications_2019_journal_like_no_abstract.csv")
BATCH_SIZE = 20 # Smaller batch for scraping

# Setup Session for Requests
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(max_retries=1)
session.mount('https://', adapter)
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
})

def clean_text(text):
    if not text: return ""
    return re.sub(r'\s+', ' ', text).strip()

def get_url_from_doi(doi):
    """Resolve DOI to URL."""
    try:
        url = f"https://doi.org/{doi}"
        r = session.get(url, timeout=10, allow_redirects=True)
        if r.status_code == 200:
            return r.url
    except:
        pass
    return None

def search_duckduckgo(query):
    """Search DDG and return first result URL."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=1))
            if results:
                return results[0]['href']
    except Exception as e:
        print(f"    ! DDG Error: {e}")
        time.sleep(2) # Backoff
    return None

def scrape_page(url):
    """Scrape abstract from page."""
    try:
        r = session.get(url, timeout=15)
        if r.status_code != 200:
            return None
        
        soup = BeautifulSoup(r.content, 'html.parser')
        
        # Heuristics
        # 1. Meta description (often contains abstract/summary)
        meta_desc = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
        if meta_desc:
            content = meta_desc.get('content', '')
            if len(content) > 100: # Arbitrary threshold for "abstract-like"
                return clean_text(content)

        # 2. Look for "Abstract" header
        # Find headers (h1, h2, h3, strong) with text "Abstract"
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'strong', 'b']):
            if 'abstract' in tag.get_text().lower():
                # Get next sibling or parent's next sibling
                # This is tricky. Often abstract is in a p tag after or a div class="abstract"
                pass 
                
        # 3. Class/ID "abstract"
        abstract_div = soup.find(class_=re.compile(r'abstract', re.I)) or soup.find(id=re.compile(r'abstract', re.I))
        if abstract_div:
            return clean_text(abstract_div.get_text())
            
    except Exception as e:
        print(f"    ! Scrape Error: {e}")
    return None

def main():
    print(f"Loading {INPUT_CSV}...")
    if not os.path.exists(INPUT_CSV):
        # Fallback to 2e.2d... just in case
        fallback = os.path.join(BASE_DIR, "2e.2d.all_listed_publications_2019_journal_like_no_abstract.csv")
        if os.path.exists(fallback):
            print(f"Falling back to {fallback}")
            INPUT_CSV_REAL = fallback
        else:
            print("File not found.")
            return
    else:
        INPUT_CSV_REAL = INPUT_CSV

    df = pd.read_csv(INPUT_CSV_REAL, low_memory=False)
    
    # Init columns if they don't exist
    for col in ['url_found', 'scraped_status', 'abstract']:
        if col not in df.columns:
            df[col] = ""
            
    # Fill NaNs with empty strings to ensure boolean comparisons work
    df['abstract'] = df['abstract'].fillna("").astype(str)
    df['url_found'] = df['url_found'].fillna("").astype(str)
    df['scraped_status'] = df['scraped_status'].fillna("").astype(str)
            
    # Identify rows to process
    # Those with missing abstract AND haven't been scraped yet (scraped_status != 'done' and != 'failed')
    def has_abstract(val):
        return len(str(val)) >= 50 and "No abstract available" not in str(val)

    mask_process = ~df['abstract'].apply(has_abstract) & (df['scraped_status'] == "")
    indices = df[mask_process].index.tolist()
    
    print(f"Found {len(indices)} rows to scrape.")
    
    save_counter = 0
    
    try:
        for idx in indices:
            doi = str(df.loc[idx, 'crossref_doi']).strip() if 'crossref_doi' in df.columns else ""
            title = str(df.loc[idx, 'Title']).strip()
            
            url = None
            
            # Strategy A: DOI
            if doi and doi != "nan":
                # If we have a DOI, try to resolve it
                # Logic: Is it likely we have a URL?
                # Actually, 776 rows SHOULD have a DOI but failed API.
                # So scraping the DOI landing page is the best bet.
                clean_doi = doi.lower().replace('https://doi.org/', '').replace('doi.org/', '')
                print(f"[{idx}] Resolving DOI: {clean_doi}...")
                url = get_url_from_doi(clean_doi)
            
            # Strategy B: Search
            if not url:
                if title and title != "nan":
                    print(f"[{idx}] Searching: {title[:50]}...")
                    url = search_duckduckgo(f'"{title}" abstract')
                    time.sleep(random.uniform(2, 4)) # Search rate limit
            
            # Scrape
            abstract = None
            if url:
                print(f"    -> Scrape: {url}")
                df.at[idx, 'url_found'] = url
                abstract = scrape_page(url)
                time.sleep(random.uniform(1, 3)) # Scrape rate limit
            
            # Update
            if abstract and len(abstract) >= 50:
                print(f"    -> Found Abstract ({len(abstract)} chars)")
                df.at[idx, 'abstract'] = abstract
                df.at[idx, 'abstract_source'] = 'web_scrape'
                df.at[idx, 'scraped_status'] = 'success'
                df.at[idx, 'abstract_retrieved'] = 1
                df.at[idx, 'abstract_char_count'] = len(abstract)
            else:
                print("    -> No abstract found.")
                df.at[idx, 'scraped_status'] = 'failed'  # Mark as failed so we don't retry endlessly in same run if restarted? Or keep empty? 
                # Let's mark 'failed' to track progress.
            
            save_counter += 1
            if save_counter >= BATCH_SIZE:
                print("Saving batch...")
                df.to_csv(INPUT_CSV_REAL, index=False)
                save_counter = 0

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        print("Final save...")
        df.to_csv(INPUT_CSV_REAL, index=False)
        print("Done.")

if __name__ == "__main__":
    main()
