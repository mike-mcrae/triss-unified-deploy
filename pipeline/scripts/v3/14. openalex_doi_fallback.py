import pandas as pd
import requests
import os
import difflib
import time
from urllib.parse import quote
import concurrent.futures
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
TARGET_CSV = os.path.join(BASE_DIR, "2c.all_listed_publications_2019_journal_like_doi.csv")

# OpenAlex API
API_URL = "https://api.openalex.org/works"
MAILTO = "mike.mcrae25@gmail.com"

def normalize_text(s):
    if not isinstance(s, str):
        return ""
    return " ".join(s.lower().split())

def similarity(s1, s2):
    return difflib.SequenceMatcher(None, normalize_text(s1), normalize_text(s2)).ratio()

def check_author_match(input_authors, openalex_authorships):
    if not input_authors or not openalex_authorships:
        return False
        
    input_norm = normalize_text(input_authors)
    
    for authorship in openalex_authorships:
        author = authorship.get('author', {})
        display_name = author.get('display_name', '')
        if not display_name:
            continue
            
        # Check last name / simple name parts
        # OpenAlex names are usually "First Last" or "Last, First"
        name_parts = normalize_text(display_name).split()
        for part in name_parts:
            if len(part) > 3 and part in input_norm:
                return True
                
    return False

def get_openalex_candidate(title, authors, year_descending):
    if not title or len(str(title)) < 5:
        return None, "Title too short"

    # Use 'search' parameter for best relevance
    query = quote(title)
    url = f"{API_URL}?search={query}&per-page=5&mailto={MAILTO}"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 429:
            time.sleep(10) # Heavy Backoff
            response = requests.get(url, timeout=10)
            
        if response.status_code != 200:
            return None, f"API Error: {response.status_code}"
            
        data = response.json()
        results = data.get('results', [])
    except Exception as e:
        return None, f"Request Error: {str(e)}"

    input_norm = normalize_text(title)
    try:
        input_year = int(float(year_descending))
    except:
        input_year = None

    for item in results:
        cand_title = item.get('title', "")
        if not cand_title:
            continue
            
        score = similarity(title, cand_title)
        
        cand_year = item.get('publication_year')
        
        year_match = False
        if input_year and cand_year:
            if abs(input_year - int(cand_year)) <= 1:
                year_match = True
        elif not input_year:
             year_match = True # Relax if we don't know year
        
        # OpenAlex authors structure: 'authorships': [{'author': {'display_name': ...}}]
        author_match = check_author_match(authors, item.get('authorships', []))
        
        # Matching Logic
        if score >= 0.90 and year_match and author_match:
             doi = item.get('doi')
             if doi:
                 # Clean DOI (OpenAlex returns https://doi.org/...)
                 doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
                 return {
                     "crossref_doi": doi,
                     "crossref_confidence": "high",
                     "crossref_match_reason": f"OpenAlex Match: Sim {score:.2f}, Year, Auth",
                     "alex_match_reason": "Match Found"
                 }, "Match Found"
        
    return None, "No strong match found"

def process_row(idx, row):
    # Aggressive rate limit delay inside worker
    time.sleep(2.0) 
    
    title = row.get('Title', '')
    authors = row.get('Authors', '')
    year = row.get('Year Descending', '')
    
    result, reason = get_openalex_candidate(title, authors, year)
    return idx, result, reason

def main():
    print(f"Loading Target: {TARGET_CSV}")
    df_target = pd.read_csv(TARGET_CSV, low_memory=False)
    
    # Add alex_match_reason column if not exists
    if 'alex_match_reason' not in df_target.columns:
        df_target['alex_match_reason'] = ""
        
    # Identify rows to process: Empty crossref_doi AND empty alex_match_reason
    # Also ignore those we already seemingly failed in openalex? (user said "where we still have crossref_doi blank")
    
    # Identify rows to process: Rows with API Error 429, 500, or "Daily Rate Limit Exceeded" in alex_match_reason
    # User instruction: "Try again noI just tested the debug_openal.py and it worked perfectly... Rerun andy 429 or 500 problems."
    
    if 'alex_match_reason' not in df_target.columns:
        print("Column alex_match_reason not found. Nothing to retry.")
        return

    # Check for 429, 500, or "Daily Rate Limit" in the reason string
    mask = df_target['alex_match_reason'].astype(str).str.contains('429|500|Daily Rate Limit', regex=True, na=False)
    
    rows_to_process = []
    for idx in df_target[mask].index:
        rows_to_process.append((idx, df_target.loc[idx]))
        
    print(f"Rows to retry (Errors): {len(rows_to_process)}")
    
    if not rows_to_process:
        print("No error rows need processing.")
        return

    processed_count = 0
    matched_count = 0
    
    # ThreadPool with SINGLE worker for strict sequential processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = {executor.submit(process_row, idx, row): idx for idx, row in rows_to_process}
        
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                idx, result, reason = future.result()
                
                if result:
                    df_target.at[idx, 'crossref_doi'] = result['crossref_doi']
                    df_target.at[idx, 'crossref_confidence'] = result['crossref_confidence']
                    df_target.at[idx, 'crossref_match_reason'] = result['crossref_match_reason']
                    df_target.at[idx, 'alex_match_reason'] = result['alex_match_reason']
                    matched_count += 1
                else:
                    df_target.at[idx, 'alex_match_reason'] = reason
                    
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                
            processed_count += 1
            if processed_count % 50 == 0:
                print(f"Processed {processed_count}/{len(rows_to_process)} | Matches: {matched_count}")
                df_target.to_csv(TARGET_CSV, index=False) # Incremental save

    
    print(f"Finished. Total Matches Found: {matched_count}")
    df_target.to_csv(TARGET_CSV, index=False)

if __name__ == "__main__":
    main()
