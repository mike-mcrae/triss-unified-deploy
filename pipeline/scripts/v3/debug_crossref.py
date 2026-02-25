import pandas as pd
import requests
import difflib
from urllib.parse import quote
import os
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

# Re-use logic from main script for consistency
# I will copy the get_crossref_candidate function logic here to ensure exact behavior

API_URL = "https://api.crossref.org/works"
MAILTO = "mike.mcrae25@gmail.com"

def normalize_text(s):
    if not isinstance(s, str):
        return ""
    return " ".join(s.lower().split())

def similarity(s1, s2):
    return difflib.SequenceMatcher(None, normalize_text(s1), normalize_text(s2)).ratio()

def check_author_match(input_authors, candidate_authors):
    if not input_authors or not candidate_authors:
        return False 
    input_norm = normalize_text(input_authors)
    for auth in candidate_authors:
        family = auth.get('family', '')
        if not family: continue
        family_norm = normalize_text(family)
        if len(family_norm) > 2 and family_norm in input_norm:
            return True
    return False

def get_crossref_candidate(title, authors, year_descending):
    query_title = quote(str(title)[:300]) # Truncate likely not needed but safe
    
    params = {'query.title': title, 'rows': 3, 'mailto': MAILTO}
    if authors and isinstance(authors, str):
        params['query.author'] = authors
        
    try:
        response = requests.get(API_URL, params=params, timeout=10)
        data = response.json()
        items = data.get('message', {}).get('items', [])
    except Exception as e:
        return None, f"Error: {e}"

    print(f"\n--- Searching: {title} ({year_descending}) ---")
    print(f"    Authors Input: {authors}")
    
    best_res = None
    
    for i, item in enumerate(items):
        cand_title = item.get('title', [''])[0]
        cand_doi = item.get('DOI')
        cand_authors = item.get('author', [])
        
        # Year
        cand_year = None
        for date_field in ['published-print', 'published-online', 'created']:
            parts = item.get(date_field, {}).get('date-parts', [])
            if parts and parts[0]:
                cand_year = parts[0][0]
                break
                
        score = similarity(title, cand_title)
        
        # Verification
        year_match = False
        try:
             # Basic int conv
             input_year = int(float(year_descending))
             if cand_year and abs(input_year - int(cand_year)) <= 1:
                 year_match = True
        except:
             year_match = True # default if input bad
             
        author_match = check_author_match(authors, cand_authors)
        
        print(f"  [{i+1}] {cand_title} ({cand_year})")
        print(f"      DOI: {cand_doi}")
        print(f"      Sim: {score:.2f} | YearMatch: {year_match} | AuthMatch: {author_match}")
        
    return "Done", "Done"

def main():
    # Load sample 
    # Load sample from original input and filter
    df_raw = pd.read_csv(str(_PIPELINE_ROOT / "1. data/1. raw/0. profiles/v3/2c.all_listed_publications_2019_filter.csv"))
    journal_mask = df_raw['Publication Type'].str.strip().str.lower().isin(["journal article", "journal", "working paper"])
    df_sample = df_raw[journal_mask].head(5)
    
    for idx, row in df_sample.iterrows():
        get_crossref_candidate(row['Title'], row['Authors'], row['Year Descending'])

if __name__ == "__main__":
    main()
