import pandas as pd
import requests
import os
import difflib
import concurrent.futures
from urllib.parse import quote
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
INPUT_CSV = os.path.join(BASE_DIR, "2c.all_listed_publications_2019_journal_like_doi.csv")
# We write back to the SAME file incrementally
OUTPUT_CSV = INPUT_CSV 

# Crossref API
API_URL = "https://api.crossref.org/works"
MAILTO = "mike.mcrae25@gmail.com"

def normalize_text(s):
    if not isinstance(s, str):
        return ""
    return " ".join(s.lower().split())

def similarity(s1, s2):
    return difflib.SequenceMatcher(None, normalize_text(s1), normalize_text(s2)).ratio()

def check_author_match(input_authors, candidate_authors):
    """
    Check if any author in candidate record matches the input authors.
    Input authors: string
    Candidate authors: list of dicts [{'given': '...', 'family': '...'}]
    """
    if not input_authors or not candidate_authors:
        return False 
        
    input_norm = normalize_text(input_authors)
    
    for auth in candidate_authors:
        family = auth.get('family', '')
        if not family:
            continue
            
        family_norm = normalize_text(family)
        
        # Check if family name is present in input string
        if len(family_norm) > 2 and family_norm in input_norm:
            return True
            
    return False

def get_crossref_candidate_pass2(title, authors, year_descending):
    if not title or len(str(title)) < 5:
        return None, "Title too short", 0
        
    # Build Query
    # https://api.crossref.org/works?query.title=...&query.author=...&rows=5 (Step 1: Top 5)
    params = {
        'query.title': title,
        'rows': 5,
        'mailto': MAILTO
    }
    
    if authors and isinstance(authors, str):
        params['query.author'] = authors
        
    try:
        response = requests.get(API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        items = data.get('message', {}).get('items', [])
    except Exception as e:
        return None, f"API Error: {str(e)}", 0
        
    # Step 2 & 3: Selection
    input_norm = normalize_text(title)
    
    input_year = None
    try:
        input_year = int(float(year_descending))
    except:
        pass

    for item in items:
        cand_title_list = item.get('title', [])
        cand_title = cand_title_list[0] if cand_title_list else ""
        cand_norm = normalize_text(cand_title)
        
        if not cand_title:
            continue
            
        # Step 2: Scoring Fields
        score = similarity(title, cand_title)
        title_contains_input = input_norm in cand_norm if len(input_norm) > 10 else False # Safety for short titles
        
        cand_year = None
        for date_field in ['published-print', 'published-online', 'created']:
            parts = item.get(date_field, {}).get('date-parts', [])
            if parts and parts[0]:
                cand_year = parts[0][0]
                break
        
        year_match = False
        if input_year and cand_year:
            if abs(input_year - int(cand_year)) <= 1:
                year_match = True
        elif not input_year:
             year_match = True # default if input missing
             
        author_match = check_author_match(authors, item.get('author', []))
        
        # Step 3: Selection Rules
        
        # Rule 1 — Exact / Near-Exact Title Match
        # title_similarity ≥ 0.95 AND year_match == True
        if score >= 0.95 and year_match:
            return {
                "crossref_doi_pass2": item.get('DOI'),
                "crossref_confidence_pass2": "high",
                "crossref_match_reason_pass2": f"Rule 1: Sim {score:.2f}, YearMatch"
            }, "Match Found", score
            
        # Rule 2 — Containment Rescue
        # title_similarity < 0.95 AND title_contains_input_title == True
        # AND author_match == True AND year_match == True
        if score < 0.95 and title_contains_input and author_match and year_match:
            return {
                "crossref_doi_pass2": item.get('DOI'),
                "crossref_confidence_pass2": "high",
                "crossref_match_reason_pass2": f"Rule 2: Containment, AuthMatch, YearMatch"
            }, "Match Found", score

        # Rule 3 — Strong Traditional Match
        # title_similarity ≥ 0.90 AND author_match == True AND year_match == True
        if score >= 0.90 and author_match and year_match:
             return {
                "crossref_doi_pass2": item.get('DOI'),
                "crossref_confidence_pass2": "medium",
                "crossref_match_reason_pass2": f"Rule 3: Sim {score:.2f}, AuthMatch, YearMatch"
            }, "Match Found", score
            
    return None, "No match found", 0

def process_row_wrapper(args):
    idx, row = args
    title = row.get('Title', '')
    authors = row.get('Authors', '')
    year = row.get('Year Descending', '')
    
    result, reason, score = get_crossref_candidate_pass2(title, authors, year)
    return idx, result, reason, score

def main():
    # Init/Cleanup
    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV, low_memory=False)

    # The user wants "new dois found should go in the crossref_doi column".
    # We will write directly to 'crossref_doi', 'crossref_confidence', 'crossref_match_reason'.
    # We also drop _pass2 columns if they exist to clean up from previous run.
    drop_cols = ['crossref_doi_pass2', 'crossref_confidence_pass2', 'crossref_match_reason_pass2']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')

    # Filter Rows to Process
    # "Only process rows where: crossref_confidence is empty OR null"
    
    rows_to_process = []
    
    for idx in df.index:
        # Check Pass 1 (or existing result)
        conf = str(df.at[idx, 'crossref_confidence'])
        is_resolved = (len(conf) > 0 and 'nan' not in conf.lower())
        
        if is_resolved:
            continue
            
        # Add to queue
        row_data = {
             'Title': df.at[idx, 'Title'],
             'Authors': df.at[idx, 'Authors'],
             'Year Descending': df.at[idx, 'Year Descending']
        }
        rows_to_process.append((idx, row_data))
        
    print(f"Rows to process (Pass 2): {len(rows_to_process)}")
    
    if not rows_to_process:
        print("Nothing to do.")
        # If nothing to do but we dropped columns, ensure we save the clean version
        df.to_csv(OUTPUT_CSV, index=False)
        return

    processed_count = 0
    stats = {'high': 0, 'medium': 0, 'none': 0}
    
    # Executor
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_row_wrapper, item): item[0] for item in rows_to_process}
        
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                idx, result, reason, score = future.result()
                
                if result:
                    # Write to MAIN columns
                    df.at[idx, 'crossref_doi'] = str(result['crossref_doi_pass2'])
                    df.at[idx, 'crossref_confidence'] = str(result['crossref_confidence_pass2'])
                    df.at[idx, 'crossref_match_reason'] = str(result['crossref_match_reason_pass2'])
                    stats[result['crossref_confidence_pass2']] += 1
                else:
                    # Write failure reason to MAIN column (or append?)
                    # If empty, just write.
                    df.at[idx, 'crossref_match_reason'] = str(reason)
                    stats['none'] += 1
                    
            except Exception as e:
                print(f"Error row {idx}: {e}")
                
            processed_count += 1
            if processed_count % 50 == 0:
                print(f"Processed {processed_count}/{len(rows_to_process)} | High: {stats['high']} | Med: {stats['medium']} | None: {stats['none']}")
                df.to_csv(OUTPUT_CSV, index=False)
                
    print("Finalizing...")
    df.to_csv(OUTPUT_CSV, index=False)
    print("Pass 2 Complete.")

if __name__ == "__main__":
    main()
