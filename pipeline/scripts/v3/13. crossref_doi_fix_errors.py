import pandas as pd
import requests
import os
import difflib
import concurrent.futures
import time
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
TARGET_CSV = os.path.join(BASE_DIR, "2c.all_listed_publications_2019_journal_like_doi.csv")
SOURCE_CSV = os.path.join(BASE_DIR, "2c.all_listed_publications_2019_filter.csv")

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
    if not input_authors or not candidate_authors:
        return False 
    input_norm = normalize_text(input_authors)
    for auth in candidate_authors:
        family = auth.get('family', '')
        if not family:
            continue
        family_norm = normalize_text(family)
        if len(family_norm) > 2 and family_norm in input_norm:
            return True
    return False

def get_crossref_candidate_pass3(title, authors, year_descending):
    if not title or len(str(title)) < 5:
        return None, "Title too short", 0
        
    params = {
        'query.title': title,
        'rows': 5,
        'mailto': MAILTO
    }
    if authors and isinstance(authors, str):
        params['query.author'] = authors
        
    try:
        response = requests.get(API_URL, params=params, timeout=10)
        
        # Handle 429 specifically with backoff
        if response.status_code == 429:
            time.sleep(2) # Wait a bit
            response = requests.get(API_URL, params=params, timeout=10) # Retry once
            
        response.raise_for_status()
        data = response.json()
        items = data.get('message', {}).get('items', [])
    except Exception as e:
        # Return specific error so we know
        return None, f"API Error: {str(e)}", 0
        
    input_norm = normalize_text(title)
    try:
        input_year = int(float(year_descending))
    except:
        input_year = None

    for item in items:
        cand_title_list = item.get('title', [])
        cand_title = cand_title_list[0] if cand_title_list else ""
        cand_norm = normalize_text(cand_title)
        
        if not cand_title:
            continue
            
        score = similarity(title, cand_title)
        title_contains_input = input_norm in cand_norm if len(input_norm) > 10 else False
        
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
             year_match = True
             
        author_match = check_author_match(authors, item.get('author', []))
        
        # Rules (Same as Pass 2)
        match_found = False
        reason_str = ""
        conf = ""
        
        if score >= 0.95 and year_match:
            match_found = True
            conf = "high"
            reason_str = f"Rule 1: Sim {score:.2f}, YearMatch (Pass 3)"
        elif score < 0.95 and title_contains_input and author_match and year_match:
             match_found = True
             conf = "high"
             reason_str = f"Rule 2: Containment, AuthMatch, YearMatch (Pass 3)"
        elif score >= 0.90 and author_match and year_match:
             match_found = True
             conf = "medium"
             reason_str = f"Rule 3: Sim {score:.2f}, AuthMatch, YearMatch (Pass 3)"
             
        if match_found:
             return {
                "crossref_doi": item.get('DOI'),
                "crossref_confidence": conf,
                "crossref_match_reason": reason_str
            }, "Match Found", score
            
    return None, "No match found (Pass 3)", 0

def process_row_wrapper(args):
    idx, row_target, row_source = args
    
    # Use Source Data
    title = row_source.get('Title', '')
    authors = row_source.get('Authors', '')
    year = row_source.get('Year Descending', '')
    
    # Check if Source Title is actually better
    if not title or len(str(title)) < 5:
         return idx, None, "Title too short (Source confirmed)", 0
    
    # Retry
    # Add delay to avoid hammering 429
    time.sleep(0.5) 
    
    result, reason, score = get_crossref_candidate_pass3(title, authors, year)
    return idx, result, reason, score

def main():
    print(f"Loading Target: {TARGET_CSV}")
    df_target = pd.read_csv(TARGET_CSV, low_memory=False)
    
    print(f"Loading Source: {SOURCE_CSV}")
    df_source = pd.read_csv(SOURCE_CSV, low_memory=False)
    
    # Index Source by article_id for fast lookup
    # Also index by (n_id, Publication Reference) for fallback
    source_by_id = {}
    source_by_ref = {}
    
    for idx, row in df_source.iterrows():
        a_id = str(row.get('article_id', ''))
        if a_id and a_id != 'nan':
            source_by_id[a_id] = row
            
        n_id = str(row.get('n_id', '')).strip()
        pub_ref = normalize_text(row.get('Publication Reference', ''))
        if n_id and pub_ref:
            source_by_ref[(n_id, pub_ref)] = row

    rows_to_process = []
    
    # Identify candidates
    for idx in df_target.index:
        reason = str(df_target.at[idx, 'crossref_match_reason'])
        article_id = str(df_target.at[idx, 'article_id'])
        
        should_retry = False
        if "title too short" in reason.lower():
            should_retry = True
        elif "429" in reason:
            should_retry = True
            
        if should_retry:
            row_source = None
            
            # Try ID match
            if article_id in source_by_id:
                row_source = source_by_id[article_id]
            
            # Try Fallback: n_id + Publication Reference
            if row_source is None:
                n_id = str(df_target.at[idx, 'n_id']).strip()
                pub_ref = normalize_text(df_target.at[idx, 'Publication Reference'])
                if (n_id, pub_ref) in source_by_ref:
                    row_source = source_by_ref[(n_id, pub_ref)]
                    # print(f"Fallback matched for {n_id}!")
            
            if row_source is not None:
                rows_to_process.append((idx, df_target.loc[idx], row_source))
            else:
                print(f"Warning: Could not find source for article_id {article_id} (n_id {df_target.at[idx, 'n_id']})")
                
    print(f"Rows to retry: {len(rows_to_process)}")
    
    if not rows_to_process:
        print("Nothing to retry.")
        return

    processed_count = 0
    stats = {'high': 0, 'medium': 0, 'none': 0}
    
    # Executor (Lower workers to avoid 429)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_row_wrapper, item): item[0] for item in rows_to_process}
        
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                idx, result, reason, score = future.result()
                
                if result:
                    df_target.at[idx, 'crossref_doi'] = str(result['crossref_doi'])
                    df_target.at[idx, 'crossref_confidence'] = str(result['crossref_confidence'])
                    df_target.at[idx, 'crossref_match_reason'] = str(result['crossref_match_reason'])
                    stats[result['crossref_confidence']] += 1
                else:
                    # Update reason (e.g. from "429" to "No match found")
                    # Unless it's still 429...
                    df_target.at[idx, 'crossref_match_reason'] = str(reason)
                    stats['none'] += 1
                    
            except Exception as e:
                print(f"Error row {idx}: {e}")
                
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"Processed {processed_count}/{len(rows_to_process)} | High: {stats['high']} | Med: {stats['medium']} | None: {stats['none']}")
                start_save = time.time()
                df_target.to_csv(TARGET_CSV, index=False)
                
    print("Finalizing...")
    df_target.to_csv(TARGET_CSV, index=False)
    print("Pass 3 (Error Fix) Complete.")

if __name__ == "__main__":
    main()
