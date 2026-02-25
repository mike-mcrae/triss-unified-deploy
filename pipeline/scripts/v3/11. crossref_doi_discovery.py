import pandas as pd
import requests
import time
import os
import difflib
import sys
import concurrent.futures
from urllib.parse import quote
import re
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
INPUT_CSV = os.path.join(BASE_DIR, "2c.all_listed_publications_2019_filter.csv")
OUTPUT_JOURNAL = os.path.join(BASE_DIR, "2c.all_listed_publications_2019_journal_like.csv")
OUTPUT_BOOKS = os.path.join(BASE_DIR, "2c.all_listed_publications_2019_books_other.csv")

# Target Types for DOI Discovery
JOURNAL_TYPES = [
    "journal article",
    "journal",
    "working paper"
]

# Crossref API
API_URL = "https://api.crossref.org/works"
MAILTO = "mike.mcrae25@gmail.com"

def normalize_text(s):
    if not isinstance(s, str):
        return ""
    return " ".join(s.lower().split())

def similarity(s1, s2):
    return difflib.SequenceMatcher(None, normalize_text(s1), normalize_text(s2)).ratio()

def extract_surname(authors_str):
    """
    Extract the first meaningful surname from an Authors string.
    Simple heuristic: take the last word of the first comma-separated part,
    or just the last word if no commas. 
    Adjust based on 'Smith, J.' vs 'J. Smith'.
    """
    if not authors_str or not isinstance(authors_str, str):
        return None
    
    # Clean up
    cleaned = authors_str.strip()
    if not cleaned:
        return None
        
    # Split by comma (assuming multiple authors separated by comma)
    first_author_part = cleaned.split(',')[0].strip()
    
    # If "Last, First" format (common in bibliographies)
    # But input format might be "Sonia Bishop" or "Bishop, Sonia".
    # Let's try to detect.
    
    # If the string has a comma, it might be "Last, First" OR "Author 1, Author 2".
    # Let's inspect the `Authors` column format from previous steps. 
    # Example from preview: "Abdel-Ghaffar SA, Huth AG..." -> seems to be "Last Initials" or "Last First".
    # "bishops@tcd.ie" -> Bishop.
    # Let's assume the first token before a comma is a surname? 
    # Or just use the whole string for the query 'query.author'.
    
    return cleaned

def check_author_match(input_authors, candidate_authors):
    """
    Check if any author in candidate record matches the input authors.
    Input authors: string
    Candidate authors: list of dicts [{'given': '...', 'family': '...'}]
    """
    if not input_authors or not candidate_authors:
        return False # formatting mismatch, can't verify
        
    input_norm = normalize_text(input_authors)
    
    match_found = False
    for auth in candidate_authors:
        family = auth.get('family', '')
        if not family:
            continue
            
        family_norm = normalize_text(family)
        
        # Check if family name is present in input string
        # Strict check: exact word match?
        # "bishop" in "bishop, sonia" -> True
        if len(family_norm) > 2 and family_norm in input_norm:
            match_found = True
            break
            
    return match_found

def get_crossref_candidate(title, authors, year_descending):
    if not title or len(str(title)) < 5:
        return None, "Title too short", 0
        
    # Build Query
    # https://api.crossref.org/works?query.title=...&query.author=...&rows=3
    params = {
        'query.title': title,
        'rows': 3,
        'mailto': MAILTO
    }
    
    # Add author query if available (extract simplified surname/string)
    # Passing the full authors string to query.author usually works well in Crossref
    if authors and isinstance(authors, str):
        params['query.author'] = authors
        
    try:
        response = requests.get(API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        items = data.get('message', {}).get('items', [])
    except Exception as e:
        return None, f"API Error: {str(e)}", 0
        
    best_candidate = None
    best_score = 0
    best_reason = "No match found"
    
    input_year = None
    try:
        input_year = int(float(year_descending))
    except (ValueError, TypeError):
        pass

    for item in items:
        cand_title_list = item.get('title', [])
        cand_title = cand_title_list[0] if cand_title_list else ""
        
        # Year
        # keys: 'published-print', 'published-online', 'created'
        # structure: {'date-parts': [[2021, 1, 1]]}
        cand_year = None
        for date_field in ['published-print', 'published-online', 'created']:
            parts = item.get(date_field, {}).get('date-parts', [])
            if parts and parts[0]:
                cand_year = parts[0][0] # Year is first element
                break
                
        cand_doi = item.get('DOI')
        cand_authors = item.get('author', [])
        
        if not cand_title:
            continue
            
        score = similarity(title, cand_title)
        
        # Year Check
        year_match = False
        if input_year and cand_year:
            try:
                if abs(input_year - int(cand_year)) <= 1:
                    year_match = True
            except:
                pass
        elif not input_year:
            # If no input year, be slightly permissive?
            # Or strict?
            year_match = True # default
            
        # Author Check
        author_match = check_author_match(authors, cand_authors)
        
        # Confidence Logic
        confidence = "none"
        match_reason = ""
        
        # We require Author Match for high/medium?
        # User said: "match on title and surname = year if possible"
        
        if score >= 0.90 and year_match and author_match:
            confidence = "high"
            match_reason = f"Title Sim: {score:.2f}, Year Match, Author Match"
        elif score >= 0.95 and year_match:
             # Very high title match, maybe Author mismatch/missing in input?
             # Let's count as medium?
             confidence = "medium"
             match_reason = f"Title Sim: {score:.2f} (Very High), Year Match, Author Mismatch/Unchecked"
        elif score >= 0.82 and year_match and author_match:
            confidence = "medium"
            match_reason = f"Title Sim: {score:.2f}, Year Match, Author Match"
        else:
            confidence = "low"
            match_reason = f"Title Sim: {score:.2f}, Year Match: {year_match}, Author Match: {author_match}"
            
        # Selection
        if confidence in ["high", "medium"]:
             # Is better?
             is_better = False
             current_val = 2 if confidence == "high" else 1
             best_val = 0
             if best_candidate:
                 best_val = 2 if best_candidate['confidence'] == "high" else 1
                 
             if current_val > best_val:
                 is_better = True
             elif current_val == best_val and score > best_score:
                 is_better = True
                 
             if is_better:
                 best_candidate = {
                     "crossref_doi": cand_doi,
                     "crossref_confidence": confidence,
                     "crossref_match_reason": match_reason,
                     "confidence": confidence
                 }
                 best_score = score
                 best_reason = match_reason

    if best_candidate:
        del best_candidate['confidence']
        return best_candidate, best_reason, best_score
        
    return None, best_reason, best_score

def process_row_wrapper(args):
    idx, row = args
    title = row.get('Title', '')
    authors = row.get('Authors', '')
    year = row.get('Year Descending', '')
    
    result, reason, score = get_crossref_candidate(title, authors, year)
    return idx, result, reason, score

def main():
    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    
    # 1. Split
    mask_journal = df['Publication Type'].str.strip().str.lower().isin(JOURNAL_TYPES)
    df_journal = df[mask_journal].copy()
    df_books = df[~mask_journal].copy()
    
    print(f"Journal-like: {len(df_journal)}")
    print(f"Books/Other: {len(df_books)}")
    
    df_books.to_csv(OUTPUT_BOOKS, index=False)
    
    # 2. Setup Journal DF
    new_cols = ['crossref_doi', 'crossref_confidence', 'crossref_match_reason']
    
    # helper for existing
    df_processing = df_journal
    
    # Check resume
    if os.path.exists(OUTPUT_JOURNAL):
        print(f"Resuming form {OUTPUT_JOURNAL}...")
        df_existing = pd.read_csv(OUTPUT_JOURNAL, low_memory=False)
        # Ensure strings
        for c in new_cols:
            if c not in df_existing.columns:
                df_existing[c] = ""
            df_existing[c] = df_existing[c].astype(str).replace('nan', '')
            
        df_processing = df_existing
    else:
        for c in new_cols:
            df_processing[c] = ""
        df_processing.to_csv(OUTPUT_JOURNAL, index=False)
        
    print("Starting Crossref Discovery (Multi-threaded)...")
    
    # Filter rows needing processing
    rows_to_process = []
    
    # Convert to list of dicts for safe processing? Or just iterate index.
    # We iterate index of df_processing
    for idx in df_processing.index:
         c_doi = str(df_processing.at[idx, 'crossref_doi'])
         c_reason = str(df_processing.at[idx, 'crossref_match_reason'])
         is_done = (len(c_doi) > 5 and 'nan' not in c_doi.lower()) or (len(c_reason) > 3)
         if not is_done:
             # Package row data
             row_data = {
                 'Title': df_processing.at[idx, 'Title'],
                 'Authors': df_processing.at[idx, 'Authors'],
                 'Year Descending': df_processing.at[idx, 'Year Descending']
             }
             rows_to_process.append((idx, row_data))
             
    print(f"Rows to process: {len(rows_to_process)}")
    
    processed_count = 0
    stats = {'high': 0, 'medium': 0, 'none': 0}
    
    # Executor
    # Crossref polite pool allows reasonable parallelism
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_row_wrapper, item): item[0] for item in rows_to_process}
        
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                idx, result, reason, score = future.result()
                
                if result:
                    df_processing.at[idx, 'crossref_doi'] = str(result['crossref_doi'])
                    df_processing.at[idx, 'crossref_confidence'] = str(result['crossref_confidence'])
                    df_processing.at[idx, 'crossref_match_reason'] = str(result['crossref_match_reason'])
                    stats[result['crossref_confidence']] += 1
                else:
                    df_processing.at[idx, 'crossref_match_reason'] = str(reason)
                    stats['none'] += 1
                    
            except Exception as e:
                print(f"Error row {idx}: {e}")
                
            processed_count += 1
            if processed_count % 50 == 0:
                print(f"Processed {processed_count}/{len(rows_to_process)} | High: {stats['high']} | Med: {stats['medium']} | None: {stats['none']}")
                df_processing.to_csv(OUTPUT_JOURNAL, index=False)
                
    print("Finalizing...")
    df_processing.to_csv(OUTPUT_JOURNAL, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
