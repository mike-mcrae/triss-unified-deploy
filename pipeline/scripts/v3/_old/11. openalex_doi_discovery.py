import pandas as pd
import requests
import time
import os
import difflib
import sys
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
INPUT_CSV = os.path.join(BASE_DIR, "2c.all_listed_publications_2019.csv")
OUTPUT_JOURNAL = os.path.join(BASE_DIR, "2c.all_listed_publications_2019_journal_like.csv")
OUTPUT_BOOKS = os.path.join(BASE_DIR, "2c.all_listed_publications_2019_books_other.csv")

# Target Types for DOI Discovery
JOURNAL_TYPES = [
    "journal article",
    "journal",
    "working paper"
]

# OpenAlex API
API_URL = "https://api.openalex.org/works"
MAILTO = "mike.mcrae25@gmail.com" # Polite pool

def normalize_text(s):
    if not isinstance(s, str):
        return ""
    return " ".join(s.lower().split())

def similarity(s1, s2):
    return difflib.SequenceMatcher(None, normalize_text(s1), normalize_text(s2)).ratio()

def get_openalex_candidate(title, year_descending):
    """
    Query OpenAlex for a title and return the best candidate if it matches criteria.
    """
    if not title or len(str(title)) < 5:
        return None, "Title too short", 0
        
    query = quote(title)
    url = f"{API_URL}?search={query}&per-page=5&mailto={MAILTO}"
    
    input_year = None
    try:
        input_year = int(float(year_descending))
    except (ValueError, TypeError):
        pass

    # Retry Logic
    max_retries = 5
    base_delay = 1
    results = []

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            
            if response.status_code == 429:
                sleep_time = base_delay * (2 ** attempt)
                # print(f"Rate limited. Sleeping {sleep_time}s...")
                time.sleep(sleep_time)
                continue
                
            response.raise_for_status()
            data = response.json()
            results = data.get('results', [])
            break # Success
            
        except Exception as e:
            if attempt == max_retries - 1:
                return None, f"API Error: {str(e)}", 0
            time.sleep(1)
            
    best_candidate = None
    best_score = 0
    best_reason = "No match found"
    
    for item in results:
        cand_title = item.get('title')
        cand_year = item.get('publication_year')
        cand_doi = item.get('doi')
        cand_id = item.get('id')

        
        if not cand_title:
            continue
            
        score = similarity(title, cand_title)
        
        # Year check
        year_match = False
        if input_year and cand_year:
            if abs(input_year - cand_year) <= 1:
                year_match = True
        elif not input_year:
             # If we don't have an input year, we can't verify, but we shouldn't penalize too hard?
             # User said: "year matches (+/- 1)". If input year missing, maybe stricter title req?
             # For now, let's treat year_match as True if input missing but be careful.
             # Actually, most rows have years. Let's require year match if input year exists.
             year_match = True # default/weak pass
        
        # Confidence Logic
        confidence = "none"
        match_reason = ""
        
        if score >= 0.90 and year_match:
            confidence = "high"
            match_reason = f"Title Sim: {score:.2f}, Year Match"
        elif score >= 0.82 and year_match:
            confidence = "medium"
            match_reason = f"Title Sim: {score:.2f}, Year Match"
        else:
            confidence = "low"
            match_reason = f"Title Sim: {score:.2f} (Low)"
            
        if confidence in ["high", "medium"]:
            # If strictly better than previous best
            # Prioritize High over Medium
            # Prioritize higher score within same confidence
            
            is_better = False
            if best_candidate is None:
                is_better = True
            else:
                current_conf_val = 2 if confidence == "high" else 1
                best_conf_val = 2 if best_candidate['confidence'] == "high" else 1
                
                if current_conf_val > best_conf_val:
                    is_better = True
                elif current_conf_val == best_conf_val and score > best_score:
                    is_better = True
            
            if is_better:
                best_candidate = {
                    "openalex_doi": cand_doi,
                    "openalex_doi_confidence": confidence,
                    "openalex_match_reason": match_reason,
                    "openalex_work_id": cand_id,
                    "openalex_source": "openalex",
                    "confidence": confidence # internal helper
                }
                best_score = score
                best_reason = match_reason

    if best_candidate:
        # Remove internal helper
        del best_candidate['confidence']
        return best_candidate, best_reason, best_score
        
    return None, best_reason, best_score

def main():
    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"Total Rows: {len(df)}")
    
    # 1. Split
    mask_journal = df['Publication Type'].str.strip().str.lower().isin(JOURNAL_TYPES)
    df_journal = df[mask_journal].copy()
    df_books = df[~mask_journal].copy()
    
    print(f"Journal-like: {len(df_journal)}")
    print(f"Books/Other: {len(df_books)}")
    
    # Save Books immediately
    print(f"Saving Books/Other to {OUTPUT_BOOKS}...")
    df_books.to_csv(OUTPUT_BOOKS, index=False)
    
    # 2. Process Journal-Like
    # Check if resumable
    new_cols = ['openalex_doi', 'openalex_doi_confidence', 'openalex_match_reason', 'openalex_work_id', 'openalex_source']
    
    # Initialize columns if not exist
    for col in new_cols:
        if col not in df_journal.columns:
            df_journal[col] = "" # or None
            
    # If loading for resume, we might load OUTPUT_JOURNAL if it exists?
    # But user said "Operate ONLY on 2c.all_listed_publications_2019_journal_like.csv" implies we create it or load it.
    # Logic: If OUTPUT_JOURNAL exists, load it to resume. Else use df_journal split from input.
    
    if os.path.exists(OUTPUT_JOURNAL):
        print(f"Resuming from existing {OUTPUT_JOURNAL}...")
        df_journal_existing = pd.read_csv(OUTPUT_JOURNAL, low_memory=False)
        
        # Force string types for our target columns to avoid float/object mismatch
        for col in new_cols:
            if col in df_journal_existing.columns:
                df_journal_existing[col] = df_journal_existing[col].astype(str).replace('nan', '')
        
        # We need to map our split logic to this file? Or better, just count handled.
        # Assuming splitting logic is deterministic, we can use the existing file as the source of truth for progress.
        df_processing = df_journal_existing
    else:
        df_processing = df_journal
        # Save initially to establish file
        df_processing.to_csv(OUTPUT_JOURNAL, index=False)

    # Multi-threading
    import concurrent.futures

    print("Starting OpenAlex Discovery (Multi-threaded)...")
    
    # Identify rows to process
    rows_to_process = []
    for idx in df_processing.index:
        current_doi = str(df_processing.at[idx, 'openalex_doi'])
        current_reason = str(df_processing.at[idx, 'openalex_match_reason'])
        
        is_processed = (len(current_reason) > 3) or (len(current_doi) > 5 and 'nan' not in current_doi.lower())
        if not is_processed:
            rows_to_process.append(idx)
            
    print(f"Rows to process: {len(rows_to_process)}")
    
    # Process function wrapper
    def process_row(idx):
        title = df_processing.at[idx, 'Title']
        year = df_processing.at[idx, 'Year Descending']
        result, reason, score = get_openalex_candidate(title, year)
        return idx, result, reason, score

    processed_count = 0
    found_high = 0
    found_medium = 0
    found_none = 0
    
    # Execute
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Map indices to futures
        future_to_idx = {executor.submit(process_row, idx): idx for idx in rows_to_process}
        
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                idx, result, reason, score = future.result()
                
                if result:
                    df_processing.at[idx, 'openalex_doi'] = str(result['openalex_doi'])
                    df_processing.at[idx, 'openalex_doi_confidence'] = str(result['openalex_doi_confidence'])
                    df_processing.at[idx, 'openalex_match_reason'] = str(result['openalex_match_reason'])
                    df_processing.at[idx, 'openalex_work_id'] = str(result['openalex_work_id'])
                    df_processing.at[idx, 'openalex_source'] = str(result['openalex_source'])
                    
                    if result['openalex_doi_confidence'] == 'high':
                        found_high += 1
                    else:
                        found_medium += 1
                else:
                    df_processing.at[idx, 'openalex_match_reason'] = "No match found" if not reason else str(reason)
                    found_none += 1
                    
            except Exception as exc:
                print(f"Row {idx} generated an exception: {exc}")
                
            processed_count += 1
            if processed_count % 50 == 0:
                 print(f"Processed {processed_count}/{len(rows_to_process)} | High: {found_high} | Med: {found_medium} | None: {found_none}")
                 # Save checkpoint (not thread safe to write while reading? Actually df is in memory, write is safeish if single thread doing write)
                 # Yes, we are in main thread loop here.
                 df_processing.to_csv(OUTPUT_JOURNAL, index=False)

    # Final Save
    print("Finalizing...")
    df_processing.to_csv(OUTPUT_JOURNAL, index=False)
    
    print("-" * 30)
    print("Summary Report")
    print("-" * 30)
    print(f"Total Rows Processed: {processed_count}")
    print(f"High Confidence: {found_high}")
    print(f"Medium Confidence: {found_medium}")
    print(f"No Match: {found_none}")
    print("-" * 30)

if __name__ == "__main__":
    main()
