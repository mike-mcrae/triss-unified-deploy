import pandas as pd
import requests
import time
import os
import re
import json
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
INPUT_CSV = os.path.join(BASE_DIR, "2d.all_listed_publications_2019_journal_like_with_doi.csv")
BATCH_SIZE = 50
MAILTO = "mikemcrae@tcd.ie" 

# Setup Session
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(max_retries=1)
session.mount('https://', adapter)
session.headers.update({'User-Agent': f'TRISS-Pipeline/1.0 (mailto:{MAILTO})'})

def clean_xml_tags(text):
    """Remove XML/HTML tags from text."""
    if not text:
        return ""
    clean = re.sub(r'<[^>]+>', '', text)
    return clean.strip()

def reconstruct_openalex_abstract(inverted_index):
    """Reconstruct abstract from OpenAlex inverted index."""
    if not inverted_index:
        return ""
    word_index = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_index.append((pos, word))
    word_index.sort(key=lambda x: x[0])
    return " ".join([w[1] for w in word_index])

def fetch_with_retry(url):
    """
    Fetch URL with robust retry logic for 429 and 500 errors.
    Returns: (status_code, json_data_or_None)
    """
    max_retries = 3
    backoff_429 = 5  # Seconds to wait on 429
    backoff_500 = 2  # Seconds to wait on 500
    
    for attempt in range(max_retries + 1):
        try:
            r = session.get(url, timeout=15)
            status = r.status_code
            
            if status == 200:
                return status, r.json()
            elif status == 404:
                return status, None
            elif status == 429:
                print(f"    ! 429 Too Many Requests. Backing off {backoff_429}s...")
                time.sleep(backoff_429)
                # Linear backoff increase? User said "backs off for a few seconds". 5s constitutes "a few".
                # Continue loop to retry
            elif status >= 500:
                print(f"    ! {status} Server Error. Retrying in {backoff_500}s...")
                time.sleep(backoff_500)
                # Continue loop to retry
            else:
                # Other errors (403, 400, etc.) - probably fatal for this DOI
                return status, None
                
        except Exception as e:
            print(f"    ! Exception: {e}")
            # Network errors might be transient
            time.sleep(1)
            if attempt == max_retries:
                 # If we can't even get a status code, return 0 or something distinctive
                 return 0, None

    # If we exhausted retries (for 429/500), return the last status seen
    # If the last attempt was 429/500, we just return it.
    return status, None


def get_crossref_abstract(doi):
    """Fetch abstract from Crossref. Returns (abstract, status_code)."""
    url = f"https://api.crossref.org/works/{doi}"
    status, data = fetch_with_retry(url)
    
    if status == 200 and data:
        item = data.get('message', {})
        abstract = item.get('abstract', '')
        if abstract:
            return clean_xml_tags(abstract), status
    
    return None, status

def get_openalex_abstract(doi):
    """Fetch abstract from OpenAlex using DOI. Returns (abstract, status_code)."""
    url = f"https://api.openalex.org/works?filter=doi:{doi}&mailto={MAILTO}"
    status, data = fetch_with_retry(url)
    
    if status == 200 and data:
        results = data.get('results', [])
        if results:
            item = results[0]
            inverted_index = item.get('abstract_inverted_index')
            if inverted_index:
                return reconstruct_openalex_abstract(inverted_index), status
    
    return None, status

def main():
    print(f"Loading {INPUT_CSV}...")
    if not os.path.exists(INPUT_CSV):
        print("File not found.")
        return

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    
    # Initialize/Ensure columns exist
    new_cols = ['abstract', 'abstract_source', 'abstract_retrieved', 'abstract_char_count', 'http_status_code']
    for col in new_cols:
        if col not in df.columns:
            df[col] = ""
            if col == 'abstract_retrieved' or col == 'abstract_char_count' or col == 'http_status_code':
                df[col] = 0

    # Ensure correct dtypes
    df['abstract_retrieved'] = df['abstract_retrieved'].fillna(0).astype(int)
    
    # Identify rows to process: 
    # Logic: Process ANY row where 'abstract' is empty or short (<50 chars).
    # Regardless of previous 'abstract_retrieved' status (re-run failures).
    
    doi_col = 'crossref_doi'
    
    mask_to_process = (
        (df[doi_col].notna()) & 
        (df[doi_col].str.strip() != "") & 
        ((df['abstract'].isna()) | (df['abstract'].str.len() < 50))
    )
    
    indices_to_process = df[mask_to_process].index.tolist()
    total_to_process = len(indices_to_process)
    
    print(f"Found {total_to_process} rows to process (Targeting missing abstracts).")
    
    processed_count = 0
    save_counter = 0

    try:
        for idx in indices_to_process:
            doi = str(df.loc[idx, doi_col]).strip()
            # Normalize DOI
            clean_doi = doi.lower().replace('https://doi.org/', '').replace('doi.org/', '')
            
            # 1. Try Crossref
            abstract, status = get_crossref_abstract(clean_doi)
            source = 'none'
            final_status = status
            
            if abstract and len(abstract) >= 50:
                source = 'crossref'
            else:
                # 2. Fallback to OpenAlex
                # Check status code from Crossref? If 404, OpenAlex might have it.
                # If 500/429 persisted, we might skip logic?
                # User says: "If a 500 is thrown then edit the request accordingly" -> we did retries.
                
                time.sleep(0.2) 
                abstract_oa, status_oa = get_openalex_abstract(clean_doi)
                # Update final status to OpenAlex's status if we tried it
                final_status = status_oa 
                
                if abstract_oa and len(abstract_oa) >= 50:
                    abstract = abstract_oa
                    source = 'openalex'
            
            # Update DataFrame
            df.at[idx, 'http_status_code'] = final_status
            
            if source != 'none' and abstract:
                df.at[idx, 'abstract'] = abstract
                df.at[idx, 'abstract_source'] = source
                df.at[idx, 'abstract_retrieved'] = 1
                df.at[idx, 'abstract_char_count'] = len(abstract)
                print(f"[{processed_count+1}/{total_to_process}] Found: {clean_doi} ({source}) [Status: {final_status}]")
            else:
                # User: "make these 0 if there is no abstract retrieved and continue this pattern moving forward"
                df.at[idx, 'abstract_retrieved'] = 0 
                # Keep source as '' or 'none'? User didn't specify, but implies resetting. 
                # Actually, if we set retrieved=0, it will be picked up next time. 
                # Storing 'none' in source is good for visibility.
                if pd.isna(df.at[idx, 'abstract_source']) or df.at[idx, 'abstract_source'] == "":
                     df.at[idx, 'abstract_source'] = 'none'
                     
                print(f"[{processed_count+1}/{total_to_process}] Failed: {clean_doi} [Status: {final_status}]")

            processed_count += 1
            save_counter += 1
            
            # Incremental Save
            if save_counter >= BATCH_SIZE:
                print(f"Saving batch ({processed_count} processed)...")
                df.to_csv(INPUT_CSV, index=False)
                save_counter = 0
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        print("Final save...")
        df.to_csv(INPUT_CSV, index=False)
        print("Done.")

if __name__ == "__main__":
    main()
