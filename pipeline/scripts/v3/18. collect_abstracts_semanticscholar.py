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

session = requests.Session()
adapter = requests.adapters.HTTPAdapter(max_retries=1)
session.mount('https://', adapter)
session.headers.update({'User-Agent': f'TRISS-Pipeline/1.0 (mailto:{MAILTO})'})

def fetch_with_retry(doi):
    """
    Fetch abstract from Semantic Scholar Graph API with robust rate limiting.
    """
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=abstract,title,year"
    max_retries = 3
    base_backoff = 10 # Start with 10s for 429
    
    for attempt in range(max_retries + 1):
        try:
            # Enforce baseline rate limit (1 req/sec)
            # We sleep BEFORE request to be safe, especially in retries
            time.sleep(1.5) 
            
            r = session.get(url, timeout=15)
            status = r.status_code
            
            if status == 200:
                return r.json(), status
            
            elif status == 429:
                sleep_time = base_backoff * (attempt + 1)
                print(f"    ! 429 Too Many Requests. Backing off {sleep_time}s...")
                time.sleep(sleep_time)
                continue # Retry
                
            elif status >= 500:
                print(f"    ! {status} Server Error. Retrying in 5s...")
                time.sleep(5)
                continue
                
            elif status == 404:
                return None, status
            
            else:
                # Other errors
                return None, status

        except Exception as e:
            print(f"    ! Exception: {e}")
            time.sleep(2)
    
    return None, 429 # Assume 429 if we ran out of retries on it

def main():
    print(f"Loading {INPUT_CSV}...")
    if not os.path.exists(INPUT_CSV):
        print("File not found.")
        return

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    
    if 'abstract_source' not in df.columns:
        df['abstract_source'] = ""
    df['abstract'] = df['abstract'].fillna("").astype(str)
    
    def has_abstract(val):
        return len(str(val)) >= 50 and "No abstract available" not in str(val)

    # Identical logic to previous run: Find rows with DOI but missing abstract
    doi_col = 'crossref_doi'
    mask_has_doi = df[doi_col].notna() & (df[doi_col].str.strip() != "")
    mask_no_abstract = ~df['abstract'].apply(has_abstract)
    
    indices_to_process = df[mask_has_doi & mask_no_abstract].index.tolist()
    total_to_process = len(indices_to_process)
    
    print(f"Found {total_to_process} rows to process (Targeting missing abstracts).")
    
    processed_count = 0
    save_counter = 0

    try:
        for idx in indices_to_process:
            doi = str(df.loc[idx, doi_col]).strip()
            clean_doi = doi.lower().replace('https://doi.org/', '').replace('doi.org/', '').replace('http://doi.org/', '')
            
            data, status = fetch_with_retry(clean_doi)
            
            # Update status code
            df.at[idx, 'http_status_code'] = status
            
            found = False
            if status == 200 and data:
                abstract = data.get('abstract', '')
                if abstract and len(abstract) >= 50:
                    df.at[idx, 'abstract'] = abstract.strip()
                    df.at[idx, 'abstract_source'] = 'semanticscholar'
                    df.at[idx, 'abstract_retrieved'] = 1
                    df.at[idx, 'abstract_char_count'] = len(abstract)
                    print(f"[{processed_count+1}/{total_to_process}] Found: {clean_doi}")
                    found = True
            
            if not found:
                 # Ensure we mark it as attempted
                 if pd.isna(df.at[idx, 'abstract_retrieved']):
                     df.at[idx, 'abstract_retrieved'] = 0
                 # Log if it was a failure vs just not found
                 if status == 200:
                     print(f"[{processed_count+1}/{total_to_process}] No abstract data: {clean_doi}")
                 else:
                     print(f"[{processed_count+1}/{total_to_process}] Failed: {clean_doi} [Status: {status}]")

            processed_count += 1
            save_counter += 1
            
            if save_counter >= BATCH_SIZE:
                print(f"Saving batch ({processed_count} processed)...")
                df.to_csv(INPUT_CSV, index=False)
                save_counter = 0

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        print("Final save...")
        df.to_csv(INPUT_CSV, index=False)
        print("Done.")

if __name__ == "__main__":
    main()
