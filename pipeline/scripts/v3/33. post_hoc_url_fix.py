import pandas as pd
import os
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

# Paths
BASE_DIR = str(_PIPELINE_ROOT / "1. data")
TARGET_CSV = os.path.join(BASE_DIR, "1. raw/0. profiles/v3/2f.all_listed_publications_2019_journal_like_with_abstract.csv")
SOURCE_FINAL = os.path.join(BASE_DIR, "1. raw/0. profiles/profiles_publications_final.csv")

# Regex
REGEX_DOI = re.compile(r'\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b')
REGEX_URL = re.compile(r'\b(https?://[^\s,]+)\b')

def normalize_title(text):
    if not isinstance(text, str): return ""
    return re.sub(r'\W+', '', text.lower())

def extract_url_or_doi(row):
    """
    Scans all values in the row for a URL or DOI.
    Prioritizes URL, then DOI.
    """
    candidates_url = []
    candidates_doi = []
    
    for val in row.values:
        val_str = str(val).strip()
        if not val_str or val_str.lower() == 'nan':
            continue
            
        # Check URL
        urls = REGEX_URL.findall(val_str)
        if urls:
            candidates_url.extend(urls)
            
        # Check DOI
        dois = REGEX_DOI.findall(val_str)
        if dois:
            candidates_doi.extend(dois)
    
    # Selection Logic
    # 1. Prefer HTTP/HTTPS URL (exclude doi.org for now if we want raw URL, but doi.org is fine)
    if candidates_url:
        # Filter out some noise if needed?
        return candidates_url[0] # Take first found
        
    if candidates_doi:
        return f"https://doi.org/{candidates_doi[0]}"
        
    return None

def main():
    print(f"Loading Target: {os.path.basename(TARGET_CSV)}")
    try:
        df_target = pd.read_csv(TARGET_CSV, low_memory=False, encoding='utf-8')
    except:
        df_target = pd.read_csv(TARGET_CSV, low_memory=False, encoding='latin1')

    print(f"Loading Source: {os.path.basename(SOURCE_FINAL)}")
    try:
        df_source = pd.read_csv(SOURCE_FINAL, low_memory=False, encoding='utf-8')
    except:
        df_source = pd.read_csv(SOURCE_FINAL, low_memory=False, encoding='latin1')

    # Updated Lookup Strategy: List of (norm_ref, url) tuples for substring matching
    print("Scanning source file for potential URLs/DOIs in ALL columns...")
    
    # Store exact title matches -> URL
    exact_lookup = {}
    # Store references -> URL for substring search
    ref_candidates = [] 
    
    count_scanned = 0
    for idx, row in df_source.iterrows():
        val = extract_url_or_doi(row)
        if not val:
            continue
            
        # 1. Exact Title Match (if 'found_title' or 'title' exists)
        if 'found_title' in row and isinstance(row['found_title'], str):
             norm = normalize_title(row['found_title'])
             if norm: exact_lookup[norm] = val
             
        if 'title' in row and isinstance(row['title'], str):
             norm = normalize_title(row['title'])
             if norm: exact_lookup[norm] = val

        # 2. Publication Reference (for substring match)
        # User says title is IN publication_reference
        if 'publication_reference' in row and isinstance(row['publication_reference'], str):
             norm_ref = normalize_title(row['publication_reference'])
             if len(norm_ref) > 20: # arbitrarymin length
                 ref_candidates.append((norm_ref, val))
        
        count_scanned += 1
        
    print(f"Built exact lookup ({len(exact_lookup)} keys) and ref candidates ({len(ref_candidates)}).")
    
    # Apply to Target
    missing_mask = df_target['main_url'].isna() | (df_target['main_url'] == '')
    missing_indices = df_target[missing_mask].index
    
    print(f"Target has {len(missing_indices)} rows with missing main_url.")
    
    filled = 0
    for idx in missing_indices:
        target_title = str(df_target.at[idx, 'Title'])
        if not target_title or target_title == 'nan':
            target_title = str(df_target.at[idx, 'Publication Reference'])
            
        norm_target = normalize_title(target_title)
        if not norm_target or len(norm_target) < 10:
             continue
        
        found_url = None
        
        # 1. Check exact match
        if norm_target in exact_lookup:
            found_url = exact_lookup[norm_target]
            
        # 2. Check substring in references (if no exact match)
        if not found_url:
            for ref_norm, url in ref_candidates:
                if norm_target in ref_norm:
                    found_url = url
                    break
        
        if found_url:
            df_target.at[idx, 'main_url'] = found_url
            df_target.at[idx, 'main_url_source'] = 'post_hoc_fix'
            filled += 1
            
    print(f"Filled {filled} rows using deep scan (Exact + Substring).")
    df_target.to_csv(TARGET_CSV, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
