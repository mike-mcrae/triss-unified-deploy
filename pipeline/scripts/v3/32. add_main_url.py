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
TARGET_CSV = str(_PIPELINE_ROOT / "1. data/1. raw/0. profiles/v3/2f.all_listed_publications_2019_journal_like_with_abstract.csv")
SOURCE_FINAL = str(_PIPELINE_ROOT / "1. data/1. raw/0. profiles/profiles_publications_final.csv")
SOURCE_FULL = str(_SHARED_DATA_ROOT / "3. final/3. profiles_publications_full_merged.csv")

def normalize_title(text):
    if not isinstance(text, str): return ""
    return re.sub(r'\W+', '', text.lower())

def clean_url(url):
    if not isinstance(url, str): return None
    url = url.strip()
    if not url or url.lower() in ['-', 'nan', 'none', '']: return None
    if not url.startswith('http'):
        # Valid guess for www?
        if url.startswith('www'): return 'https://' + url
        # If it looks like a DOI, return None (handled elsewhere) or fix it?
        # For now, simplistic check
        return None 
    return url

def main():
    print(f"Loading Target: {os.path.basename(TARGET_CSV)}")
    try:
        df = pd.read_csv(TARGET_CSV, low_memory=False, encoding='utf-8')
    except:
        df = pd.read_csv(TARGET_CSV, low_memory=False, encoding='latin1')

    print("Loading Source 1: profiles_publications_final.csv")
    try:
        df_src1 = pd.read_csv(SOURCE_FINAL, low_memory=False, encoding='utf-8')
    except:
        df_src1 = pd.read_csv(SOURCE_FINAL, low_memory=False, encoding='latin1')
        
    print("Loading Source 2: profiles_publications_full_merged.csv")
    try:
        df_src2 = pd.read_csv(SOURCE_FULL, low_memory=False, encoding='utf-8')
    except:
        df_src2 = pd.read_csv(SOURCE_FULL, low_memory=False, encoding='latin1')
        
    # Build Lookups (Normalized Title -> URL)
    # Prefer df_src2 (Full Merged) then df_src1? Or combine?
    # Logic says "check final, AND full_merged". I'll build a combined lookup.
    
    url_lookup = {}
    
    # Process Source 2 first (maybe fresher?), then Source 1 to fill gaps? 
    # Or Source 1 first? The user listed Final then Full Merged.
    # I'll process in reverse order of preference if I want to overwrite, or check if key exists.
    # Let's iterate both.
    
    sources = [
        (df_src1, 'URL'), # Check header name from previous step
        (df_src2, 'URL')  # Check header name from previous step
    ]
    
    print("Building URL Lookup Table...")
    for src_df, url_col in sources:
        if url_col not in src_df.columns:
            print(f"Warning: Column {url_col} not found in source.")
            continue
            
        for _, row in src_df.iterrows():
            title = row.get('title', '') # Assuming 'title' exists
            if not isinstance(title, str): # Try 'Title'
                 title = row.get('Title', '')
            
            url = row.get(url_col, '')
            
            norm_title = normalize_title(str(title))
            clean = clean_url(str(url))
            
            if norm_title and clean:
                url_lookup[norm_title] = clean

    print(f"Lookup Table Built: {len(url_lookup)} titles with URLs.")

    # Apply Logic
    # 1. crossref_doi -> https://doi.org/...
    # 2. Tara Handle -> http://hdl.handle.net/... (if not http)
    # 3. Url
    # 4. url_found
    # 5. Lookup Title
    
    regex_handle = re.compile(r'^\d+/\d+$') # Simple handle format check "1234/5678"

    def get_main_url(row):
        # 1. DOI
        doi = str(row.get('crossref_doi', '')).strip()
        if doi and doi.lower() not in ['nan', '-', '']:
            return f"https://doi.org/{doi}"
            
        # 2. Tara Handle
        tara = str(row.get('Tara Handle', '')).strip()
        if tara and tara.lower() not in ['nan', '-', '']:
            if tara.startswith('http'): return tara
            if regex_handle.match(tara): return f"http://hdl.handle.net/{tara}"
            # If it's just a string but not a standard handle format, doubtful...
            # But let's assume if it is in that column, it is a handle.
            return f"http://hdl.handle.net/{tara}"

        # 3. Url
        url = str(row.get('Url', '')).strip()
        if url and url.lower() not in ['nan', '-', '']:
            # Basic cleanup
            if not url.startswith('http'):
                 if url.startswith('www'): return 'https://' + url
                 return None # Skip invalid
            return url
            
        # 4. url_found
        found = str(row.get('url_found', '')).strip()
        if found and found.lower() not in ['nan', '-', '']:
             if not found.startswith('http'): return None
             return found
             
        # 5. Lookup
        title = str(row.get('Title', '')).strip()
        if title:
            norm = normalize_title(title)
            if norm in url_lookup:
                return url_lookup[norm]
                
        return None

    print("Calculating main_url...")
    df['main_url'] = df.apply(get_main_url, axis=1)
    
    filled_count = df['main_url'].notna().sum()
    print(f"Filled main_url for {filled_count} / {len(df)} rows.")
    
    df.to_csv(TARGET_CSV, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
