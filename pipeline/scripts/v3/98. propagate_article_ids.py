import pandas as pd
import os
import difflib
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
# Configuration
BASE_DIR = str(_PIPELINE_ROOT / "1. data/1. raw/0. profiles/v3")
RAW_MASTER_CSV = os.path.join(BASE_DIR, "0.all_listed_pubs.csv")
ENRICHED_MASTER_CSV = os.path.join(BASE_DIR, "0.all_listed_pubs_fully_enriched.csv")

# List of target files to update (Enriched master is also a target effectively, but handled specially)
TARGET_FILES = [
    "0c.all_listed_publications_filter.csv",
    "2.all_listed_pubs_2019.csv",
    "2c.all_listed_publications_2019_filter.csv",
    "2c.all_listed_publications_2019_journal_like.csv",
    "2c.all_listed_publications_2019_journal_like_doi.csv",
    "2c.all_listed_publications_2019_books_other.csv"
]

def normalize_text(s):
    if not isinstance(s, str):
        return ""
    return " ".join(s.lower().split())

def main():
    print(f"Loading Raw Master (IDs): {RAW_MASTER_CSV}...")
    try:
        df_raw = pd.read_csv(RAW_MASTER_CSV, low_memory=False)
    except Exception as e:
        print(f"Error loading raw master CSV: {e}")
        return

    print(f"Loading Enriched Master (Titles): {ENRICHED_MASTER_CSV}...")
    try:
        df_enriched = pd.read_csv(ENRICHED_MASTER_CSV, low_memory=False)
    except Exception as e:
        print(f"Error loading enriched master CSV: {e}")
        return

    # 1. Merge ID from Raw to Enriched by Index
    print("Merging article_id to Enriched Master by Index...")
    if len(df_raw) != len(df_enriched):
        print(f"WARNING: Row count mismatch! Raw={len(df_raw)}, Enriched={len(df_enriched)}")
        # If mismatch is small, maybe we can still proceed if alignment is mostly correct?
        # But safest is to abort if major mismatch. 
        # Given our debug, they match (24567).
        if abs(len(df_raw) - len(df_enriched)) > 5:
            print("Aborting due to significant mismtatch.")
            return

    # Copy ID
    df_enriched['article_id'] = df_raw['article_id']

    # Move column to start
    cols = list(df_enriched.columns)
    if 'article_id' in cols:
        cols.remove('article_id')
        cols.insert(7, 'article_id') # After email? n_id=0..email=6..
    df_enriched = df_enriched[cols]
    
    # Save Enriched with ID
    print(f"Saving updated {ENRICHED_MASTER_CSV}...")
    df_enriched.to_csv(ENRICHED_MASTER_CSV, index=False)
    
    # 2. Build Lookup from Enriched (which now has IDs AND Enriched Titles)
    lookup = {}
    print("Building lookup index from Enriched Master...")
    
    for idx, row in df_enriched.iterrows():
        n_id = str(row.get('n_id', '')).strip()
        title = normalize_text(row.get('Title', ''))
        year = str(row.get('Year Descending', '')).strip()
        
        # Clean year
        try:
            year = str(int(float(year)))
        except:
            pass
            
        key = (n_id, title, year)
        
        article_id = row.get('article_id')
        if not article_id:
            continue
            
        if key not in lookup:
            lookup[key] = []
        lookup[key].append(article_id)

    print(f"Lookup index built with {len(lookup)} unique keys.")

    # 3. Process Downstream Targets
    for filename in TARGET_FILES:
        filepath = os.path.join(BASE_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Skipping {filename} (not found)")
            continue
            
        print(f"Processing {filename}...")
        try:
            df_target = pd.read_csv(filepath, low_memory=False)
        except Exception as e:
            print(f"  Error reading {filename}: {e}")
            continue
            
        # Add article_id column if missing
        if 'article_id' not in df_target.columns:
            cols = list(df_target.columns)
            if 'n_id' in cols:
                idx_nid = cols.index('n_id')
                cols.insert(idx_nid + 1, 'article_id')
            else:
                cols.insert(0, 'article_id')
                
            df_target['article_id'] = ""
            # We enforce order at save time
        
        # Iterate and Assign
        matched_count = 0
        key_usage = {} 
        
        # Ensure 'article_id' is object type
        df_target['article_id'] = df_target['article_id'].astype('object')

        for idx, row in df_target.iterrows():
            # If already has ID? The user implies we shoud set it.
            
            n_id = str(row.get('n_id', '')).strip()
            title = normalize_text(row.get('Title', ''))
            year = str(row.get('Year Descending', '')).strip()
            try:
                year = str(int(float(year)))
            except:
                pass
            
            key = (n_id, title, year)
            
            candidates = lookup.get(key)
            
            if candidates:
                usage = key_usage.get(key, 0)
                if usage < len(candidates):
                    assigned_id = candidates[usage]
                    key_usage[key] = usage + 1
                    df_target.at[idx, 'article_id'] = assigned_id
                    matched_count += 1
            else:
                # Debug failed matches?
                pass
                
        # Save
        cols = list(df_target.columns)
        if 'article_id' in cols:
            cols.remove('article_id')
            if 'n_id' in cols:
                idx_nid = cols.index('n_id')
                cols.insert(idx_nid + 1, 'article_id')
            else:
                cols.insert(0, 'article_id')
        
        df_target = df_target[cols]
        df_target.to_csv(filepath, index=False)
        print(f"  Saved {filename}. Matched {matched_count}/{len(df_target)} rows.")

    print("Propagation Complete.")

if __name__ == "__main__":
    main()
