import pandas as pd
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

# Paths
BASE_DIR = str(_PIPELINE_ROOT / "1. data/1. raw/0. profiles/v3")
FILE_2D = os.path.join(BASE_DIR, "2d.all_listed_publications_2019_journal_like_with_doi.csv")
FILE_2E = os.path.join(BASE_DIR, "2e.all_listed_publications_2019_journal_like_no_abstract copy.csv")
FILE_2F = os.path.join(BASE_DIR, "2f.all_listed_publications_2019_journal_like_with_abstract.csv")

def main():
    print("--- 1. Loading 2d (Original) ---")
    try:
        df_2d = pd.read_csv(FILE_2D, low_memory=False, encoding='utf-8')
    except UnicodeDecodeError:
        df_2d = pd.read_csv(FILE_2D, low_memory=False, encoding='latin1')
    
    initial_2d = len(df_2d)
    print(f"Loaded {initial_2d} rows from 2d.")
    
    # Filter 2d: Keep only rows with valid abstracts
    # Criteria: abstract is not null/empty and length > 20 (to align with previous logic)
    df_2d['abstract'] = df_2d['abstract'].astype(str).replace('nan', '')
    df_2d_filtered = df_2d[df_2d['abstract'].str.len() > 20].copy()
    
    kept_2d = len(df_2d_filtered)
    dropped_2d = initial_2d - kept_2d
    print(f"Kept {kept_2d} rows with existing abstracts.")
    print(f"Dropped {dropped_2d} rows (missing/short abstracts).")
    
    print("\n--- 2. Loading 2e (Recovered) ---")
    try:
        df_2e = pd.read_csv(FILE_2E, low_memory=False, encoding='utf-8')
    except UnicodeDecodeError:
        df_2e = pd.read_csv(FILE_2E, low_memory=False, encoding='latin1')
        
    initial_2e = len(df_2e)
    print(f"Loaded {initial_2e} rows from 2e (Recovered/Scraped).")
    
    print("\n--- 3. Merging ---")
    df_final = pd.concat([df_2d_filtered, df_2e], ignore_index=True)
    
    total_final = len(df_final)
    print(f"Final Count: {total_final} rows ({kept_2d} + {initial_2e}).")
    
    print(f"\n--- 4. Saving to {os.path.basename(FILE_2F)} ---")
    df_final.to_csv(FILE_2F, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
