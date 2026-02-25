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

# Configuration
BASE_DIR = str(_PIPELINE_ROOT / "1. data/1. raw/0. profiles/v3")
INPUT_CSV = os.path.join(BASE_DIR, "2c.all_listed_publications_2019_journal_like_doi.csv")
OUTPUT_REMOVED = os.path.join(BASE_DIR, "2c.editor_appointments.csv")

# Secondary targets to filter by ID
SECONDARY_FILES = [
    os.path.join(BASE_DIR, "2c.all_listed_publications_2019_journal_like.csv"),
    os.path.join(BASE_DIR, "2c.all_listed_publications_2019_filter.csv")
]

# Column Name (Updated based on actual file)
FILTER_COL = "editorial_appointment"

def main():
    print(f"Loading {INPUT_CSV}...")
    if not os.path.exists(INPUT_CSV):
        print(f"Error: File not found: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    original_count = len(df)
    
    if FILTER_COL not in df.columns:
        print(f"Error: Column '{FILTER_COL}' not found.")
        print(f"Available columns: {list(df.columns)}")
        return

    # Filter Logic: exact match for 1 (or string "1", or float 1.0)
    # Convert to string, strip, compare to "1"
    mask_remove = df[FILTER_COL].astype(str).str.strip().str.replace('.0', '', regex=False) == "1"
    
    df_removed = df[mask_remove].copy()
    df_kept = df[~mask_remove].copy()
    
    print(f"Original Count: {original_count}")
    print(f"Rows to Remove (Flag == 1): {len(df_removed)}")
    print(f"Rows to Keep: {len(df_kept)}")
    
    if len(df_removed) == 0:
        print("No rows found with flag '1'. Check source file.")
        return

    # Save Removed
    print(f"Saving removed rows to {OUTPUT_REMOVED}...")
    df_removed.to_csv(OUTPUT_REMOVED, index=False)
    
    # Save Kept (Update Input)
    print(f"Updating {INPUT_CSV}...")
    df_kept.to_csv(INPUT_CSV, index=False)
    
    # Identify IDs to remove from other files
    remove_ids = set(df_removed['article_id'].dropna().astype(str).tolist())
    print(f"\nProceeding to filter {len(remove_ids)} IDs from secondary files...")
    
    for sec_path in SECONDARY_FILES:
        if os.path.exists(sec_path):
            print(f"Processing {os.path.basename(sec_path)}...")
            df_sec = pd.read_csv(sec_path, low_memory=False)
            
            if 'article_id' in df_sec.columns:
                mask_sec = df_sec['article_id'].astype(str).isin(remove_ids)
                df_sec_kept = df_sec[~mask_sec]
                
                print(f"  Removed: {mask_sec.sum()}")
                print(f"  Remaining: {len(df_sec_kept)}")
                
                df_sec_kept.to_csv(sec_path, index=False)
            else:
                print("  'article_id' not found, skipping.")
        else:
            print(f"  File not found: {sec_path}")
            
    print("Done.")

if __name__ == "__main__":
    main()
