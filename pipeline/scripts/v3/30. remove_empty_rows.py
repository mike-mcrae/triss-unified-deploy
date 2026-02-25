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

TARGET_CSV = str(_PIPELINE_ROOT / "1. data/1. raw/0. profiles/v3/2e.all_listed_publications_2019_journal_like_no_abstract copy.csv")

def main():
    print(f"Loading {TARGET_CSV}...")
    try:
        df = pd.read_csv(TARGET_CSV, low_memory=False, encoding='utf-8')
    except UnicodeDecodeError:
        print("UTF-8 failed, trying latin1...")
        df = pd.read_csv(TARGET_CSV, low_memory=False, encoding='latin1')
    
    initial_count = len(df)
    
    # Drop rows where ALL columns are NaN
    df_cleaned = df.dropna(how='all')
    
    # Also drop rows where all columns are either NaN or empty strings (if read as object)
    # This is a bit more aggressive: clean up whitespace and check if row is empty
    # But strictly "completely empty" usually means all NaNs in pandas after read_csv
    
    final_count = len(df_cleaned)
    removed_count = initial_count - final_count
    
    if removed_count > 0:
        print(f"Removed {removed_count} empty rows.")
        df_cleaned.to_csv(TARGET_CSV, index=False)
        print("File saved.")
    else:
        print("No empty rows found.")

if __name__ == "__main__":
    main()
