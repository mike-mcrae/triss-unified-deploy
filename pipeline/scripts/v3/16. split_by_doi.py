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
OUTPUT_WITH_DOI = os.path.join(BASE_DIR, "2d.all_listed_publications_2019_journal_like_with_doi.csv")
OUTPUT_NO_DOI = os.path.join(BASE_DIR, "2d.all_listed_publications_2019_journal_like_no_doi.csv")

def main():
    print(f"Loading {INPUT_CSV}...")
    if not os.path.exists(INPUT_CSV):
        print(f"Error: File not found: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    original_count = len(df)
    
    if 'crossref_doi' not in df.columns:
        print("Error: 'crossref_doi' column not found.")
        return

    # Filter Logic
    # Consider empty strings or whitespace as NaN for this purpose
    mask_has_doi = df['crossref_doi'].notna() & (df['crossref_doi'].astype(str).str.strip() != "")
    
    df_with_doi = df[mask_has_doi].copy()
    df_no_doi = df[~mask_has_doi].copy()
    
    count_with = len(df_with_doi)
    count_no = len(df_no_doi)
    
    print(f"Original Count: {original_count}")
    print(f"Rows WITH DOI: {count_with}")
    print(f"Rows WITHOUT DOI: {count_no}")
    print(f"Sum Check: {count_with + count_no} (Should be {original_count})")
    
    if count_with + count_no != original_count:
        print("WARNING: Sum mismatch!")
        
    print(f"Saving to {OUTPUT_WITH_DOI}...")
    df_with_doi.to_csv(OUTPUT_WITH_DOI, index=False)
    
    print(f"Saving to {OUTPUT_NO_DOI}...")
    df_no_doi.to_csv(OUTPUT_NO_DOI, index=False)
    
    print("Done.")

if __name__ == "__main__":
    main()
