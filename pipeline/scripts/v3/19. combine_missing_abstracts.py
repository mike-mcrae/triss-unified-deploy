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
INPUT_WITH_DOI = os.path.join(BASE_DIR, "2d.all_listed_publications_2019_journal_like_with_doi.csv")
INPUT_NO_DOI = os.path.join(BASE_DIR, "2d.all_listed_publications_2019_journal_like_no_doi.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "2e.2d.all_listed_publications_2019_journal_like_no_abstract.csv")

def main():
    print("Loading datasets...")
    
    # 1. Load With-DOI dataset
    if os.path.exists(INPUT_WITH_DOI):
        df_doi = pd.read_csv(INPUT_WITH_DOI, low_memory=False)
        print(f"Loaded 'With DOI' file: {len(df_doi)} rows")
    else:
         print(f"Error: {INPUT_WITH_DOI} not found.")
         return

    # 2. Filter for missing abstracts
    # Definition of "present abstract": >= 50 chars and not "No abstract available"
    def has_abstract(val):
        s = str(val)
        return len(s) >= 50 and "No abstract available" not in s

    # Ensure abstract column exists
    if 'abstract' not in df_doi.columns:
        df_doi['abstract'] = ""
    
    # Filter
    mask_missing = ~df_doi['abstract'].apply(has_abstract)
    df_doi_missing = df_doi[mask_missing].copy()
    print(f"Filtered 'With DOI' missing abstracts: {len(df_doi_missing)} rows")

    # 3. Load No-DOI dataset
    if os.path.exists(INPUT_NO_DOI):
        df_no_doi = pd.read_csv(INPUT_NO_DOI, low_memory=False)
        print(f"Loaded 'No DOI' file: {len(df_no_doi)} rows")
    else:
        print(f"Error: {INPUT_NO_DOI} not found.")
        return

    # 4. Align columns
    # We want the output to have the same structure as df_doi (which has extra cols like abstract_source etc)
    # Add missing columns to df_no_doi
    for col in df_doi.columns:
        if col not in df_no_doi.columns:
            df_no_doi[col] = "" # Initialize empty
            if 'retrieved' in col or 'count' in col:
                df_no_doi[col] = 0
    
    # Ensure order matches
    df_no_doi = df_no_doi.reindex(columns=df_doi.columns, fill_value="")

    # 5. Concatenate
    df_combined = pd.concat([df_doi_missing, df_no_doi], ignore_index=True)
    
    print("-" * 30)
    print(f"Combined Rows: {len(df_combined)}")
    print(f"  - From DOI (Missing): {len(df_doi_missing)}")
    print(f"  - From No DOI: {len(df_no_doi)}")
    
    # 6. Save
    print(f"Saving to {OUTPUT_FILE}...")
    df_combined.to_csv(OUTPUT_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
