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
INPUT_CSV = os.path.join(BASE_DIR, "2d.all_listed_publications_2019_journal_like_with_doi.csv")

def main():
    print(f"Loading {INPUT_CSV}...")
    if not os.path.exists(INPUT_CSV):
        print("File not found.")
        return

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    
    print("-" * 30)
    print(f"Total Rows: {len(df)}")
    
    # helper for safe len
    def safe_len(x):
        try:
            return len(str(x)) if pd.notna(x) else 0
        except:
            return 0

    # Abstract Counts
    has_abstract = df['abstract'].notna() & (df['abstract'].apply(safe_len) >= 50)
    count_has_abstract = has_abstract.sum()
    print(f"Rows with Abstract (>=50 chars): {count_has_abstract} ({count_has_abstract/len(df):.1%})")
    print(f"Rows Missing Abstract: {len(df) - count_has_abstract}")
    
    print("-" * 30)
    print("Source Distribution:")
    if 'abstract_source' in df.columns:
        print(df['abstract_source'].value_counts(dropna=False))
    else:
        print("Column 'abstract_source' not found.")

    print("-" * 30)
    print("HTTP Status Code Distribution (for failures?):")
    if 'http_status_code' in df.columns:
         # Maybe group by retrieved status?
         print(df['http_status_code'].value_counts(dropna=False).head(10))
    else:
        print("Column 'http_status_code' not found.")

    print("-" * 30)
    # Check if 'abstract_retrieved' matches reality
    if 'abstract_retrieved' in df.columns:
        retrieved_flag = df['abstract_retrieved'] == 1
        mismatch = retrieved_flag & (~has_abstract)
        if mismatch.sum() > 0:
            print(f"WARNING: {mismatch.sum()} rows marked as retrieved but have no/short abstract.")
        else:
            print("Verification: 'abstract_retrieved' flag seems consistent with content.")
            
if __name__ == "__main__":
    main()
