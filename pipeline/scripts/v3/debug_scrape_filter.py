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

BASE_DIR = str(_PIPELINE_ROOT / "1. data/1. raw/0. profiles/v3")
INPUT_CSV = os.path.join(BASE_DIR, "2e.all_listed_publications_2019_journal_like_no_abstract.csv")

def main():
    print(f"Loading {INPUT_CSV}...")
    if not os.path.exists(INPUT_CSV):
        print("File not found.")
        return

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    
    print("-" * 30)
    print("Columns:", df.columns.tolist())
    
    if 'scraped_status' in df.columns:
        print("\nscraped_status unique values:", df['scraped_status'].unique())
        print("scraped_status value counts summary:")
        print(df['scraped_status'].value_counts(dropna=False))
        
        # Test my filter logic
        # mask_process = ~df['abstract'].apply(has_abstract) & (df['scraped_status'] == "")
        
        # Check nulls vs empty strings
        nulls = df['scraped_status'].isna().sum()
        empty_strings = (df['scraped_status'] == "").sum()
        print(f"\nNulls: {nulls}")
        print(f"Empty Strings: {empty_strings}")
        
    else:
        print("\n'scraped_status' column missing.")

    if 'abstract' in df.columns:
        print("\nabstract sample (head):")
        print(df[['abstract']].head())

if __name__ == "__main__":
    main()
