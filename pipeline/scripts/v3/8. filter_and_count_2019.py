import pandas as pd
import os
import sys
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

# Files
BASE_DIR = str(_PIPELINE_ROOT / "1. data/1. raw/0. profiles/v3")
INPUT_CSV = os.path.join(BASE_DIR, "0.all_listed_pubs_fully_enriched.csv")
OUTPUT_CSV_2019 = os.path.join(BASE_DIR, "2.all_listed_pubs_2019.csv")
OUTPUT_COUNTS = os.path.join(BASE_DIR, "2b.publication_types_totals_2019.csv")

def main():
    print(f"Loading {INPUT_CSV}...")
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input file not found: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"Total records: {len(df)}")
    
    # Clean Year Column
    # 'Year Descending' might contain strings, ranges (2019-2020), or specialized terms (In Press)
    # We need to coerce to numeric 
    
    # 1. Force numeric, coerce errors to NaN
    df['year_clean'] = pd.to_numeric(df['Year Descending'], errors='coerce')
    
    # 2. Filter >= 2019
    # Note: What about "In Press" or "Submitted"? Usually they don't have a year or have a future year? 
    # If they failed numeric conversion, they are NaNs.
    # Let's inspect what we are dropping.
    
    # Strategy: Keep if year >= 2019. Drop NaNs? 
    # If a paper is "In Press" (no year), strictly it's not "2019 and sooner" (meaning later). 
    # But usually In Press is "2024+" effectively.
    # However, strict numeric filter is safest for now.
    
    df_2019 = df[df['year_clean'] >= 2019].copy()
    
    # Drop temp column
    df_2019.drop(columns=['year_clean'], inplace=True)
    
    print(f"Filtered records (>= 2019): {len(df_2019)}")
    
    # Save Filtered Dataset
    print(f"Saving to {OUTPUT_CSV_2019}...")
    df_2019.to_csv(OUTPUT_CSV_2019, index=False)
    
    # Count Publication Types
    print("Counting publication types...")
    if 'Publication Type' not in df_2019.columns:
        print("Warning: 'Publication Type' column missing.")
        return

    counts = df_2019['Publication Type'].value_counts().reset_index()
    counts.columns = ['Publication Type', 'Count']
    
    # Save Counts
    print(f"Saving counts to {OUTPUT_COUNTS}...")
    counts.to_csv(OUTPUT_COUNTS, index=False)
    
    print("-" * 30)
    print("Top Publication Types (2019+):")
    print(counts.head(10))
    print("-" * 30)
    print("Done.")

if __name__ == "__main__":
    main()
