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

# Configuration
BASE_DIR = str(_PIPELINE_ROOT / "1. data/1. raw/0. profiles/v3")
INPUT_FULL = os.path.join(BASE_DIR, "0.all_listed_pubs_fully_enriched.csv")
OUTPUT_FULL_FILTERED = os.path.join(BASE_DIR, "0c.all_listed_publications_filter.csv")

INPUT_2019 = os.path.join(BASE_DIR, "2.all_listed_pubs_2019.csv")
OUTPUT_2019_FILTERED = os.path.join(BASE_DIR, "2c.all_listed_publications_2019_filter.csv")

TARGET_TYPES = [
    "Book", 
    "Book Chapter", 
    "Journal", 
    "Journal Article", 
    "Working Paper"
]

def filter_csv(input_path, output_path):
    print(f"Processing {input_path}...")
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        return

    df = pd.read_csv(input_path, low_memory=False)
    original_count = len(df)
    
    if 'Publication Type' not in df.columns:
        print("Error: 'Publication Type' column missing.")
        return

    # Filter
    # Use isin for exact match. 
    # Note: "Journal" might not exist if "Journal Article" is used, but we include it just in case.
    df_filtered = df[df['Publication Type'].isin(TARGET_TYPES)].copy()
    filtered_count = len(df_filtered)
    
    print(f"Original: {original_count} -> Filtered: {filtered_count}")
    print(f"Kept Types: {df_filtered['Publication Type'].unique()}")
    
    # Save
    print(f"Saving to {output_path}...")
    df_filtered.to_csv(output_path, index=False)
    print("Saved.")

def main():
    print("Starting Publication Type Filtering...")
    print(f"Target Types: {TARGET_TYPES}")
    
    # 1. Full Dataset
    filter_csv(INPUT_FULL, OUTPUT_FULL_FILTERED)
    
    print("-" * 20)
    
    # 2. 2019+ Dataset
    filter_csv(INPUT_2019, OUTPUT_2019_FILTERED)
    
    print("Done.")

if __name__ == "__main__":
    main()
