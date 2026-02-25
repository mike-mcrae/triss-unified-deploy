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

# Files
BASE_DIR = str(_PIPELINE_ROOT / "1. data/1. raw/0. profiles/v3")
INPUT_CSV = os.path.join(BASE_DIR, "0.all_listed_pubs.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "1.publication_types_totals.csv")

def main():
    print(f"Loading {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV, low_memory=False)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    if 'Publication Type' not in df.columns:
        print("Error: 'Publication Type' column not found.")
        return

    # Count values
    counts = df['Publication Type'].value_counts().reset_index()
    counts.columns = ['Publication Type', 'Count']
    
    print("-" * 30)
    print("Publication Type Totals:")
    print(counts)
    print("-" * 30)

    # Save
    print(f"Saving to {OUTPUT_CSV}...")
    counts.to_csv(OUTPUT_CSV, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
