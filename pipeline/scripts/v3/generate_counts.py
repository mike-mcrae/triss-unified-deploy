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

BASE_DIR = str(_PIPELINE_ROOT / "1. data/3. Final")

FILE_1 = f"{BASE_DIR}/2. All listed publications.csv"
FILE_2 = f"{BASE_DIR}/3. All listed publications 2019 +.csv"
FILE_3 = f"{BASE_DIR}/4.All measured publications.csv"

def safe_read_csv(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        return pd.read_csv(f, low_memory=False)

def generate_counts():
    print("Reading CSVs...")
    df1 = safe_read_csv(FILE_1)
    df2 = safe_read_csv(FILE_2)
    df3 = safe_read_csv(FILE_3)

    print("Generating counts for File 1 (All listed publications)...")
    keys1 = ['department', 'school', 'n_id', 'Publication Type', 'sample_articles_chosen']
    # Fill NA for grouping
    df1[keys1] = df1[keys1].fillna("Unknown")
    counts1 = df1.groupby(keys1).size().reset_index(name='count')
    out1 = f"{BASE_DIR}/counts_2_all_listed.json"
    counts1.to_json(out1, orient='records', indent=4)
    print(f"Saved: {out1}")

    print("Generating counts for File 2 (All listed 2019+)...")
    keys2 = ['department', 'school', 'n_id', 'Publication Type', 'sample_articles_chosen']
    df2[keys2] = df2[keys2].fillna("Unknown")
    counts2 = df2.groupby(keys2).size().reset_index(name='count')
    out2 = f"{BASE_DIR}/counts_3_all_listed_2019_plus.json"
    counts2.to_json(out2, orient='records', indent=4)
    print(f"Saved: {out2}")

    print("Generating counts for File 3 (All measured publications)...")
    keys3 = ['department', 'school', 'n_id', 'Publication Type']
    df3[keys3] = df3[keys3].fillna("Unknown")
    counts3 = df3.groupby(keys3).size().reset_index(name='count')
    out3 = f"{BASE_DIR}/counts_4_all_measured.json"
    counts3.to_json(out3, orient='records', indent=4)
    print(f"Saved: {out3}")

if __name__ == "__main__":
    generate_counts()
