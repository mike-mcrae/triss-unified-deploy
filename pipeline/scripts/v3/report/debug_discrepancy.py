import pandas as pd
from pathlib import Path
import os

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

BASE = _PIPELINE_ROOT
FILE_LISTED = BASE / "1. data/3. final/3. All listed publications 2019 +.csv"
FILE_MEASURED = BASE / "1. data/3. final/4.All measured publications.csv"

listed = pd.read_csv(FILE_LISTED)
measured = pd.read_csv(FILE_MEASURED)

def map_pub_type(t):
    t = str(t).lower()
    if "journal" in t:
        return "Journal Article"
    elif "working paper" in t:
        return "Working Paper"
    elif "book" in t:
        return "Book/Chapter"
    else:
        return "Other"

listed['Collapsed Type'] = listed['Publication Type'].apply(map_pub_type)
measured['Collapsed Type'] = measured['Publication Type'].apply(map_pub_type)

# Core types in listed
core_listed = listed[listed['Collapsed Type'] != 'Other']
print(f"Total Core Types in Listed: {len(core_listed)}")

# Core types in measured
print(f"Total Core Types in Measured: {len(measured)}")
print(f"Total Measured Abstracts Analysed: {measured['abstract'].notna().sum() if 'abstract' in measured.columns else 0}")
print(f"Total Measured WITH NO Abstract: {measured['abstract'].isna().sum() if 'abstract' in measured.columns else 0}")

# Dropped core types
dropped_core = core_listed[~core_listed['article_id'].isin(measured['article_id'])]
print(f"\nTotal Core Types Dropped: {len(dropped_core)}")

# Why were they dropped?
# Let's check duplicates in listed?
print(f"Are there duplicate article_ids in Listed core? {core_listed['article_id'].duplicated().sum()}")

# Let's check if there's a file that shows the dropping process (like 2a or 2b scripts of pipeline)
# Just show a few dropped items and see if they look weird
print("\nSample of dropped core items:")
print(dropped_core[['article_id', 'Publication Type', 'Title']].head(10))

