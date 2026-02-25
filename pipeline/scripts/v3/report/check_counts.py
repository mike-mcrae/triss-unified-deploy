#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

print(f"Total Listed (2019+): {len(listed)}")
print(f"Total Measured (2019+): {len(measured)}")

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

print("\n--- Types in Listed (2019+) ---")
print(listed['Collapsed Type'].value_counts())

print("\n--- Types in Measured (2019+) ---")
print(measured['Collapsed Type'].value_counts())

# Identify dropped
dropped = listed[~listed['article_id'].isin(measured['article_id'])]
print(f"\nTotal Dropped: {len(dropped)}")
print("\n--- Types of Dropped Publications ---")
print(dropped['Collapsed Type'].value_counts())

# Show a few examples of dropped journal articles
dropped_journals = dropped[dropped['Collapsed Type'] == 'Journal Article']
print(f"\nExamples of dropped Journal Articles ({len(dropped_journals)} total):")
if len(dropped_journals) > 0:
    for i, row in dropped_journals.head(5).iterrows():
        print(f"  - {row['article_id']}: {row['Title'][:60]}... by {row['n_id']}")
