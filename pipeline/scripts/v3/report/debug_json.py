#!/usr/bin/env python3
import json
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
with open(BASE / "1. data/3. final/counts_2019_plus.json") as f:
    data = json.load(f)

tally = {"Journal Article": 0, "Book Chapter": 0, "Book": 0, "Working Paper": 0, "Other": 0, "Total": 0}

for k, v in data.get("n_id", {}).items():
    for type_id, type_info in v.get("by_type", {}).items():
        label = type_info["label"]
        in_sample = type_info["in_sample"]
        
        tally[label] += in_sample
        tally["Total"] += in_sample

print("TOTALS FROM JSON in_sample:")
for k, v in tally.items():
    print(f"  {k}: {v}")
