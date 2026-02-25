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
csv_path = _PIPELINE_ROOT / "1. data/1. raw/0. profiles/v3/2e.all_listed_publications_2019_books_other_abstracts.csv"
try:
    df = pd.read_csv(csv_path, low_memory=False, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(csv_path, low_memory=False, encoding="latin1")
print(f"Total Rows: {len(df)}")
populated = df[df['abstract'].notna() & (df['abstract'] != '') & (df['abstract'] != 'NO_CONTENT')]
print(f"Populated Abstracts: {len(populated)}")
