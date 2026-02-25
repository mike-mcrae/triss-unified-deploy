#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate publication types descriptive table for the V3 LaTeX report.
Produces: publication_types_by_school_tabular.tex
"""

import os
import pandas as pd
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

# Paths
BASE = _PIPELINE_ROOT
PUBS_2019 = BASE / "1. data/3. final/4.All measured publications.csv"

OUTDIR = BASE / "1. data/5. analysis/global/latex"
OUTDIR.mkdir(parents=True, exist_ok=True)

SCHOOL_ABBREV = {
    "Linguistic, Speech and Communication Sciences": "LSLCS",
    "Religion, Theology and Peace Studies": "RTP",
    "Social Work and Social Policy": "SWSP",
    "Social Sciences and Philosophy": "SSP",
}

# 1. Load Data
pubs = pd.read_csv(PUBS_2019)
pubs["school"] = pubs["school"].replace(SCHOOL_ABBREV)

# 2. Map Publication Types — Books and Book Chapters as separate categories
def map_pub_type(t):
    t = str(t).lower()
    if "journal" in t:
        return "Journal Article"
    elif "working paper" in t:
        return "Working Paper"
    elif "book chapter" in t or "chapter" in t:
        return "Book Chapter"
    elif "book" in t:
        return "Book"
    else:
        return "Other"

pubs["Collapsed Type"] = pubs["Publication Type"].apply(map_pub_type)

# 3. Aggregate
agg = pubs.groupby(["school", "Collapsed Type"]).size().unstack(fill_value=0)

# Ensure all expected columns exist
for c in ["Journal Article", "Book", "Book Chapter", "Working Paper", "Other"]:
    if c not in agg.columns:
        agg[c] = 0

agg = agg[["Journal Article", "Book", "Book Chapter", "Working Paper"]]
agg["Total"] = agg.sum(axis=1)

agg.loc["Total"] = agg.sum()
agg = agg.reset_index()

# 4. Write tabular latex
def write_tabular_only(df, path, columns, col_spec, headers):
    with open(path, "w") as f:
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(headers) + " \\\\\n")
        f.write("\\midrule\n")
        for i, row in df.iterrows():
            if row["school"] == "Total":
                f.write("\\midrule\n")
            f.write(" & ".join(str(row[c]) for c in columns) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

write_tabular_only(
    agg, OUTDIR / "publication_types_by_school_tabular.tex",
    ["school", "Journal Article", "Book", "Book Chapter", "Working Paper", "Total"],
    "p{3.5cm} c c c c c",
    ["School", "Journal Article", "Book", "Book Chapter", "Working Paper", "Total"]
)

print(f"Generated publication types table at {OUTDIR}")
