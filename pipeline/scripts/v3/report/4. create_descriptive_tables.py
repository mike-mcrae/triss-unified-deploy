#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate descriptive tables for the V3 LaTeX report.
1. faculty_catalogue_by_department_tabular.tex
2. faculty_catalogue_by_school_tabular.tex
3. research_metadata_coverage_tabular.tex
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
BASE      = _PIPELINE_ROOT
PROFILES  = BASE / "1. data/3. final/1. profiles_summary.csv"
PUBS_ALL  = BASE / "1. data/3. final/2. All listed publications.csv"
PUBS_2019 = BASE / "1. data/3. final/4.All measured publications.csv"  # The final cleaned publications
GLOBAL    = BASE / "1. data/5. analysis/global"

OUTDIR = BASE / "1. data/5. analysis/global/latex"
OUTDIR.mkdir(parents=True, exist_ok=True)

SCHOOL_ABBREV = {
    "Linguistic, Speech and Communication Sciences": "LSLCS",
    "Religion, Theology and Peace Studies": "RTP",
    "Social Work and Social Policy": "SWSP",
    "Social Sciences and Philosophy": "SSP",
}

DEPT_ABBREV = {
    "Clin Speech & Language Studies": "Clin Speech",
    "Religions and Theology": "Theology",
    "School of Religion": "Religion",
    "Trinity Business School": "Business",
}

# 1. Load Data
profiles = pd.read_csv(PROFILES)
pubs_all = pd.read_csv(PUBS_ALL)
pubs_2019 = pd.read_csv(PUBS_2019)

profiles["n_id"] = pd.to_numeric(profiles["n_id"], errors="coerce")
pubs_all["n_id"] = pd.to_numeric(pubs_all["n_id"], errors="coerce")
pubs_2019["n_id"] = pd.to_numeric(pubs_2019["n_id"], errors="coerce")

profiles["school"] = profiles["school"].replace(SCHOOL_ABBREV)
profiles["department"] = profiles["department"].replace(DEPT_ABBREV)
profiles["n_publications"] = pd.to_numeric(profiles["n_publications"], errors="coerce").fillna(0).astype(int)

# 2. Coverage Table
total_researchers = profiles["n_id"].nunique()
total_publications = len(pubs_all)
publications_2019_count = len(pubs_2019)
researchers_with_pubs_2019 = pubs_2019["n_id"].nunique()
abstracts_analysed = pubs_2019["abstract"].notna().sum() if "abstract" in pubs_2019.columns else publications_2019_count

# Researchers actually analysed = those with ≥1 abstract → embedding generated → placed in UMAP
umap_path = GLOBAL / "global_umap_coordinates.csv"
if umap_path.exists():
    umap_df = pd.read_csv(umap_path)
    researchers_analysed = umap_df["n_id"].nunique()
else:
    # Fallback: count researchers who have at least one abstract
    researchers_analysed = pubs_2019[pubs_2019["abstract"].notna()]["n_id"].nunique()

rows = [
    ("Total TRISS Researchers", total_researchers),
    ("Researchers with Publications (2019+)", researchers_with_pubs_2019),
    ("Researchers Analysed (with abstracts)", researchers_analysed),
    ("Publications Since 2019", publications_2019_count),
    ("Publication Abstracts Analysed", abstracts_analysed),
]
coverage_df = pd.DataFrame(rows, columns=["Measure", "Count"])

with open(OUTDIR / "research_metadata_coverage_tabular.tex", "w") as f:
    f.write("\\begin{tabular}{l c}\n")
    f.write("\\toprule\n")
    f.write("Measure & Count \\\\\n")
    f.write("\\midrule\n")
    for _, row in coverage_df.iterrows():
        f.write(f"{row['Measure']} & {row['Count']} \\\\\n")
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")

# 3. Department & School Tables
def is_core_type(t):
    t = str(t).lower()
    return "journal" in t or "working paper" in t or "book" in t

pubs_all["is_core"] = pubs_all["Publication Type"].apply(is_core_type)
core_pubs_all = pubs_all[pubs_all["is_core"]].groupby("n_id").size().reset_index(name="core_pubs_all")

pubs_2019_by_researcher = pubs_2019.groupby("n_id", as_index=False).size().rename(columns={"size": "publications_2019_plus"})
df = profiles.merge(pubs_2019_by_researcher, on="n_id", how="left")
df = df.merge(core_pubs_all, on="n_id", how="left")
df["publications_2019_plus"] = df["publications_2019_plus"].fillna(0).astype(int)
df["core_pubs_all"] = df["core_pubs_all"].fillna(0).astype(int)
df["active_2019_plus"] = df["publications_2019_plus"] > 0
df.rename(columns={"n_publications": "output_all"}, inplace=True)

dept = df.groupby(["school", "department"], as_index=False).agg(
    researchers_all=("n_id", "nunique"),
    researchers_2019_plus=("active_2019_plus", "sum"),
    output_all=("output_all", "sum"),
    publications_all=("core_pubs_all", "sum"),
    publications_2019_plus=("publications_2019_plus", "sum"),
)
dept["avg_pubs_all"] = (dept["publications_all"] / dept["researchers_all"]).round(1)
dept["avg_pubs_2019_plus"] = (dept["publications_2019_plus"] / dept["researchers_2019_plus"].replace(0, pd.NA)).round(1)
dept["avg_pubs_2019_plus"] = dept["avg_pubs_2019_plus"].fillna("")

school = dept.groupby("school", as_index=False).agg(
    researchers_all=("researchers_all", "sum"),
    researchers_2019_plus=("researchers_2019_plus", "sum"),
    output_all=("output_all", "sum"),
    publications_all=("publications_all", "sum"),
    publications_2019_plus=("publications_2019_plus", "sum"),
)
school["avg_pubs_all"] = (school["publications_all"] / school["researchers_all"]).round(1)
school["avg_pubs_2019_plus"] = (school["publications_2019_plus"] / school["researchers_2019_plus"].replace(0, pd.NA)).round(1)
school["avg_pubs_2019_plus"] = school["avg_pubs_2019_plus"].fillna("")

def suppress_repeated_values(df, column):
    out, last = [], None
    for v in df[column]:
        out.append("" if v == last else v)
        last = v
    df = df.copy()
    df[column] = out
    return df

def add_total_row(df, label_col):
    df = df.copy()
    df["_is_total"] = False
    total = {
        label_col: "Total",
        "researchers_all": df["researchers_all"].sum(),
        "researchers_2019_plus": df["researchers_2019_plus"].sum(),
        "output_all": df["output_all"].sum(),
        "publications_all": df["publications_all"].sum(),
        "publications_2019_plus": df["publications_2019_plus"].sum(),
    }
    total["avg_pubs_all"] = round(total["publications_all"] / total["researchers_all"], 1) if total["researchers_all"] > 0 else ""
    total["avg_pubs_2019_plus"] = round(total["publications_2019_plus"] / total["researchers_2019_plus"], 1) if total["researchers_2019_plus"] > 0 else ""
    total["_is_total"] = True
    return pd.concat([df, pd.DataFrame([total])], ignore_index=True)

def write_tabular_only(df, path, columns, col_spec, headers):
    with open(path, "w") as f:
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(headers) + " \\\\\n")
        f.write("\\midrule\n")
        for _, row in df.iterrows():
            if row.get("_is_total", False):
                f.write("\\midrule\n")
            f.write(" & ".join(str(row[c]) for c in columns) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

dept_for_latex = dept.sort_values(["school", "department"]).pipe(suppress_repeated_values, "school").pipe(add_total_row, "school")
write_tabular_only(
    dept_for_latex, OUTDIR / "faculty_catalogue_by_department_tabular.tex",
    ["school", "department", "researchers_all", "researchers_2019_plus", "output_all", "publications_all", "publications_2019_plus", "avg_pubs_all", "avg_pubs_2019_plus"],
    "p{2.5cm} p{4.5cm} c c c c c c c",
    ["School", "Department", "R (All)", "R (2019+)", "Output", "Pubs (All)", "Pubs (2019+)", "Avg. Pubs (All)", "Avg. Pubs (2019+)"]
)

school_for_latex = school.sort_values("school").pipe(add_total_row, "school")
write_tabular_only(
    school_for_latex, OUTDIR / "faculty_catalogue_by_school_tabular.tex",
    ["school", "researchers_all", "researchers_2019_plus", "output_all", "publications_all", "publications_2019_plus", "avg_pubs_all", "avg_pubs_2019_plus"],
    "p{3.5cm} c c c c c c c",
    ["School", "R (All)", "R (2019+)", "Output", "Pubs (All)", "Pubs (2019+)", "Avg. Pubs (All)", "Avg. Pubs (2019+)"]
)

print(f"Generated descriptive tables at {OUTDIR}")
