#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse TRISS RSS profile HTMLs and build:
- profiles_parsed.json (raw parsed profile data + publications)
- profiles_summary.csv (one row per person with aggregate publication-type counts)

Publication-type aggregation scheme is documented in:
- triss-pipeline-2026/README_publication_types.md
"""

import os
import re
import json
import unicodedata
import pandas as pd
from bs4 import BeautifulSoup
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

# --------------------------------------------------------------------
# PATHS
# --------------------------------------------------------------------
BASE_DIR = str(_PIPELINE_ROOT)
HTML_DIR = os.path.join(BASE_DIR, "1. data/1. raw/0. profiles")
STAFF_CSV = os.path.join(BASE_DIR, "1. data/1. raw/staff_cleaned.csv")

OUT_JSON  = os.path.join(HTML_DIR, "profiles_parsed.json")
OUT_CSV   = os.path.join(HTML_DIR, "profiles_summary.csv")
OUT_UNMATCHED = os.path.join(HTML_DIR, "profiles_unmatched.csv")

# --------------------------------------------------------------------
# NAME NORMALIZATION
# --------------------------------------------------------------------
def norm_name(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("’", "'").replace("`", "'")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def best_name_split(base: str, staff_lookup: dict, max_last_parts: int = 3):
    parts = base.split("_")
    if len(parts) < 2:
        return base, ""

    max_k = min(max_last_parts, len(parts) - 1)
    for k in range(max_k, 0, -1):
        last_raw = "_".join(parts[:k])
        first_raw = "_".join(parts[k:])

        last_key = norm_name(last_raw.replace("_", " "))
        first_key = norm_name(first_raw.replace("_", " "))

        if (last_key, first_key) in staff_lookup:
            return last_raw, first_raw

    return parts[0], "_".join(parts[1:])


# --------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------
def clean_text(txt):
    return " ".join(txt.split()) if txt else None


def extract_text(soup, selector):
    el = soup.select_one(selector)
    return clean_text(el.get_text(" ", strip=True)) if el else None


def extract_table(section_id, soup):
    panel = soup.select_one(f"#{section_id}_data_panel")
    if not panel:
        return None

    table = panel.select_one("table[id$='_orig']")
    if not table:
        for t in panel.select("table"):
            if t.select_one("tbody td"):
                table = t
                break
    if not table:
        return None

    headers = [clean_text(th.get_text(" ", strip=True)) for th in table.select("th")]
    rows = []
    for tr in table.select("tbody tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        cells = [clean_text(td.get_text(" ", strip=True)) for td in tds]
        if headers and len(cells) == len(headers):
            rows.append(dict(zip(headers, cells)))
        else:
            rows.append(cells)
    return rows or None


def extract_research_interests_apex(soup):
    label_re = re.compile(r"^\s*Description of Research Interests\b", re.I)

    for el in soup.find_all(["div", "p", "td", "span", "li"]):
        t = clean_text(el.get_text(" ", strip=True) or "")
        if not t:
            continue
        if "Show All" in t and "CV Sections" in t:
            continue

        if label_re.search(t):
            v = label_re.sub("", t).strip(" :\n\t")
            v = re.sub(r"^\s*Biography\b", "", v, flags=re.I).strip(" :\n\t")
            if not v:
                sib = el.find_next_sibling(["div", "p", "td"])
                if sib:
                    v = clean_text(sib.get_text(" ", strip=True) or "")
                    v = re.sub(r"^\s*Biography\b", "", v, flags=re.I).strip(" :\n\t")
            if v and len(v) >= 10 and "Home RSS CV" not in v:
                return v

    return None


# --------------------------------------------------------------------
# PARSE SINGLE PROFILE
# --------------------------------------------------------------------
def parse_profile(html_path):
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    research_interests = extract_text(soup, "#P396_RESEARCH_INTERESTS_DISPLAY")
    if not research_interests:
        label = "Description of Research Interests"
        for el in soup.find_all(["div", "p", "td", "span", "li"]):
            txt = clean_text(el.get_text(" ", strip=True) or "")
            if not txt:
                continue
            if "Show All CV Sections" in txt or "OR Select CV Section" in txt:
                continue
            if txt.startswith(label):
                val = txt[len(label):].strip(" :\n\t")
                if val.lower().startswith("biography"):
                    val = val[len("biography"):].strip(" :\n\t")
                if len(val) >= 10 and "Home RSS CV" not in val:
                    research_interests = val
                    break

    return {
        "email": extract_text(soup, "#P396_EMAIL_DISPLAY a"),
        "biography": extract_text(soup, "#P396_BIOGRAPHY_DISPLAY"),
        "research_interests": research_interests or extract_research_interests_apex(soup),
        "themes": extract_table("Themes", soup),
        "keywords": extract_table("Keywords", soup),
        "tags": extract_table("Tags", soup),
        "research_projects": extract_table("Research_Projects", soup),
        "publications": extract_table("Publications", soup),
    }


# --------------------------------------------------------------------
# AGGREGATION SCHEME
# --------------------------------------------------------------------
AGG_SCHEME = {
    "agg_journal_articles": [
        "Journal Article",
        "Research Note",
        "Journal",
        "Published Abstract",
    ],
    "agg_books_chapters": [
        "Book",
        "Book Chapter",
        "Critical Edition (Book)",
        "Critical Edition (Chapter)",
    ],
    "agg_conference_outputs": [
        "Conference Paper",
        "Proceedings of a Conference",
        "Meeting Abstract",
        "Poster",
    ],
    "agg_presentations": [
        "Invited Talk",
        "Oral Presentation",
    ],
    "agg_reports_working": [
        "Report",
        "Working Paper",
        "Protocol or guideline",
    ],
    "agg_other": [
        "Archaeological excavation work",
        "Book Review",
        "Broadcast",
        "Campus Company",
        "Case Study",
        "Dataset",
        "Digital research resource production",
        "Editorial Board",
        "Exhibition",
        "Fiction and creative prose",
        "Fieldwork collection",
        "Film production",
        "Impact Case Study",
        "Map, GIS map",
        "Meetings /Conferences Organised",
        "Miscellaneous",
        "Music Production",
        "News Item",
        "Newspaper/Magazine Articles",
        "Patent",
        "Poetry",
        "Review",
        "Review Article",
        "Script",
        "Software",
        "Test or assessment",
        "Theatre Production",
        "Thesis",
        "Translation",
        "Visual art production",
        "Item in dictionary or encyclopaedia, etc",
    ],
}


def pub_year_is_2019_plus(pub):
    year = pub.get("Year") or pub.get("Year Descending") or ""
    year = str(year).strip()
    if year.isdigit() and int(year) >= 2019:
        return True
    return False


def count_agg_types(pubs, only_2019_plus=False):
    counts = {k: 0 for k in AGG_SCHEME.keys()}
    type_to_group = {}
    for grp, types in AGG_SCHEME.items():
        for t in types:
            type_to_group[t] = grp

    for pub in pubs:
        if not isinstance(pub, dict):
            continue
        if only_2019_plus and not pub_year_is_2019_plus(pub):
            continue
        ptype = pub.get("Publication Type")
        if not ptype:
            continue
        grp = type_to_group.get(ptype)
        if grp:
            counts[grp] += 1
        else:
            # unseen type falls into other
            counts["agg_other"] += 1

    return counts


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
def main():
    staff_df = pd.read_csv(STAFF_CSV, dtype=str)
    for col in ["firstname", "lastname", "n_id"]:
        if col not in staff_df.columns:
            raise ValueError(f"Missing required column '{col}' in {STAFF_CSV}")

    staff_df["firstname_key"] = staff_df["firstname"].apply(norm_name)
    staff_df["lastname_key"]  = staff_df["lastname"].apply(norm_name)
    staff_df["n_id"] = pd.to_numeric(staff_df["n_id"], errors="coerce")

    staff_lookup = (
        staff_df.dropna(subset=["n_id"]).drop_duplicates(
            subset=["lastname_key", "firstname_key"], keep="first"
        )
        .set_index(["lastname_key", "firstname_key"])["n_id"].to_dict()
    )

    records = []
    summary_rows = []
    unmatched = []

    html_files = [f for f in os.listdir(HTML_DIR) if f.endswith(".html")]

    for fname in html_files:
        base = os.path.splitext(fname)[0]
        last_raw, first_raw = best_name_split(base, staff_lookup)

        last_key = norm_name(last_raw.replace("_", " "))
        first_key = norm_name(first_raw.replace("_", " "))

        n_id = staff_lookup.get((last_key, first_key))
        if pd.isna(n_id) or n_id is None:
            unmatched.append({
                "file": fname,
                "lastname_raw": last_raw,
                "firstname_raw": first_raw,
            })
            continue

        html_path = os.path.join(HTML_DIR, fname)
        parsed = parse_profile(html_path)

        # Attach identity
        parsed["n_id"] = int(n_id)
        parsed["firstname"] = staff_df.loc[
            (staff_df["lastname_key"] == last_key) &
            (staff_df["firstname_key"] == first_key),
            "firstname"
        ].iloc[0]
        parsed["lastname"] = staff_df.loc[
            (staff_df["lastname_key"] == last_key) &
            (staff_df["firstname_key"] == first_key),
            "lastname"
        ].iloc[0]
        parsed["department"] = staff_df.loc[
            (staff_df["lastname_key"] == last_key) &
            (staff_df["firstname_key"] == first_key),
            "department"
        ].iloc[0]
        parsed["school"] = staff_df.loc[
            (staff_df["lastname_key"] == last_key) &
            (staff_df["firstname_key"] == first_key),
            "school"
        ].iloc[0]
        parsed["role"] = staff_df.loc[
            (staff_df["lastname_key"] == last_key) &
            (staff_df["firstname_key"] == first_key),
            "role"
        ].iloc[0]

        records.append(parsed)

        pubs = parsed.get("publications") or []
        counts_all = count_agg_types(pubs, only_2019_plus=False)
        counts_2019 = count_agg_types(pubs, only_2019_plus=True)

        row = {
            "n_id": int(n_id),
            "firstname": parsed.get("firstname"),
            "lastname": parsed.get("lastname"),
            "department": parsed.get("department"),
            "school": parsed.get("school"),
            "role": parsed.get("role"),
            "email": parsed.get("email"),
            "biography": parsed.get("biography"),
            "research_interests": parsed.get("research_interests"),
            "n_publications": len(pubs),
        }

        for k in AGG_SCHEME.keys():
            row[k] = counts_all[k]
            row[f"{k}_2019"] = counts_2019[k]

        # composite published / non-published
        row["agg_published_material"] = (
            row["agg_journal_articles"] +
            row["agg_books_chapters"] +
            row["agg_reports_working"]
        )
        row["agg_published_material_2019"] = (
            row["agg_journal_articles_2019"] +
            row["agg_books_chapters_2019"] +
            row["agg_reports_working_2019"]
        )
        row["agg_non_published"] = (
            row["agg_conference_outputs"] +
            row["agg_presentations"] +
            row["agg_other"]
        )
        row["agg_non_published_2019"] = (
            row["agg_conference_outputs_2019"] +
            row["agg_presentations_2019"] +
            row["agg_other_2019"]
        )

        summary_rows.append(row)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    pd.DataFrame(summary_rows).to_csv(OUT_CSV, index=False)

    if unmatched:
        pd.DataFrame(unmatched).to_csv(OUT_UNMATCHED, index=False)
        print(f"⚠️ Unmatched profiles: {len(unmatched)} -> {OUT_UNMATCHED}")

    print(f"✅ Wrote JSON: {OUT_JSON}")
    print(f"✅ Wrote summary: {OUT_CSV}")


if __name__ == "__main__":
    main()
