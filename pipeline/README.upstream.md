# TRISS Pipeline 2026

This folder is a clean rebuild of the TRISS pipeline using a fresh scrape.

## Structure
- `0. scripts/` scripts for scraping and processing
- `1. data/` raw and intermediate data

## Current Step
### 1) Scrape RSS profile HTMLs
Script:
- `0. scripts/0.scrape_names_from_rss.py`

Inputs:
- `1. data/1. raw/staff_cleaned.csv`

Outputs:
- `1. data/1. raw/0. profiles/*.html`
- `1. data/1. raw/0. profiles/profile_scrape_log.csv`

Notes:
- You need to run this with a Selenium browser open on the RSS page.
- The script selects each person and saves the resulting HTML.

## Next Steps (planned)
- Parse HTMLs to extract profile + publications
- Filter to “published” items
- Fetch abstracts (Crossref/OpenAlex fallback)
- Clean abstracts
- Create final merged datasets

## Handoff Notes (Feb 7, 2026)
Use this section to continue the pipeline with full network access.

### What is already done
- HTMLs parsed and summaries built.
  - `1. data/1. raw/0. profiles/profiles_parsed.json`
  - `1. data/1. raw/0. profiles/profiles_summary.csv`
- Published-only (2019+) list created:
  - `1. data/1. raw/0. profiles/profiles_publications_published.csv` (5,579 rows)
- OpenAlex initial merge:
  - `1. data/1. raw/0. profiles/profiles_publications_openalex_merged.csv`
- Missing-DOI OpenAlex run completed (4 shards) and merged:
  - `1. data/1. raw/0. profiles/output/profiles_publications_openalex_missing_1.csv`
  - `..._2.csv`, `..._3.csv`, `..._4.csv`
  - Combined: `profiles_publications_openalex_missing_combined.csv`
  - Updated base: `profiles_publications_openalex_merged_updated.csv`
- OpenAlex missing-DOI results were weak:
  - In `profiles_publications_openalex_missing_combined.csv`: only 8 abstracts, 0 DOIs.
  - In `profiles_publications_openalex_merged_updated.csv`: 1,090 DOIs, 997 abstracts.

### Critical issue
Network access for Crossref/OpenAlex failed in the agent environment (DNS). Run Crossref and any other API steps locally with full network.

### Crossref (must run locally)
Use `profiles_publications_openalex_merged_updated.csv` as input (already set in script):
- Script: `0. scripts/3b.get_abstracts_cross_ref_api.py`
- It processes only rows missing DOI or abstract (treats `-` as missing).

Run (2 instances):
```
rm -f "triss-pipeline-2026/1. data/1. raw/0. profiles/output/profiles_publications_crossref_"*.csv
nohup "/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/triss-pipeline-2026/.venv/bin/python" "/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/triss-pipeline-2026/0. scripts/3b.get_abstracts_cross_ref_api.py" --instance 1 --total_instances 2 > /tmp/triss_crossref_1.log 2>&1 &
nohup "/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/triss-pipeline-2026/.venv/bin/python" "/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/triss-pipeline-2026/0. scripts/3b.get_abstracts_cross_ref_api.py" --instance 2 --total_instances 2 > /tmp/triss_crossref_2.log 2>&1 &
```

Then:
- Clean: `0. scripts/3c.clean_abstracts_cross_ref_api.py`
- Combine: `0. scripts/6.combine_crossref_outputs.py`
- Merge found abstracts with full list:
  - `0. scripts/5.merge_found_abstracts_with_full.py`

### Fallback (only after Crossref)
Use the fallback script if Crossref still leaves missing abstracts:
- Script: `0. scripts/4a.fetch_abstracts.py`
- It tries OpenAlex/SS for DOI, then URL HTML/Selenium if URL exists.
- Do not launch Selenium until fallback stage.

### Files to use going forward
- Base list: `profiles_publications_openalex_merged_updated.csv`
- Crossref outputs: `output/profiles_publications_crossref_*.csv`
- Cleaned Crossref: `profiles_publications_crossref_cleaned.csv`
- Final merged abstracts: to be produced after fallback and merge
