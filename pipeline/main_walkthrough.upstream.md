# TRISS Pipeline V3 Documentation

This document outlines the **V3 Pipeline** used to process, clean, and enrich publication data for the TRISS project. The pipeline transitions from raw scraped JSON data to a fully enriched CSV dataset suitable for analysis, leveraging LLMs to recover missing metadata.

## 1. Input Data

The pipeline operates on two primary data sources:

1.  **`staff_cleaned.csv`** (User Created)
    -   **Location**: `1. data/1. raw/staff_cleaned.csv`
    -   **Description**: A manually curated list of staff members.
    -   **Schema**: `department`, `school`, `lastname`, `firstname`, `role`, `n_id`.
    -   **Role**: Used to link publications to specific staff members via `n_id`.

2.  **`profiles_parsed.json`** (Scraper Output)
    -   **Location**: `1. data/1. raw/0. profiles/v3/profiles_parsed.json`
    -   **Description**: The raw output from the RSS/Profile scraper. contains a nested JSON structure where each key is a profile ID containing a list of publication objects.

## 2. Pipeline Steps & Scripts

All scripts are located in: `0. scripts/v3/`
All outputs are saved to: `1. data/1. raw/0. profiles/v3/`

### Step 1: JSON to CSV Conversion
-   **Script**: `3. json_to_csv.py`
-   **Purpose**: Flattens the nested `profiles_parsed.json` into a tabular CSV format. It also merges staff metadata (Department, School) from `staff_cleaned.csv` (via `profiles_summary.csv` logic).
-   **Input**: `profiles_parsed.json`
-   **Output**: `0.all_listed_pubs.csv`
    -   ~24,567 rows.
    -   One row per publication.

### Step 2: Publication Type Analysis
-   **Script**: `4. count_pub_types.py`
-   **Purpose**: Aggregates and counts publications by their raw "Type" string to understand the distribution of content (e.g., Journal Article vs. Conference Paper).
-   **Input**: `0.all_listed_pubs.csv`
-   **Output**: `1.publication_types_totals.csv`

### Step 3: Missing Year Extraction (LLM)
-   **Script**: `5. extract_year_llm.py`
-   **Purpose**: Identifies records where the `Year` field is missing (`"-"`). Uses OpenAI (GPT-3.5) to infer the year from the `Publication Reference` string.
-   **Method**: Sequential row processing with rate limiting.
-   **Input**: `0.all_listed_pubs.csv`
-   **Output**: `0.all_listed_pubs_enriched.csv`
    -   Adds columns: `llm_extraction_meta` (confidence scores).
    -   Recovered ~92 missing years.

### Step 4: Author & Title Extraction (LLM Batch)
-   **Script**: `7. extract_authors_title_openai_batch.py`
-   **Purpose**: Extracts structured `Authors` and `Title` fields from the unstructured `Publication Reference` string for all records where these fields were missing or unparsed.
-   **Method**: **OpenAI Batch API** (Asynchronous).
    1.  Generates a JSONL file of ~24,500 requests.
    2.  Uploads to OpenAI and creates a Batch Job.
    3.  Polls for completion (up to 24h).
    4.  Downloads and merges results.
-   **Input**: `0.all_listed_pubs_enriched.csv`
-   **Output**: `0.all_listed_pubs_fully_enriched.csv`
    -   Adds columns: `Authors`, `Title`, `llm_authors_title_meta`.
    -   **Status**: Batch Completed (`batch_6992ef6d0c948190aa56c06bbb66b145`). Processed 24,548 rows.

### Step 5: 2019+ Filter & Count
-   **Script**: `8. filter_and_count_2019.py`
-   **Purpose**: Creates a subset of publications from 2019 onwards and generates publication type counts for this subset.
-   **Input**: `0.all_listed_pubs_fully_enriched.csv`
-   **Outputs**:
    -   `2.all_listed_pubs_2019.csv` (7,962 records).
    -   `2b.publication_types_totals_2019.csv` (Type counts).

### Step 6: Publication Type Filtering
-   **Script**: `9. filter_pub_types.py`
-   **Purpose**: Creates subsets containing only key academic outputs ("Book", "Book Chapter", "Journal", "Journal Article", "Working Paper").
-   **Inputs**:
    1.  `0.all_listed_pubs_fully_enriched.csv`
    2.  `2.all_listed_pubs_2019.csv`
-   **Outputs**:
    -   `0c.all_listed_publications_filter.csv` (15,696 records from full dataset).
    -   `2c.all_listed_publications_2019.csv` (5,125 records from 2019+ subset).

### Step 7: Profile Aggregation (V3)
-   **Script**: `10. aggregate_profiles_v3.py`
-   **Purpose**: Generates per-profile statistics using the V3 data and specific publication type filters.
-   **Input**: `profiles_parsed.json`, `staff_cleaned.csv`
-   **Output**: `profiles_summary_v3.csv`
    -   Aggregates: `total_filtered_pubs`, `total_filtered_pubs_2019`.
    -   Breakdowns: Counts for Book, Journal Article, etc. (All years and 2019+).

## 3. Final Outputs

The pipeline produces a hierarchy of progressively enriched datasets:

| File Name | Description | Key Features |
| :--- | :--- | :--- |
| **`0.all_listed_pubs.csv`** | Raw Flattened Data | Base dataset from JSON. |
| **`0.all_listed_pubs_enriched.csv`** | + Recovered Years | Includes LLM-inferred years for missing records. |
| **`0.all_listed_pubs_fully_enriched.csv`** | + Authors & Titles | **Final Full Dataset**. Fully parsed authors and titles. |
| **`0c.all_listed_publications_filter.csv`** | Key Types Subset | Filtered for Books, Journals, Working Papers only. |
| **`2.all_listed_pubs_2019.csv`** | 2019+ Subset | Filtered dataset representing recent output. |
| **`2c.all_listed_publications_2019.csv`** | 2019+ Key Types | Combined date and type filtering. |
| **`profiles_summary.csv`** | Profile Statistics | Aggregated counts per staff member (V3 logic). |

## 4. Operational Notes

-   **Resumability**: The Batch extraction script (`7. extract_authors_title_openai_batch.py`) maintains state in `_batch_runs/authors_title/state.json`. It can be stopped and restarted without losing progress.
-   **Logging**: All LLM interactions are logged to `0.all_listed_pubs_llm_log.jsonl` (for the prototype) or handled via the Batch API response files.
