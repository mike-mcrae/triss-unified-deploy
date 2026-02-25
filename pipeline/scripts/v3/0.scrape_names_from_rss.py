#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRISS RSS profile scraper (2026 pipeline)
- Uses staff_cleaned.csv as authoritative source of n_id
- Loads each person in RSS dropdown and saves HTML

You will need to run this manually with a Selenium/Chrome session.
"""

import os
import time
import pandas as pd
from seleniumbase import Driver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
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

BASE_DIR = str(_PIPELINE_ROOT)
CSV_PATH = os.path.join(BASE_DIR, "1. data/1. raw/staff_cleaned.csv")
SAVE_DIR = os.path.join(BASE_DIR, "1. data/1. raw/0. profiles")
TRACKING_CSV = os.path.join(SAVE_DIR, "profile_scrape_log.csv")

os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- Load staff list ----------
df = pd.read_csv(CSV_PATH)

# ---------- Launch browser ----------
driver = Driver(uc=True, headless=False)
wait = WebDriverWait(driver, 15)

driver.get("https://www.tcd.ie/library/riss/research-support-system.php")

# You must navigate to the RSS page manually if needed, then run this script.
# If already on page, it will proceed to scrape.

# ---------- Scrape loop ----------
for i, row in df.iterrows():
    try:
        entry_text = f"{row.lastname}, {row.firstname} ({row.role}, {row.department})"
        print(f"[{i+1}/{len(df)}] Processing: {entry_text}")
        dropdown = driver.find_element(By.XPATH, '//*[@id="P396_PERSON"]')
        select = Select(dropdown)
        select.select_by_visible_text(entry_text)
        time.sleep(2)
        if i == 0:
            dropdown2 = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="Publications_row_select"]')))
            select2 = Select(dropdown2)
            select2.select_by_value("100000")
            time.sleep(3)
        html_source = driver.page_source
        safe_name = f"{row.lastname}_{row.firstname}.html".replace(" ", "_")
        save_path = os.path.join(SAVE_DIR, safe_name)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html_source)
        df.loc[i, "status"] = "success"
        if (i + 1) % 10 == 0:
            df.to_csv(TRACKING_CSV, index=False)
    except Exception as e:
        print(f"  ❌ Failed for {row.lastname}, {row.firstname}: {e}")
        df.loc[i, "status"] = f"failed: {e}"
        df.to_csv(TRACKING_CSV, index=False)

# Final write
if "status" in df.columns:
    df.to_csv(TRACKING_CSV, index=False)

print(f"Done. HTML saved to: {SAVE_DIR}")
