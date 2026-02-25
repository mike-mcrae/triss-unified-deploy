#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TRISS V3 Pipeline: Create Distilled Synthesis JSON
Objective: Combine the V2 core identity summary with the new V3 global and school-level
semantic cluster descriptions. Clean the LLM outputs to remove repetitive prefixes
like "This cluster explores...".
"""

import json
import re
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

BASE = _PROJECT_ROOT
V2_SYNTHESIS_FILE = BASE / "1. data/3. final/9. triss_distilled_synthesis.json"
V3_GLOBAL_CLUSTERS = BASE / "triss-pipeline-2026/1. data/5. analysis/global/global_cluster_descriptions.json"
V3_SCHOOL_DIR = BASE / "triss-pipeline-2026/1. data/5. analysis/schools"
OUT_FILE = BASE / "triss-pipeline-2026/1. data/5. analysis/global/triss_distilled_synthesis_v3.json"

def clean_description(desc):
    """
    Remove repetitive prefixes from LLM cluster descriptions and ensure correct capitalization.
    """
    # Patterns to remove
    patterns = [
        r"^This\s+(thematic\s+)?(academic\s+)?cluster\s+(explores|examines|focuses\s+on|addresses|discusses|highlights|investigates)(\s+themes\s+related\s+to|various\s+aspects\s+of|the\s+intersection\s+of|the\s+dynamics\s+of)?\s*",
        r"^These\s+studies\s+(explore|examine|focus\s+on)\s*",
        r"^This\s+group\s+of\s+research\s+"
    ]
    
    cleaned = desc
    for p in patterns:
        cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE)
    
    # Capitalize first letter
    if len(cleaned) > 0:
        cleaned = cleaned[0].upper() + cleaned[1:]
        
    return cleaned

def main():
    print("Generating distilled synthesis JSON...")
    
    # Load V2 core identity
    with open(V2_SYNTHESIS_FILE, "r") as f:
        v2_data = json.load(f)
        
    core_identity = v2_data.get("core_identity_summary", "")
    
    # Load V3 Global Themes
    with open(V3_GLOBAL_CLUSTERS, "r") as f:
        global_clusters = json.load(f)
        
    cross_cutting_themes = []
    for k, v in global_clusters.items():
        cross_cutting_themes.append({
            "theme_name": v.get("topic_name", ""),
            "description": clean_description(v.get("description", ""))
        })
        
    # Load V3 School Themes
    school_themes = {}
    if V3_SCHOOL_DIR.exists():
        for school_path in [d for d in V3_SCHOOL_DIR.iterdir() if d.is_dir()]:
            school_name = school_path.name
            desc_file = school_path / "school_cluster_descriptions.json"
            if desc_file.exists():
                with open(desc_file, "r") as f:
                    s_clusters = json.load(f)
                
                s_themes = []
                for k, v in s_clusters.items():
                    s_themes.append({
                        "theme_name": v.get("topic_name", ""),
                        "description": clean_description(v.get("description", ""))
                    })
                school_themes[school_name] = s_themes
                
    # Compile output
    out_data = {
        "institution_name": "TRISS",
        "core_identity_summary": core_identity,
        "cross_cutting_themes": cross_cutting_themes,
        "school_themes": school_themes
    }
    
    # Save
    with open(OUT_FILE, "w") as f:
        json.dump(out_data, f, indent=2)
        
    print(f"Saved distilled synthesis to {OUT_FILE}")
    print(f"Included {len(cross_cutting_themes)} global themes and themes for {len(school_themes)} schools.")

if __name__ == "__main__":
    main()
