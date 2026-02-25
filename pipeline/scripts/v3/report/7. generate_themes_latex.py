#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TRISS V3 Pipeline: Generate Themes LaTeX
Objective: Parse the distilled synthesis JSON and generate LaTeX fragments for
1) The Global TRISS core identity and cross-cutting themes
2) The School-level themes

These will be \input into report.tex.
"""

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

BASE = _PROJECT_ROOT
SYNTHESIS_JSON = BASE / "triss-pipeline-2026/1. data/5. analysis/global/triss_distilled_synthesis_v3.json"
GLOBAL_TEX_OUT = BASE / "triss-pipeline-2026/1. data/5. analysis/global/latex/triss_overall_profile.tex"
SCHOOLS_BASE_DIR = BASE / "triss-pipeline-2026/1. data/5. analysis/schools"

SCHOOL_NAMES_MAP = {
    "ssp": "Social Sciences and Philosophy",
    "linguistic__speech_and_communication_sciences": "Linguistic, Speech and Communication Sciences",
    "law": "Law",
    "psychology": "Psychology",
    "education": "Education",
    "business": "Business",
    "social_work_and_social_policy": "Social Work and Social Policy",
    "religion__theology_and_peace_studies": "Religion, Theology and Peace Studies"
}

def escape_latex(text):
    """
    Escape basic LaTeX special characters.
    """
    if not text:
        return ""
    text = text.replace("&", r"\&")
    text = text.replace("%", r"\%")
    text = text.replace("$", r"\$")
    text = text.replace("#", r"\#")
    text = text.replace("_", r"\_")
    text = text.replace("{", r"\{")
    text = text.replace("}", r"\}")
    text = text.replace("~", r"\textasciitilde{}")
    text = text.replace("^", r"\textasciicircum{}")
    return text

def main():
    with open(SYNTHESIS_JSON, "r") as f:
        data = json.load(f)
        
    # 1. Generate Global Profile LaTeX
    GLOBAL_TEX_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(GLOBAL_TEX_OUT, "w") as f:
        f.write("% Auto-generated TRISS Global Profile\n")
        f.write(r"\vspace{1em}" + "\n")
        f.write(r"\noindent \textbf{TRISS Core Identity:}" + "\n")
        f.write(r"\begin{quote}" + "\n")
        f.write(escape_latex(data.get("core_identity_summary", "")) + "\n")
        f.write(r"\end{quote}" + "\n\n")
        
        f.write(r"\noindent \textbf{Primary Cross-Cutting Sub-themes:}" + "\n")
        f.write(r"\begin{itemize}" + "\n")
        f.write(r"  \setlength{\itemsep}{1pt}" + "\n")
        f.write(r"  \setlength{\parskip}{0pt}" + "\n")
        f.write(r"  \setlength{\parsep}{0pt}" + "\n")
        
        themes = data.get("cross_cutting_themes", [])
        for theme in themes:
            t_name = escape_latex(theme.get("theme_name", ""))
            t_desc = escape_latex(theme.get("description", ""))
            f.write(f"  \\item \\textbf{{{t_name}}}: {t_desc}\n")
            
        f.write(r"\end{itemize}" + "\n")
        
    print(f"Wrote global profile to {GLOBAL_TEX_OUT}")
    
    # 2. Generate School Profiles LaTeX
    school_themes = data.get("school_themes", {})
    for folder_name, themes in school_themes.items():
        if not themes:
            continue
            
        display_name = SCHOOL_NAMES_MAP.get(folder_name, folder_name.replace("_", " ").title())
        
        school_dir = SCHOOLS_BASE_DIR / folder_name
        school_dir.mkdir(parents=True, exist_ok=True)
        tex_out = school_dir / "school_profile.tex"
        
        with open(tex_out, "w") as f:
            f.write(f"% Auto-generated School Profile for {folder_name}\n")
            f.write(f"\\subsubsection*{{{escape_latex(display_name)}}}\n")
            f.write(r"\begin{itemize}" + "\n")
            f.write(r"  \setlength{\itemsep}{1pt}" + "\n")
            f.write(r"  \setlength{\parskip}{0pt}" + "\n")
            f.write(r"  \setlength{\parsep}{0pt}" + "\n")
            
            for theme in themes:
                t_name = escape_latex(theme.get("theme_name", ""))
                t_desc = escape_latex(theme.get("description", ""))
                f.write(f"  \\item \\textbf{{{t_name}}}: {t_desc}\n")
                
            f.write(r"\end{itemize}" + "\n")
            
        print(f"Wrote school profile to {tex_out}")

if __name__ == "__main__":
    main()
