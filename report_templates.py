"""
report_templates.py  — FINAL VERSION WITH FUZZY MATCHING
---------------------------------------------------------
✔ Recursive scan of /home/thirumal/Desktop/cbct_ai_app/templates
✔ Robust fuzzy case-type mapping (Full Skull, Maxilla, Mandible, Implant…)
✔ Supports filename typos (SKUl, SKuLL, SKULL, etc.)
✔ Extracts FINDINGS, IMPRESSION, RECOMMENDATIONS
✔ Logs how many templates are loaded per case-type
"""

import os
import re
from typing import List, Dict
from docx import Document

# ------------------------------------------------------------
# 1. REAL TEMPLATE DIRECTORY
# ------------------------------------------------------------
TEMPLATE_DIR = "/home/thirumal/Desktop/cbct_ai_app/templates"

# ------------------------------------------------------------
# 2. FUZZY CASE-TYPE PATTERNS
# ------------------------------------------------------------
CASE_PATTERNS = {
    "full_skull": [
        r"full\s*skull",
        r"cbct\s*full\s*skull",
        r"full\s*skul",           # SKUl, SKULl, etc.
        r"skull",
        r"skul",                  # fuzzy root
        r"full\s*head",
        r"pan\s*facial",
    ],
    "maxilla": [
        r"maxilla",
        r"maxillary",
        r"upper\s*jaw",
    ],
    "mandible": [
        r"mandible",
        r"mandibular",
        r"lower\s*jaw",
    ],
    "implant": [
        r"implant",
        r"fixture",
        r"osteotomy",
    ],
    "cyst": [
        r"cyst",
        r"cystic",
    ],
    "fracture": [
        r"fracture",
        r"crack",
        r"break",
    ],
    "impacted": [
        r"impacted",
        r"unerupted",
    ],
    "supernumerary": [
        r"supernumerary",
        r"extra\s*tooth",
    ],
    "periapical": [
        r"periapical",
    ],
    "rct": [
        r"root\s*canal",
        r"rct",
    ],
    "orthodontic": [
        r"orthodont",
        r"braces",
    ],
    "sinus": [
        r"sinus",
        r"sinusal",
    ],
    "ridge": [
        r"ridge",
        r"alveolar\s*ridge",
    ],
}

# ------------------------------------------------------------
# 3. STORAGE
# ------------------------------------------------------------
TEMPLATES: Dict[str, Dict[str, List[str]]] = {}
TEMPLATE_COUNTS: Dict[str, int] = {}


# ------------------------------------------------------------
# 4. Cleaning utility
# ------------------------------------------------------------
def _clean(text: str) -> str:
    text = text.replace("\r", "").strip()
    text = re.sub(r"[ \t]+", " ", text)
    return text


# ------------------------------------------------------------
# 5. Extract sections from a DOCX
# ------------------------------------------------------------
def _extract_sections(path: str) -> Dict[str, List[str]]:
    findings, impression, recommendations = [], [], []
    try:
        doc = Document(path)
    except Exception:
        return {"findings": [], "impression": [], "recommendations": []}

    mode = None
    for para in doc.paragraphs:
        t = _clean(para.text)
        if not t:
            continue
        up = t.upper()

        if "RADIOGRAPHIC INTERPRETATION" in up:
            mode = "findings"
            continue

        if "RADIOGRAPHIC IMPRESSION" in up:
            mode = "impression"
            continue

        if mode == "findings":
            findings.append(t)
        elif mode == "impression":
            impression.append(t)

        # recommendation detection
        if "correlate" in t.lower() or "recommend" in t.lower():
            recommendations.append(t)

    if not recommendations:
        recommendations = [
            "Correlate imaging with clinical findings.",
            "If diagnostic uncertainty persists, consider further clinical evaluation."
        ]

    return {
        "findings": findings,
        "impression": impression,
        "recommendations": recommendations,
    }


# ------------------------------------------------------------
# 6. Fuzzy case-type inference
# ------------------------------------------------------------
def _infer_case_type(filename: str) -> str:
    name = filename.lower()

    for case_type, patterns in CASE_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, name, flags=re.I):
                return case_type

    return "general"


# ------------------------------------------------------------
# 7. Recursive loader
# ------------------------------------------------------------
def load_all_templates():
    global TEMPLATES, TEMPLATE_COUNTS
    TEMPLATES = {}
    TEMPLATE_COUNTS = {}

    if not os.path.isdir(TEMPLATE_DIR):
        print(f"[TEMPLATE ERROR] Directory not found: {TEMPLATE_DIR}")
        return

    for root, _, files in os.walk(TEMPLATE_DIR):
        for f in files:
            if not f.lower().endswith(".docx"):
                continue

            full_path = os.path.join(root, f)
            case_type = _infer_case_type(f)
            sections = _extract_sections(full_path)

            if case_type not in TEMPLATES:
                TEMPLATES[case_type] = {
                    "findings": [],
                    "impression": [],
                    "recommendations": []
                }
                TEMPLATE_COUNTS[case_type] = 0

            # append unique
            for sec in ["findings", "impression", "recommendations"]:
                for line in sections[sec]:
                    if line and line not in TEMPLATES[case_type][sec]:
                        TEMPLATES[case_type][sec].append(line)

            TEMPLATE_COUNTS[case_type] += 1

    # fallback general
    if "general" not in TEMPLATES:
        TEMPLATES["general"] = {
            "findings": ["Normal anatomical structures within expected limits."],
            "impression": ["No significant abnormality detected."],
            "recommendations": ["Correlate imaging with clinical findings."]
        }
        TEMPLATE_COUNTS["general"] = 1

    print("\n=== TEMPLATE LOADING REPORT ===")
    for ct, count in TEMPLATE_COUNTS.items():
        print(f"{ct}: {count} files loaded")
    print("================================\n")


# ------------------------------------------------------------
# 8. Public API
# ------------------------------------------------------------
def list_case_types() -> List[str]:
    if not TEMPLATES:
        load_all_templates()
    return list(TEMPLATES.keys())


def get_template_for_case(case_type: str) -> List[str]:
    if not TEMPLATES:
        load_all_templates()
    return TEMPLATES.get(case_type, TEMPLATES["general"])["findings"]


def get_impression_for_cases(case_types: List[str]) -> List[str]:
    if not TEMPLATES:
        load_all_templates()
    out = []
    for ct in case_types:
        out.extend(TEMPLATES.get(ct, {}).get("impression", []))
    if not out:
        out = TEMPLATES["general"]["impression"]
    return out


def get_recommendations_for_cases(case_types: List[str]) -> List[str]:
    if not TEMPLATES:
        load_all_templates()
    out = []
    for ct in case_types:
        out.extend(TEMPLATES.get(ct, {}).get("recommendations", []))
    if not out:
        out = TEMPLATES["general"]["recommendations"]
    return out


# ------------------------------------------------------------
# 9. Auto-load at import
# ------------------------------------------------------------
try:
    load_all_templates()
except Exception as e:
    print(f"[TEMPLATE LOAD ERROR] {e}")
