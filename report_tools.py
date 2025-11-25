# report_tools.py
import re
from typing import Dict, List, Tuple
import math

# utility regex
_re_mm = re.compile(r"(\d{1,3}(?:\.\d{1,2})?)\s*mm", flags=re.I)
_re_region_measure = re.compile(r"(region|tooth|rgn|site)\s*[:#]?\s*([0-9]{1,3})", flags=re.I)
_re_numbers_in_context = re.compile(r"(height|width|distance|canal|sinus|apex)[\s:]*([0-9]{1,3}(?:\.\d{1,2})?)\s*mm", flags=re.I)

def _split_sentences(text: str) -> List[str]:
    # naive but fine for clinical text
    s = re.split(r'(?<=[\.\?\!])\s+', text.strip())
    return [x.strip() for x in s if x.strip()]

def _extract_measurements(text: str) -> List[Tuple[str, float]]:
    out = []
    for m in _re_numbers_in_context.finditer(text):
        key = m.group(1).lower()
        val = float(m.group(2))
        out.append((key, val))
    # also generic mm occurrences
    for m in _re_mm.finditer(text):
        val = float(m.group(1))
        out.append(("mm", val))
    return out

def refine_report(raw_text: str, metadata: Dict = None, retrieved: List[str] = None) -> Dict:
    """Turn raw LLM/RAG text into structured clinical report dictionary.
    Returns:
      {
        short_summary: str,
        findings: List[str],
        impression: List[str],
        recommendations: List[str],
        implant_table: List[dict]  # optional
      }
    """
    if metadata is None: metadata = {}
    if retrieved is None: retrieved = []

    sentences = _split_sentences(raw_text or "")
    short_summary = ""
    findings = []
    impression = []
    recommendations = []

    # 1) Short summary: first strong sentence or first line
    if sentences:
        short_summary = sentences[0]
    else:
        short_summary = "No summary available."

    # 2) Findings: sentences containing clinical keywords
    keywords = ["bone", "sinus", "canal", "fracture", "cyst", "periapical", "density", "height", "width", "implant", "dehiscence", "root", "perforation"]
    for s in sentences:
        if any(kw in s.lower() for kw in keywords):
            findings.append(s)

    # fallback: if none, take first 3 sentences
    if not findings:
        findings = sentences[0:3]

    # 3) Impression: condense findings (first 1-2 lines of findings)
    if findings:
        impression_text = " ".join(findings[:2])
        impression.append(impression_text)
    else:
        impression.append("No definite impression could be made from available text.")

    # 4) Recommendations: look for "recommend" or propose basic ones
    for s in sentences:
        if "recommend" in s.lower() or "suggest" in s.lower() or "plan" in s.lower():
            recommendations.append(s)
    if not recommendations:
        # heuristic recommendations
        recommendations.append("Clinical correlation and operator planning recommended.")
        recommendations.append("Verify vertical clearance and proximity to adjacent critical structures (mandibular canal / sinus).")

    # 5) Implant table: attempt to parse region-based measurements
    measures = _extract_measurements(raw_text)
    # naive grouping: if we have >=2 mm values, create a single aggregated table row
    implant_table = []
    if measures:
        # try to find explicit "height" and "width" pairs - grouping by index
        heights = [v for k, v in measures if "height" in k or k=="mm"]
        widths = [v for k, v in measures if "width" in k or k=="mm"]
        # create a single suggestion row if possible
        # we keep this conservative and readable for clinicians
        height = heights[0] if heights else None
        width = widths[0] if widths else None
        row = {
            "region": metadata.get("region", "unspecified"),
            "height_mm": round(height,2) if height else None,
            "width_mm": round(width,2) if width else None,
            "density_est": metadata.get("density","unknown"),
            "suitability": _assess_suitability(height, width)
        }
        implant_table.append(row)

    # Compose structured dict
    out = {
        "short_summary": short_summary,
        "findings": findings,
        "impression": impression,
        "recommendations": recommendations,
        "implant_table": implant_table
    }
    return out

def _assess_suitability(height, width):
    """very conservative rules:
       - height >= 10 mm -> ok
       - width >= 5 mm -> ok
       otherwise flagged.
    """
    if height is None and width is None:
        return "unknown"
    h_ok = (height is not None and height >= 10)
    w_ok = (width is not None and width >= 5)
    if h_ok and w_ok:
        return "suitable"
    if (height is not None and height < 8) or (width is not None and width < 4.5):
        return "insufficient — consider grafting"
    return "borderline — careful planning"

def predict_risks(report_text: str, metadata: Dict = None) -> List[Dict]:
    """Return list of detected risks with scores 0..1 and short rationale.
       Deterministic rules: parse numeric cues (height, width, distance) and detect keywords.
    """
    if metadata is None:
        metadata = {}
    risks = []
    text = report_text or ""
    measures = _extract_measurements(text)

    # parse heights / widths / distances from measures
    heights = [v for k,v in measures if "height" in k or k=="mm"]
    widths = [v for k,v in measures if "width" in k or k=="mm"]
    distances = [v for k,v in measures if "distance" in k or "canal" in k or "sinus" in k]

    # Insufficient height risk
    if heights:
        min_h = min(heights)
        if min_h < 8:
            risks.append({"risk":"insufficient_height", "score":0.95, "reason": f"Minimum height {min_h} mm (<8 mm)."})
        elif min_h < 10:
            risks.append({"risk":"borderline_height", "score":0.6, "reason": f"Minimum height {min_h} mm (8-10 mm borderline)."})
    # Insufficient width risk
    if widths:
        min_w = min(widths)
        if min_w < 4.5:
            risks.append({"risk":"insufficient_width", "score":0.9, "reason": f"Minimum width {min_w} mm (<4.5 mm)."})
        elif min_w < 5.0:
            risks.append({"risk":"borderline_width", "score":0.55, "reason": f"Minimum width {min_w} mm (4.5-5.0 mm borderline)."})
    # Canal/sinus proximity
    if distances:
        min_d = min(distances)
        if min_d < 2.0:
            risks.append({"risk":"critical_proximity", "score":0.95, "reason": f"Critical structure distance {min_d} mm (<2 mm)."})
        elif min_d < 3.0:
            risks.append({"risk":"close_proximity", "score":0.6, "reason": f"Structure distance {min_d} mm (2-3 mm)."})
    # keyword-based risks
    kw_map = {
        "fracture": ("possible_fracture", 0.9),
        "cyst": ("cystic_lesion", 0.8),
        "periapical": ("periapical_pathology", 0.7),
        "dehiscence": ("cortical_dehiscence",0.8),
        "sinus": ("sinus_proximity",0.5)
    }
    for kw, (rid, score) in kw_map.items():
        if kw in text.lower():
            risks.append({"risk": rid, "score": score, "reason": f"Keyword '{kw}' found in report."})

    # If no explicit risks found, add a low baseline confidence
    if not risks:
        risks.append({"risk": "low_risk_detected", "score": 0.15, "reason": "No high-risk numeric or keyword indicators found."})

    # Normalize scores (optional, keep as-is)
    for r in risks:
        r["score"] = round(float(r["score"]), 3)

    return risks
