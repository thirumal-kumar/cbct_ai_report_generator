# report_formatter.py
"""
Generate radiologist-style plain-text CBCT reports from:
 - patient_info: dict (PatientName, PatientID, Age, Sex, StudyDate)
 - clinical_notes: str
 - cbct_summary: dict produced by pipeline_example.py / dicom_reader
 - retrieval_snippets: optional list[str] (contextual RAG references)
Return: (report_text:str, short_impression:str)
"""

from datetime import datetime
import math
import textwrap

def _fmt_patient_block(patient_info):
    lines = []
    lines.append(f"Patient Name : {patient_info.get('PatientName','-')}")
    if patient_info.get("PatientID"):
        lines.append(f"Patient ID   : {patient_info.get('PatientID')}")
    if patient_info.get("Age") or patient_info.get("Sex"):
        lines.append(f"Age/Sex      : {patient_info.get('Age','-')} / {patient_info.get('Sex','-')}")
    if patient_info.get("StudyDate"):
        lines.append(f"Study Date   : {patient_info.get('StudyDate')}")
    return "\n".join(lines)

def _summarize_hu_stats(stats):
    if not stats:
        return "HU stats not available."
    mean = stats.get("mean")
    std = stats.get("std")
    mn = stats.get("min")
    mx = stats.get("max")
    return f"Volume intensity (approx): mean={_safe_round(mean)} ; std={_safe_round(std)} ; range={_safe_round(mn)}–{_safe_round(mx)}"

def _safe_round(x, ndigits=1):
    try:
        if x is None:
            return "NA"
        return round(float(x), ndigits)
    except Exception:
        return "NA"

def _bone_height_phrase(height_mm):
    # conservative thresholds — tune later
    if height_mm is None:
        return "Bone height: not measured."
    if height_mm >= 10.0:
        return f"Bone height {height_mm:.1f} mm — adequate for standard implant placement."
    if 8.0 <= height_mm < 10.0:
        return f"Bone height {height_mm:.1f} mm — borderline; may require short implant or minor augmentation."
    return f"Bone height {height_mm:.1f} mm — inadequate; augmentation likely required."

def _bone_width_phrase(width_mm):
    if width_mm is None:
        return "Bone width: not measured."
    if width_mm >= 6.0:
        return f"Bone width {width_mm:.1f} mm — adequate for standard implant diameters."
    if 4.0 <= width_mm < 6.0:
        return f"Bone width {width_mm:.1f} mm — narrow; consider narrow-diameter implant or augmentation."
    return f"Bone width {width_mm:.1f} mm — inadequate for immediate implant without augmentation."

def _nerve_proximity_phrase(distance_mm):
    if distance_mm is None:
        return "Inferior alveolar nerve proximity: not measured."
    if distance_mm >= 3.0:
        return f"Inferior alveolar nerve clearance {distance_mm:.1f} mm — acceptable safety margin."
    if 1.0 <= distance_mm < 3.0:
        return f"Inferior alveolar nerve clearance {distance_mm:.1f} mm — close; caution advised."
    return f"Inferior alveolar nerve clearance {distance_mm:.1f} mm — unsafe for implant without nerve risk mitigation."

def _format_findings(cbct_summary, retrieval_snippets=None):
    # cbct_summary expected keys: shape, stats, thickness_examples, measurements dict...
    s = []
    shape = cbct_summary.get("shape")
    if shape:
        s.append(f"CT volume: {shape[0]} slices; in-plane {shape[1]}×{shape[2]} voxels.")
    stats = cbct_summary.get("stats")
    s.append(_summarize_hu_stats(stats))

    # measurements
    m = cbct_summary.get("measurements", {}) or {}
    # possible keys: global_bone_height_mm, global_bone_width_mm, canal_distance_mm
    if m.get("global_bone_height_mm") is not None:
        s.append(_bone_height_phrase(m.get("global_bone_height_mm")))
    if m.get("global_bone_width_mm") is not None:
        s.append(_bone_width_phrase(m.get("global_bone_width_mm")))
    if m.get("canal_distance_mm") is not None:
        s.append(_nerve_proximity_phrase(m.get("canal_distance_mm")))

    # thickness examples (voxel-based)
    t = cbct_summary.get("thickness_example")
    if t:
        if t.get("thickness_vox", 0) > 0:
            s.append(f"Example bone thickness measured along sample line: {t.get('thickness_vox')} voxels (voxel->mm conversion required).")
        else:
            s.append("Example bone thickness: not appreciable on sampled line.")

    # Append small RAG evidence if provided (concise)
    if retrieval_snippets:
        s.append("Reference evidence summary from archive: " + retrieval_snippets[0][:320].replace("\n"," "))

    return "\n".join(s)

def _derive_impression(cbct_summary):
    m = cbct_summary.get("measurements", {}) or {}
    # basic rule-based decision for impression
    height = m.get("global_bone_height_mm")
    width = m.get("global_bone_width_mm")
    canal = m.get("canal_distance_mm")

    # default impression
    if height is None and width is None and canal is None:
        return "Limited quantitative data available from CBCT; visual inspection recommended. Correlate with clinical findings."

    # simple scoring
    scores = []
    if height is not None:
        if height >= 10.0:
            scores.append(0)  # good
        elif height >= 8.0:
            scores.append(1)  # borderline
        else:
            scores.append(2)  # poor
    if width is not None:
        if width >= 6.0:
            scores.append(0)
        elif width >= 4.0:
            scores.append(1)
        else:
            scores.append(2)
    if canal is not None:
        if canal >= 3.0:
            scores.append(0)
        elif canal >= 1.0:
            scores.append(1)
        else:
            scores.append(2)

    # interpret worst score
    worst = max(scores) if scores else 1
    if worst == 0:
        return "Impression: Adequate bone volume for implant placement in the assessed regions. No immediate contraindication identified on CBCT. Correlate with clinical exam."
    if worst == 1:
        return "Impression: Borderline bone volume in one or more regions; consider narrow implant or minor augmentation. Exercise clinical caution."
    return "Impression: Inadequate bone volume and/or hazardous proximity to neurovascular structures — pre-implant augmentation or surgical planning required."

def _format_recommendations(cbct_summary):
    recs = []
    m = cbct_summary.get("measurements", {}) or {}
    height = m.get("global_bone_height_mm")
    width = m.get("global_bone_width_mm")
    canal = m.get("canal_distance_mm")

    if height is not None and height < 10:
        recs.append("Consider bone augmentation (crestal or lateral) or use of short/narrow implants for regions with <10 mm height.")
    if width is not None and width < 6:
        recs.append("Consider ridge augmentation or narrow-diameter implants for regions with <6 mm width.")
    if canal is not None and canal < 3:
        recs.append("Plan nerve mapping and consider staged approach if nerve proximity <3 mm.")
    if not recs:
        recs.append("No specific surgical modifications required based on current CBCT measurements; correlate clinically.")
    return "\n".join(f"- {r}" for r in recs)

def build_radiology_report(patient_info: dict,
                          clinical_notes: str,
                          cbct_summary: dict,
                          retrieval_snippets: list = None,
                          author_name: str = "CBCT AI Assistant") -> dict:
    """
    Returns:
      {
        "report_text": "...",
        "impression": "...",
        "recommendations": "...",
        "generated_on": "..."
      }
    """
    header = "CONE BEAM COMPUTED TOMOGRAPHY — REPORT"
    patient_block = _fmt_patient_block(patient_info)
    study_block = f"Region(s) scanned: {patient_info.get('Region','Not specified')}\nIndication: {patient_info.get('Indication','-')}\nReferring clinician: {patient_info.get('Referring','-')}"
    findings = _format_findings(cbct_summary, retrieval_snippets)
    impression = _derive_impression(cbct_summary)
    recommendations = _format_recommendations(cbct_summary)
    footer = f"Report generated by: {author_name} on {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n*** End of report ***"

    # assemble text
    parts = [
        header,
        "",
        patient_block,
        "",
        study_block,
        "",
        "CLINICAL NOTES:",
        clinical_notes or "-",
        "",
        "FINDINGS:",
        findings,
        "",
        "RADIOGRAPHIC IMPRESSION:",
        impression,
        "",
        "RECOMMENDATIONS:",
        recommendations,
        "",
        footer
    ]
    report_text = "\n".join(parts)

    # wrap text to reasonable width
    report_text = "\n".join(textwrap.fill(line, width=100) if len(line) > 120 else line for line in report_text.split("\n"))

    return {
        "report_text": report_text,
        "impression": impression,
        "recommendations": recommendations,
        "generated_on": datetime.utcnow().isoformat() + "Z"
    }
