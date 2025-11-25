# pdf_generator.py (v4.1) — structured CBCT radiology PDF with logo + extra sections

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
)
import io
import os
from datetime import datetime


def generate_pdf(
    report_text: str,
    metadata: dict,
    case_type: str = "",
    retrieval_confidence: float = None,
    retrieved_files: list = None,
    warnings: list = None,
    logo_path: str = None
) -> bytes:

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=20,
        leftMargin=20,
        topMargin=30,
        bottomMargin=30
    )

    styles = getSampleStyleSheet()
    title_style = styles["Heading1"]
    title_style.fontSize = 16
    title_style.leading = 20

    header2 = styles["Heading2"]
    header3 = styles["Heading3"]
    normal = styles["Normal"]
    normal.fontSize = 10

    elems = []

    # ------------------------------------------------------------
    # HEADER: Logo + Title
    # ------------------------------------------------------------
    if logo_path and os.path.exists(logo_path):
        try:
            img = Image(logo_path, width=35 * mm, height=35 * mm)
            img.hAlign = "LEFT"
            elems.append(img)
        except Exception:
            pass

    elems.append(Paragraph("CBCT AI Report", title_style))
    elems.append(Spacer(1, 6))

    # ------------------------------------------------------------
    # PATIENT METADATA SECTION (table)
    # ------------------------------------------------------------
    meta_items = [
        ["Patient ID", metadata.get("PatientID", "")],
        ["Patient Name", metadata.get("PatientName", "")],
        ["Age / Sex", f"{metadata.get('PatientAge','')} / {metadata.get('PatientSex','')}"],
        ["Study Date", metadata.get("StudyDate","")],
        ["Scanner", metadata.get("ManufacturerModelName","")],
    ]
    t = Table(meta_items, colWidths=[90 * mm, 80 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (1,0), colors.whitesmoke),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("BOX", (0,0), (-1,-1), 0.25, colors.grey),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
    ]))
    elems.append(t)
    elems.append(Spacer(1, 10))

    # ------------------------------------------------------------
    # CASE TYPE SECTION
    # ------------------------------------------------------------
    elems.append(Paragraph("Case Type", header3))
    elems.append(Paragraph(case_type.capitalize() if case_type else "Unknown", normal))
    elems.append(Spacer(1, 6))

    # ------------------------------------------------------------
    # RETRIEVAL CONFIDENCE
    # ------------------------------------------------------------
    elems.append(Paragraph("Retrieval Confidence", header3))
    if retrieval_confidence is not None:
        elems.append(Paragraph(f"{retrieval_confidence:.3f}", normal))
    else:
        elems.append(Paragraph("N/A", normal))
    elems.append(Spacer(1, 6))

    # ------------------------------------------------------------
    # RETRIEVED GUIDELINE FILES (Top-K)
    # ------------------------------------------------------------
    elems.append(Paragraph("Retrieved Reference Files", header3))
    if retrieved_files:
        for f in retrieved_files:
            elems.append(Paragraph(f"• {f}", normal))
    else:
        elems.append(Paragraph("None", normal))
    elems.append(Spacer(1, 8))

    # ------------------------------------------------------------
    # WARNINGS / SAFETY FLAGS
    # ------------------------------------------------------------
    elems.append(Paragraph("Safety Warnings", header3))
    if warnings:
        for w in warnings:
            elems.append(Paragraph(f"• {w}", normal))
    else:
        elems.append(Paragraph("No warnings", normal))
    elems.append(Spacer(1, 12))

    # ------------------------------------------------------------
    # MAIN REPORT SECTION
    # ------------------------------------------------------------
    elems.append(Paragraph("AI-Generated Report", header2))
    elems.append(Spacer(1, 6))

    for para in report_text.split("\n\n"):
        elems.append(Paragraph(para.replace("\n", "<br/>"), normal))
        elems.append(Spacer(1, 4))

    elems.append(Spacer(1, 20))
    elems.append(Paragraph(
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        normal
    ))

    # Build PDF
    doc.build(elems)
    buffer.seek(0)
    return buffer.getvalue()
