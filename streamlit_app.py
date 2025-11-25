# streamlit_app.py (v5.1) — Clean Dental Dashboard (Professional Radiology UI)
import streamlit as st
import requests
import json
import base64
import os
from PIL import Image
from io import BytesIO

# ---- CONFIG ----
FASTAPI_URL = "http://127.0.0.1:8000"
st.set_page_config(page_title="CBCT AI Reporter — Clinical Dashboard", layout="wide", initial_sidebar_state="collapsed")

# Local preview image path (from uploaded container files)
# Developer note: this path was provided from uploaded assets in the session.
PREVIEW_FILE = "/mnt/data/a70640e4-a5cc-40d5-8583-8598c446440b.png"

# Color palette
PALETTE = {
    "bg": "#F8FAFC",            # dental white
    "card_bg": "#FFFFFF",
    "primary": "#1E3A8A",       # royal blue
    "accent": "#0D9488",        # teal/aqua
    "implant": "#059669",       # green
    "endodontic": "#EA580C",    # orange
    "general": "#2563EB",       # blue
    "impaction": "#7C3AED",     # purple
    "warning": "#FACC15",       # yellow
    "danger": "#DC2626",        # red
    "muted": "#6B7280"
}

# Helper: case-type -> badge color
CASE_COLORS = {
    "implant": PALETTE["implant"],
    "endodontic": PALETTE["endodontic"],
    "impaction": PALETTE["impaction"],
    "orthodontic": PALETTE["accent"],
    "general": PALETTE["general"]
}

# ---- STYLES ----
st.markdown(
    f"""
    <style>
    .page-background {{ background: {PALETTE['bg']}; }}
    .card {{
        background: {PALETTE['card_bg']};
        border-radius: 8px;
        padding: 18px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        margin-bottom: 18px;
    }}
    .muted {{ color: {PALETTE['muted']}; }}
    .primary {{ color: {PALETTE['primary']}; font-weight:600; }}
    .badge {{
        display:inline-block;
        padding:6px 10px;
        border-radius: 999px;
        color: white;
        font-weight:600;
        font-size:0.9rem;
    }}
    .small {{ font-size:0.9rem; color:{PALETTE['muted']}; }}
    .conf-bar {{
        border-radius: 6px;
        height: 14px;
        width:100%;
        background: linear-gradient(90deg, rgba(37,99,235,0.15), rgba(13,148,136,0.15));
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- LAYOUT ----
st.markdown("<div class='page-background'></div>", unsafe_allow_html=True)
st.title("🦷 CBCT AI Report Generator — Clinical Dashboard")

col1, col2 = st.columns([1, 2.2], gap="large")

# ---- LEFT COLUMN: Upload + Metadata + Preview ----
with col1:
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Upload Study")
        uploaded_file = st.file_uploader("Upload CBCT DICOM (.dcm), ZIP (DICOM series) or PDF (test)", type=["dcm","zip","pdf"])
        tight_mode = st.checkbox("TIGHT MODE — no speculation (recommended)", value=False)
        st.markdown("</div>", unsafe_allow_html=True)

    # patient preview card
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Study Preview")
        if os.path.exists(PREVIEW_FILE):
            try:
                img = Image.open(PREVIEW_FILE)
                st.image(img, caption="Preview (first available slice / uploaded image)", use_column_width=True)
            except Exception:
                st.info("Preview file present but could not be rendered.")
        else:
            st.info("No preview image found. A DICOM-slice preview will appear here when available.")
        st.markdown("</div>", unsafe_allow_html=True)

    # quick actions
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Quick Actions")
        if st.button("Rebuild RAG Index (admin)"):
            try:
                r = requests.post(f"{FASTAPI_URL}/admin/rebuild_index", timeout=120)
                if r.status_code == 200:
                    st.success(f"RAG rebuilt: {r.json()}")
                else:
                    st.error(f"Failed: {r.status_code} — {r.text}")
            except Exception as e:
                st.error(f"Error: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

# ---- RIGHT COLUMN: Results & Report ----
with col2:
    # placeholder for output
    result_container = st.empty()

    if uploaded_file:
        # show processing card
        result_container.markdown("<div class='card'>", unsafe_allow_html=True)
        st.info("Uploading study and generating evidence-aware report (this may take 20-60s)...")
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            res = requests.post(f"{FASTAPI_URL}/generate_report/?tight={str(tight_mode).lower()}", files=files, timeout=180)

            if res.status_code != 200:
                st.error(f"Backend error: {res.status_code}\n{res.text}")
                result_container.markdown("</div>", unsafe_allow_html=True)
            else:
                data = res.json()
                # header
                st.markdown("<div style='display:flex; justify-content:space-between; align-items:center;'>", unsafe_allow_html=True)
                st.markdown(f"<div><h2 style='margin:0; color:{PALETTE['primary']}'>Case Summary</h2></div>", unsafe_allow_html=True)
                # case type badge
                case_type = (data.get("case_type") or "general").lower()
                badge_color = CASE_COLORS.get(case_type, PALETTE["general"])
                st.markdown(f"<div><span class='badge' style='background:{badge_color}'>{case_type.capitalize()}</span></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                # confidence meter
                conf = float(data.get("retrieval_confidence", 0.0))
                if conf >= 0.7:
                    conf_color = PALETTE["implant"]
                elif conf >= 0.4:
                    conf_color = PALETTE["warning"]
                else:
                    conf_color = PALETTE["danger"]

                st.write("")  # spacer
                st.write(f"**Retrieval confidence:** {conf:.3f}")
                # progress bar with color tile
                st.markdown(f"""
                    <div style="background:#E6EEF8; padding:8px; border-radius:8px;">
                      <div style="width:100%; background:#E5F6F1; border-radius:6px; height:12px;">
                        <div style="width:{conf*100}%; height:12px; background:{conf_color}; border-radius:6px;"></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Pre-summary + warnings
                st.subheader("Pre-summary")
                st.markdown(f"<div class='small'>{data.get('pre_summary','')}</div>", unsafe_allow_html=True)

                st.subheader("Warnings")
                if data.get("warnings"):
                    for w in data["warnings"]:
                        st.error(f"⚠️ {w}")
                else:
                    st.success("No warnings")

                # retrieved evidence (collapsible)
                with st.expander("Retrieved reference files & snippets (evidence)", expanded=False):
                    evidence = data.get("evidence_used", [])
                    if evidence:
                        for ev in evidence:
                            st.markdown(f"**{ev.get('filename')}**")
                            st.write(ev.get("snippet", "")[:600] + ("..." if len(ev.get("snippet","")) > 600 else ""))
                            st.markdown("---")
                    else:
                        st.write("No retrieved evidence files.")

                # Report area
                st.subheader("AI-Generated Clinical Report (Initial Draft)")
                st.text_area("Initial draft", value=data.get("report",""), height=220)

                # Critic notes
                st.subheader("Automated QA — Critic Notes")
                critic = data.get("critic_notes", "")
                if critic:
                    st.text_area("Critic notes", value=critic, height=140)
                else:
                    st.write("No critic notes (report passed automated QA).")

                # Final report
                st.subheader("Final Report (Post-critic)")
                final_report = data.get("final_report", data.get("report",""))
                st.text_area("Final report", value=final_report, height=300)

                # Download PDF button
                st.markdown("---")
                st.write("**Export**")
                if st.button("Generate & Download PDF"):
                    pdf_payload = json.dumps({
                        "final_report": data.get("final_report", final_report),
                        "metadata": data.get("metadata", {}),
                        "case_type": data.get("case_type",""),
                        "retrieval_confidence": data.get("retrieval_confidence", None),
                        "retrieved_files": data.get("retrieved_files", []),
                        "warnings": data.get("warnings", []),
                        "pre_summary": data.get("pre_summary",""),
                        "critic_notes": data.get("critic_notes",""),
                        "evidence_used": data.get("evidence_used", [])
                    })
                    pdf_res = requests.post(f"{FASTAPI_URL}/generate_pdf/", data=pdf_payload, headers={"Content-Type":"application/json"}, timeout=120)
                    if pdf_res.status_code == 200:
                        pdf_bytes = pdf_res.content
                        b64 = base64.b64encode(pdf_bytes).decode()
                        href = f'<a href="data:application/pdf;base64,{b64}" download="cbct_report_final.pdf">📥 Download Final PDF</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    else:
                        st.error(f"PDF generation failed: {pdf_res.status_code} {pdf_res.text}")

                # Regenerate in Tight Mode
                if st.button("Regenerate (TIGHT MODE)"):
                    st.info("Regenerating in tight mode (no speculation)...")
                    try:
                        regen = requests.post(f"{FASTAPI_URL}/generate_report/?tight=true", files=files, timeout=180)
                        if regen.status_code == 200:
                            rdata = regen.json()
                            st.success("Regenerated (TIGHT MODE) — review final report below.")
                            st.text_area("Tight final report", value=rdata.get("final_report",""), height=300)
                        else:
                            st.error(f"Regeneration failed: {regen.status_code} {regen.text}")
                    except Exception as e:
                        st.error(f"Regeneration error: {e}")

                st.markdown("</div>", unsafe_allow_html=True)
                result_container.empty()

        except Exception as e:
            st.error(f"Failed to contact backend: {e}")
            result_container.markdown("</div>", unsafe_allow_html=True)
