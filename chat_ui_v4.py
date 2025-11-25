# chat_ui_v4.py
import streamlit as st
import requests
import base64
import time
import json

API_URL = st.secrets.get("backend_url", "http://127.0.0.1:8000")
GEN_REPORT_ENDPOINT = f"{API_URL.rstrip('/')}/generate_report/"

st.set_page_config(page_title="CBCT AI Reporter", layout="wide")
st.title("🦷 CBCT AI Report Generator — Clinical Dashboard")

with st.sidebar:
    st.header("Upload Study")
    st.write("Upload CBCT DICOM (.dcm), ZIP (DICOM series) or PDF (test)")

file = st.file_uploader("Drop CBCT (DCM / ZIP / PDF)", type=["dcm", "zip", "pdf"])
status_box = st.empty()

def display_status(msg, level="info"):
    if level == "info":
        status_box.info(msg)
    elif level == "success":
        status_box.success(msg)
    elif level == "warning":
        status_box.warning(msg)
    else:
        status_box.error(msg)

if file is None:
    st.info("No file uploaded yet. Upload a CBCT study (single DICOM or ZIP) to start.")
else:
    st.write(f"Uploaded and saved: {file.name} ({len(file.getvalue())/1024/1024:.1f} MB)")
    if st.button("Generate Report"):
        # Clear previous
        display_status("Uploading study and generating evidence-aware report (this may take up to 3 minutes)...", "info")
        try:
            files = {"file": (file.name, file.getvalue())}
            # send with large timeout
            r = requests.post(GEN_REPORT_ENDPOINT, files=files, timeout=240)
            if r.status_code != 200:
                try:
                    err = r.json()
                    display_status(f"Report generation failed: {err}", "error")
                except Exception:
                    display_status(f"Report generation failed: HTTP {r.status_code} - {r.text}", "error")
            else:
                data = r.json()
                # present results
                st.subheader("AI-Generated Clinical Report")
                st.text_area("Report (radiologist-style)", value=data.get("report",""), height=420)
                st.subheader("Case Summary")
                st.write("Retrieval confidence:", data.get("retrieval_confidence", 0))
                st.write("Retrieved snippets:", data.get("retrieved_snippets_count", 0))
                if data.get("warnings"):
                    st.subheader("Warnings")
                    for w in data["warnings"]:
                        st.warning(w)
                # Export PDF button
                if st.button("Download PDF of Report"):
                    payload = {
                        "report": data.get("report",""),
                        "metadata": data.get("metadata", {}),
                        "cbct_summary": data.get("cbct_summary", {})
                    }
                    pdf_res = requests.post(f"{API_URL.rstrip('/')}/export_case/", json=payload, timeout=120)
                    if pdf_res.status_code == 200:
                        # This endpoint returns short indicator; fallback: create client-side PDF from text
                        # If server returned raw pdf bytes we could decode. For now show the text as downloadable file.
                        pdf_bytes = None
                        try:
                            # Try to use server PDF if present (not guaranteed)
                            pdf_json = pdf_res.json()
                            if "pdf_bytes_base64" in pdf_json:
                                display_status("PDF exported on server (short indicator).", "success")
                                st.download_button("Download server PDF (placeholder)", data=json.dumps(payload, indent=2), file_name="report.json", mime="application/json")
                            else:
                                st.download_button("Download Report (JSON)", data=json.dumps(payload, indent=2), file_name="report.json", mime="application/json")
                        except Exception:
                            st.download_button("Download Report (JSON)", data=json.dumps(payload, indent=2), file_name="report.json", mime="application/json")
                    else:
                        display_status("PDF generation failed.", "error")
                display_status("Report generated successfully.", "success")
        except requests.exceptions.ReadTimeout:
            display_status("QA call failed: read timeout. Backend may be busy. Try again or check logs.", "error")
        except requests.exceptions.ConnectionError:
            display_status("Failed to contact backend (connection error). Is the server running?", "error")
        except Exception as e:
            display_status(f"Unexpected error: {e}", "error")
