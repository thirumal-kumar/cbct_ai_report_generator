# chat_ui_v4_stream.py
import streamlit as st
import requests
import json
import time
import os
from pathlib import Path

st.set_page_config(page_title="CBCT AI Report Generator — Streaming", layout="wide")

# --- IMPORTANT: backend default port is 8000 (FastAPI / uvicorn) ---
API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")
API_STREAM = f"{API_BASE}/generate_report_stream/"

st.title("🦷 CBCT AI Report Generator — Streaming (Patched)")
st.markdown(
    """
Upload CBCT DICOM (.dcm), ZIP (DICOM folder), or PDF test case.  
The system will stream progress logs step-by-step and generate a radiologist-style report.
"""
)

# Condition selector (optional)
conditions = st.multiselect(
    "Radiologist-selected conditions (Optional):",
    options=[
        "Full Skull",
        "Maxilla",
        "Mandible",
        "Sinus",
        "TMJ",
        "Implant Assessment",
        "Airway",
        "Cyst",
        "Impacted",
        "Periapical",
    ],
)

uploaded_file = st.file_uploader(
    "Choose CBCT DICOM (.dcm), ZIP folder, or PDF test file",
    type=["dcm", "zip", "pdf"],
    help="Limit 200MB per file • DCM, ZIP, PDF",
)

start_btn = st.button("Generate AI Report (Streaming)", use_container_width=True)

# helpers
def _safe_json_load(s: str):
    if not s:
        return None
    s = s.strip()
    # Expect SSE style: data: {...}
    if s.startswith("data:"):
        s = s[len("data:"):].strip()
    # Try to parse JSON, otherwise try to find the first { and parse
    try:
        return json.loads(s)
    except Exception:
        idx = s.find("{")
        if idx >= 0:
            try:
                return json.loads(s[idx:])
            except Exception:
                return None
        return None

def _get_event_payload(msg: dict):
    if not isinstance(msg, dict):
        return None, {}
    event = msg.get("event") or msg.get("type") or msg.get("stage") or None
    payload = msg.get("data") or msg.get("payload") or {}
    if payload is None:
        payload = {}
    return event, payload

def _make_static_url(api_base: str, path: str):
    if not path:
        return None
    # If backend already returned a url, prefer it
    if isinstance(path, str) and path.startswith("http"):
        return path
    fname = os.path.basename(path)
    return f"{api_base.rstrip('/')}/static/results/{fname}"

# Main action
if start_btn:
    if uploaded_file is None:
        st.error("Please upload a file first.")
        st.stop()

    # UI placeholders
    progress = st.empty()
    logs = st.empty()
    report_area = st.empty()
    downloads_area = st.empty()

    progress.info("⏳ Starting...")

    log_lines = []
    final_report_text = None
    docx_url = None
    pdf_url = None
    stream_error = False

    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
    data = {"conditions": json.dumps(conditions)}

    # Use requests to POST and stream
    try:
        with requests.post(API_STREAM, files=files, data=data, stream=True, timeout=1800) as resp:
            if resp.status_code != 200:
                st.error(f"Server returned status {resp.status_code}: {resp.text[:200]}")
                st.stop()

            # iterate lines
            for raw in resp.iter_lines(decode_unicode=True):
                # skip keepalive/empty lines
                if not raw:
                    continue
                raw = raw.strip()
                # parse SSE "data: {...}" lines
                msg = _safe_json_load(raw)
                if msg is None:
                    # unknown raw content -> log it
                    log_lines.append(f"[{time.strftime('%H:%M:%S')}] RAW: {raw}")
                    logs.code("\n".join(log_lines[-800:]))
                    continue

                event, payload = _get_event_payload(msg)
                ev_name = (event or "unknown").lower()

                # Append human-friendly log
                try:
                    pl_summary = json.dumps(payload, ensure_ascii=False)
                except Exception:
                    pl_summary = str(payload)
                log_lines.append(f"[{time.strftime('%H:%M:%S')}] {ev_name}: {pl_summary}")
                logs.code("\n".join(log_lines[-800:]))

                # Update progress UI based on event
                if ev_name in ("uploaded",):
                    progress.success(f"Uploaded: {payload.get('filename') or uploaded_file.name}")
                elif ev_name.startswith("loading"):
                    progress.info("Decoding DICOM/CBCT...")
                elif ev_name == "loaded_cbct":
                    progress.success("CBCT loaded.")
                elif ev_name in ("detector", "detector_done"):
                    progress.info("Running condition detector...")
                elif ev_name == "case_type_selection":
                    progress.info(f"Case types selected: {payload.get('conditions')}")
                elif ev_name in ("measurements", "measurements_done"):
                    progress.info("Computing conservative clinically relevant measurements...")
                elif ev_name == "analyzer_start":
                    progress.info(payload.get("message", "Analyzer starting..."))
                elif ev_name == "analyzer_done":
                    progress.success("Analyzer finished.")
                elif ev_name == "analysis_json_ready":
                    url = payload.get("url") or _make_static_url(API_BASE, payload.get("path"))
                    log_lines.append(f"[{time.strftime('%H:%M:%S')}] Analysis JSON: {url}")
                    logs.code("\n".join(log_lines[-800:]))
                elif ev_name == "analysis_image_ready":
                    url = payload.get("url") or _make_static_url(API_BASE, payload.get("path"))
                    log_lines.append(f"[{time.strftime('%H:%M:%S')}] Image {payload.get('name')}: {url}")
                    logs.code("\n".join(log_lines[-800:]))
                elif ev_name == "final_report":
                    final_report_text = payload.get("report_text") or payload.get("text") or final_report_text
                    progress.success("Final report received.")
                elif ev_name == "docx_ready":
                    docx_url = payload.get("url") or _make_static_url(API_BASE, payload.get("path"))
                    log_lines.append(f"[{time.strftime('%H:%M:%S')}] DOCX saved at: {docx_url}")
                    logs.code("\n".join(log_lines[-800:]))
                elif ev_name == "pdf_ready":
                    pdf_url = payload.get("url") or _make_static_url(API_BASE, payload.get("path"))
                    log_lines.append(f"[{time.strftime('%H:%M:%S')}] PDF saved at: {pdf_url}")
                    logs.code("\n".join(log_lines[-800:]))
                elif ev_name == "error":
                    stream_error = True
                    err_msg = payload.get("message") or str(payload)
                    progress.error(f"Error: {err_msg}")
                    log_lines.append(f"[ERROR] {err_msg}")
                    logs.code("\n".join(log_lines[-800:]))
                elif ev_name in ("complete", "done"):
                    log_lines.append(f"[{time.strftime('%H:%M:%S')}] Stream completed.")
                    logs.code("\n".join(log_lines[-800:]))
                    break

    except requests.exceptions.ReadTimeout:
        progress.error("Connection timed out while streaming.")
        stream_error = True
    except Exception as e:
        progress.error(f"Streaming failed: {e}")
        log_lines.append(f"[EXCEPTION] {e}")
        logs.code("\n".join(log_lines[-800:]))
        stream_error = True

    # Show final report if present
    if final_report_text:
        report_area.title("Final Report")
        report_area.text_area("Report", final_report_text, height=520)

    # Provide download links (prefer backend-provided URLs)
    if docx_url or pdf_url:
        with downloads_area:
            st.markdown("### Downloads")
            if docx_url:
                st.markdown(f"- **DOCX:** [{os.path.basename(docx_url)}]({docx_url})")
            if pdf_url:
                st.markdown(f"- **PDF:** [{os.path.basename(pdf_url)}]({pdf_url})")
            if docx_url:
                st.write("DOCX (backend):")
                st.code(docx_url)
            if pdf_url:
                st.write("PDF (backend):")
                st.code(pdf_url)
    else:
        if stream_error:
            st.warning("Streaming ended with errors; no download links available.")
        else:
            st.info("Report generated, but download paths were not returned by backend. Check backend logs.")

    progress.info("Streaming finished.")
