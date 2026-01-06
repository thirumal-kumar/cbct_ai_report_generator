import os
import io
import json
import zipfile
import tempfile

import streamlit as st
import pydicom

from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

from pdf_generator import generate_pdf


# --------------------------------------------------
# ENV (local .env OR Streamlit secrets)
# --------------------------------------------------
load_dotenv()

OPENROUTER_API_KEY = (
    st.secrets.get("OPENROUTER_API_KEY")
    if "OPENROUTER_API_KEY" in st.secrets
    else os.getenv("OPENROUTER_API_KEY")
)

OPENROUTER_MODEL = (
    st.secrets.get("OPENROUTER_MODEL")
    if "OPENROUTER_MODEL" in st.secrets
    else os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-70b-instruct")
)

if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY is not configured.")
    st.stop()

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)


# --------------------------------------------------
# EVIDENCE
# --------------------------------------------------
RAG_DB_DIR = "rag_db"
EVIDENCE_TEXTS = []
EVIDENCE_FILES = []

for p in sorted(os.listdir(RAG_DB_DIR)):
    if p.endswith(".txt"):
        with open(os.path.join(RAG_DB_DIR, p), "r", encoding="utf-8") as f:
            txt = f.read().strip()
            if txt:
                EVIDENCE_TEXTS.append(txt)
                EVIDENCE_FILES.append(p)


# --------------------------------------------------
# DICOM HELPERS
# --------------------------------------------------
def extract_dicom_metadata(path):
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    except Exception:
        return {"Modality": "Unknown"}

    def s(v): return str(v) if v is not None else None

    return {
        "PatientID": s(getattr(ds, "PatientID", None)),
        "PatientAge": s(getattr(ds, "PatientAge", None)),
        "PatientSex": s(getattr(ds, "PatientSex", None)),
        "Modality": s(getattr(ds, "Modality", "CBCT")),
        "Manufacturer": s(getattr(ds, "Manufacturer", None)),
        "Rows": s(getattr(ds, "Rows", None)),
        "Columns": s(getattr(ds, "Columns", None)),
    }


def save_uploaded_file(uploaded_file):
    suffix = uploaded_file.name.lower().split(".")[-1]
    tmp = tempfile.mkdtemp()

    if suffix == "dcm":
        path = os.path.join(tmp, uploaded_file.name)
        open(path, "wb").write(uploaded_file.getvalue())
        return path

    if suffix == "zip":
        zpath = os.path.join(tmp, uploaded_file.name)
        open(zpath, "wb").write(uploaded_file.getvalue())
        with zipfile.ZipFile(zpath) as z:
            z.extractall(tmp)
        for root, _, files in os.walk(tmp):
            for f in files:
                if f.lower().endswith(".dcm"):
                    return os.path.join(root, f)

    if suffix == "pdf":
        path = os.path.join(tmp, uploaded_file.name)
        open(path, "wb").write(uploaded_file.getvalue())
        return path

    return None


# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------
st.set_page_config(
    page_title="CBCT AI Report Generator — Clinical Dashboard",
    layout="wide"
)

st.title("🦷 CBCT AI Report Generator — Clinical Dashboard")
st.subheader("Upload Study")

uploaded_file = st.file_uploader(
    "Upload CBCT DICOM (.dcm), ZIP (DICOM series), or PDF (test)",
    type=["dcm", "zip", "pdf"]
)

tight_mode = st.checkbox("TIGHT MODE — no speculation", value=True)

if uploaded_file:
    with st.spinner("Uploading study and generating evidence-aware report (this may take 20–60s)..."):
        dicom_path = save_uploaded_file(uploaded_file)

        if not dicom_path:
            st.error("Failed to process uploaded file.")
            st.stop()

        metadata = extract_dicom_metadata(dicom_path)

        evidence_text = "\n\n".join(EVIDENCE_TEXTS[:3])
        retrieved_files = EVIDENCE_FILES[:3]

        prompt = (
            "You are a conservative dental CBCT reporting assistant.\n\n"
            f"METADATA:\n{json.dumps(metadata, indent=2)}\n\n"
            f"EVIDENCE:\n{evidence_text}\n\n"
            "Generate a concise CBCT report. "
            "Do NOT invent findings."
        )

        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=600
        )

        report = response.choices[0].message.content

    st.success("Report generated successfully")

    st.subheader("Final CBCT Report")
    st.write(report)

    with st.expander("Patient / Study Metadata"):
        st.json(metadata)

    with st.expander("Evidence Sources Used"):
        for f in retrieved_files:
            st.write(f"- {f}")

    pdf_bytes = generate_pdf(
        report_text=report,
        metadata=metadata,
        case_type="CBCT",
        retrieval_confidence=None,
        retrieved_files=retrieved_files,
        warnings=[]
    )

    st.download_button(
        "Download PDF Report",
        data=pdf_bytes,
        file_name="CBCT_Report.pdf",
        mime="application/pdf"
    )
