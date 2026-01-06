# app.py — CBCT AI Report Generator (STABLE v6.0, FAISS-FREE)

import os, io, json, glob, zipfile, tempfile
import numpy as np
import pydicom

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI

from pdf_generator import generate_pdf


# ======================================================
# ENV
# ======================================================
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "meta-llama/llama-3.1-70b-instruct"
)

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY missing in .env")

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)


# ======================================================
# APP
# ======================================================
app = FastAPI(
    title="CBCT AI Report Generator",
    version="6.0 (Stable, FAISS-free)"
)


# ======================================================
# EMBEDDINGS + EVIDENCE
# ======================================================
embedder = SentenceTransformer("all-mpnet-base-v2")

RAG_DB_DIR = "rag_db"
EVIDENCE_TEXTS = []
EVIDENCE_FILES = []

for p in sorted(glob.glob(os.path.join(RAG_DB_DIR, "*.txt"))):
    txt = open(p, "r", encoding="utf-8").read().strip()
    if txt:
        EVIDENCE_TEXTS.append(txt)
        EVIDENCE_FILES.append(os.path.basename(p))


# ======================================================
# DICOM HELPERS
# ======================================================
def extract_dicom_metadata(path):
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    except Exception:
        return {"Modality": "Unknown"}

    def s(v):
        return str(v) if v is not None else None

    return {
        "PatientID": s(getattr(ds, "PatientID", None)),
        "PatientAge": s(getattr(ds, "PatientAge", None)),
        "PatientSex": s(getattr(ds, "PatientSex", None)),
        "Modality": s(getattr(ds, "Modality", "CBCT")),
        "Manufacturer": s(getattr(ds, "Manufacturer", None)),
        "Rows": s(getattr(ds, "Rows", None)),
        "Columns": s(getattr(ds, "Columns", None)),
    }


def save_upload(data, name):
    ext = name.lower().split(".")[-1]

    if ext == "dcm":
        f = tempfile.NamedTemporaryFile(delete=False, suffix=".dcm")
        f.write(data); f.close()
        return f.name

    if ext == "zip":
        d = tempfile.mkdtemp()
        zp = os.path.join(d, "u.zip")
        open(zp, "wb").write(data)
        with zipfile.ZipFile(zp) as z:
            z.extractall(d)
        for r, _, fs in os.walk(d):
            for fn in fs:
                if fn.lower().endswith(".dcm"):
                    return os.path.join(r, fn)
        raise HTTPException(400, "ZIP has no DICOM")

    if ext == "pdf":
        f = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        f.write(data); f.close()
        return f.name

    raise HTTPException(400, "Unsupported file type")


# ======================================================
# API MODELS
# ======================================================
class ReportResponse(BaseModel):
    metadata: dict
    retrieved_files: list
    final_report: str


# ======================================================
# MAIN ENDPOINT
# ======================================================
@app.post("/generate_report/", response_model=ReportResponse)
async def generate_report(
    file: UploadFile = File(...),
    tight: bool = True
):
    raw = await file.read()
    path = save_upload(raw, file.filename)

    metadata = extract_dicom_metadata(path)

    # Evidence selection (simple + robust)
    retrieved_files = EVIDENCE_FILES[:3]
    evidence_text = "\n\n".join(EVIDENCE_TEXTS[:3])

    prompt = (
        "You are a conservative dental CBCT reporting assistant.\n\n"
        f"METADATA:\n{json.dumps(metadata, indent=2)}\n\n"
        f"EVIDENCE:\n{evidence_text}\n\n"
        "Generate a concise CBCT report. "
        "Do NOT invent findings or measurements."
    )

    try:
        r = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=600
        )
        report = r.choices[0].message.content
    except Exception as e:
        raise HTTPException(500, f"LLM failure: {e}")

    return ReportResponse(
        metadata=metadata,
        retrieved_files=retrieved_files,
        final_report=report
    )


# ======================================================
# PDF
# ======================================================
@app.post("/generate_pdf/")
async def generate_pdf_api(data: dict = Body(...)):
    pdf = generate_pdf(
        report_text=data.get("final_report", ""),
        metadata=data.get("metadata", {}),
        case_type="CBCT",
        retrieval_confidence=None,
        retrieved_files=data.get("retrieved_files", []),
        warnings=[]
    )
    return StreamingResponse(
        io.BytesIO(pdf),
        media_type="application/pdf"
    )

