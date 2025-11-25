# app.py (v5.0) — CBCT RAG + Agentic (hardened, evidence-first, critic loop)
import os
import io
import glob
import json
import zipfile
import tempfile
import faiss
import numpy as np
import pydicom
from typing import List, Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from openai import OpenAI  # OpenRouter-compatible client (new SDK)
from pdf_generator import generate_pdf  # keep your upgraded pdf_generator.py

# -----------------------
# Load env & client
# -----------------------
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-70b-instruct")
OPENROUTER_MODEL_CLASSIFIER = os.getenv("OPENROUTER_MODEL_CLASSIFIER", "")
OPENROUTER_MODEL_CRITIC = os.getenv("OPENROUTER_MODEL_CRITIC", OPENROUTER_MODEL)

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY missing in .env")

# ✅ Correct domain
client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# -----------------------
# Settings & embedder
# -----------------------
RAG_DB_DIR = "rag_db"
FAISS_INDEX_FILE = "rag_index.faiss"
MAPPING_FILE = "rag_mapping.json"
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-mpnet-base-v2")
embedder = SentenceTransformer(EMBED_MODEL)

app = FastAPI(title="CBCT RAG Agentic (v5.0)", version="5.0")


# -----------------------
# Helpers: RAG index
# -----------------------
def build_index():
    texts, filenames, tags = [], [], []
    for path in sorted(glob.glob(f"{RAG_DB_DIR}/*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read().strip()
        if not txt:
            continue
        texts.append(txt)
        fname = os.path.basename(path)
        filenames.append(fname)
        tags.append(fname.lower())
    if not texts:
        raise RuntimeError("rag_db is empty — add guidelines & sample cases")
    embs = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    d = embs.shape[1]
    idx = faiss.IndexFlatL2(d)
    idx.add(embs.astype("float32"))
    faiss.write_index(idx, FAISS_INDEX_FILE)
    with open(MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump({"texts": texts, "filenames": filenames, "tags": tags}, f, indent=2)
    return idx, {"texts": texts, "filenames": filenames, "tags": tags}


def load_index():
    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(MAPPING_FILE):
        idx = faiss.read_index(FAISS_INDEX_FILE)
        with open(MAPPING_FILE, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        # ensure tags exist (backwards compatibility)
        if "tags" not in mapping:
            mapping["tags"] = [fn.lower() for fn in mapping.get("filenames", [])]
        return idx, mapping
    return build_index()


faiss_index, rag_mapping = load_index()


# -----------------------
# Safe DICOM metadata extraction
# -----------------------
def safe_val(v):
    if v is None:
        return None
    if isinstance(v, (int, float, str, bool)):
        return v
    try:
        return str(v)
    except Exception:
        return repr(v)


def extract_dicom_metadata(path: str) -> dict:
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    except Exception:
        return {
            "PatientID": os.path.basename(path),
            "PatientName": "Unknown",
            "PatientAge": None,
            "PatientSex": None,
            "Manufacturer": "Unknown",
            "ManufacturerModelName": "Unknown",
            "StudyDate": None,
            "Modality": "UNKNOWN"
        }

    return {
        "PatientID": safe_val(getattr(ds, "PatientID", "Unknown")),
        "PatientName": safe_val(getattr(ds, "PatientName", "")),
        "PatientAge": safe_val(getattr(ds, "PatientAge", "")),
        "PatientSex": safe_val(getattr(ds, "PatientSex", "")),
        "Manufacturer": safe_val(getattr(ds, "Manufacturer", "Unknown")),
        "ManufacturerModelName": safe_val(getattr(ds, "ManufacturerModelName", "Unknown")),
        "StudyDate": safe_val(getattr(ds, "StudyDate", "Unknown")),
        "Modality": safe_val(getattr(ds, "Modality", "CBCT")),
        "Rows": safe_val(getattr(ds, "Rows", None)),
        "Columns": safe_val(getattr(ds, "Columns", None)),
        "SliceThickness": safe_val(getattr(ds, "SliceThickness", None)),
        "PixelSpacing": safe_val(getattr(ds, "PixelSpacing", None)),
        "KVP": safe_val(getattr(ds, "KVP", None))
    }


# -----------------------
# File handling
# -----------------------
def save_and_detect_dicom(uploaded_bytes: bytes, filename: str):
    ext = filename.lower().split(".")[-1]
    if ext == "dcm":
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".dcm")
        tmp.write(uploaded_bytes); tmp.close()
        return tmp.name
    if ext == "zip":
        tmp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(tmp_dir, "upload.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_bytes)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmp_dir)
        for root, _, files in os.walk(tmp_dir):
            for fn in files:
                if fn.lower().endswith(".dcm"):
                    return os.path.join(root, fn)
        raise HTTPException(400, "ZIP contains no DICOM files")
    if ext == "pdf":
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(uploaded_bytes); tmp.close()
        return tmp.name
    raise HTTPException(400, "Upload must be .dcm, .zip or .pdf for testing")


# -----------------------
# Metadata tokens + pre-summary
# -----------------------
def build_metadata_tokens(metadata: dict, filename: str) -> str:
    tokens = []
    for k in ("ManufacturerModelName", "Modality", "SliceThickness", "PixelSpacing", "KVP"):
        v = metadata.get(k)
        if v:
            tokens.append(f"{k}:{v}")
    name = filename.lower()
    for kw in ("maxilla", "mandible", "implant", "endo", "endodontic", "impaction", "third", "wisdom", "orth"):
        if kw in name:
            tokens.append(kw)
    return " | ".join(tokens) if tokens else filename


def generate_pre_summary(metadata: dict, filename: str, retrieved_files: List[str], case_type: str) -> str:
    # short bullet summary to stabilize LLM
    parts = []
    parts.append(f"Detected case type (heuristic): {case_type}")
    parts.append(f"Scanner: {metadata.get('ManufacturerModelName', 'Unknown')}, Modality: {metadata.get('Modality', 'Unknown')}")
    if retrieved_files:
        parts.append(f"Top retrieved references: {', '.join(retrieved_files[:4])}")
    return " ; ".join(parts)


# -----------------------
# Case-type classifier (hybrid)
# -----------------------
def rule_based_case_type(metadata: dict, filename: str) -> str:
    name = filename.lower()
    if any(x in name for x in ("implant","edent","edentulous","site")):
        return "implant"
    if any(x in name for x in ("endo","endodontic","periapical","apex")):
        return "endodontic"
    if any(x in name for x in ("impaction","third","wisdom","38","48")):
        return "impaction"
    try:
        rows = int(metadata.get("Rows") or 0)
        cols = int(metadata.get("Columns") or 0)
        if rows*cols < 500*500 and rows*cols>0:
            return "endodontic"
    except Exception:
        pass
    return "general"


def hybrid_case_type(metadata: dict, filename: str, retrieved_files: List[str]) -> str:
    # rule-based
    base = rule_based_case_type(metadata, filename)
    # retrieval signal: if retrieved files contain many 'implant' tokens -> override
    tags = " ".join([f.lower() for f in retrieved_files])
    if "implant" in tags and base != "endodontic":
        return "implant"
    if "endodontic" in tags or "endo" in tags:
        return "endodontic"
    # fallback optional LLM confirmation (if model provided)
    if OPENROUTER_MODEL_CLASSIFIER:
        q = (
            "Given this short summary and retrieved reference filenames, "
            "classify the case into one label: endodontic, implant, impaction, orthodontic, general.\n\n"
            f"SUMMARY: Scanner={metadata.get('ManufacturerModelName')}; Modality={metadata.get('Modality')}\n"
            f"FILES: {', '.join(retrieved_files[:6])}\n\nReturn single label only."
        )
        try:
            r = client.chat.completions.create(
                model=OPENROUTER_MODEL_CLASSIFIER,
                messages=[{"role":"user","content":q}],
                temperature=0.0,
                max_tokens=6
            )
            lab = r.choices[0].message.content.strip().lower()
            if lab in ("endodontic","implant","impaction","orthodontic","general"):
                return lab
        except Exception:
            pass
    return base


# -----------------------
# RAG retrieval (filtered)
# -----------------------
def retrieve_similar(query: str, top_k: int = 4) -> Tuple[List[str], List[str], List[float]]:
    q = embedder.encode([query], convert_to_numpy=True).astype("float32")
    D, I = faiss_index.search(q, top_k*3)
    texts, files, scores = [], [], []
    for idx, dist in zip(I[0], D[0]):
        if idx < 0: continue
        texts.append(rag_mapping["texts"][idx])
        files.append(rag_mapping["filenames"][idx])
        scores.append(float(dist))
        if len(texts) >= top_k: break
    return texts, files, scores


# -----------------------
# Evidence mapping helper
# -----------------------
def make_evidence_map(retrieved_texts: List[str], retrieved_files: List[str]) -> List[dict]:
    evidence = []
    for txt, fn in zip(retrieved_texts, retrieved_files):
        snippet = txt[:400].strip().replace("\n"," ")
        evidence.append({"filename": fn, "snippet": snippet})
    return evidence


# -----------------------
# Prompt constructor with strict guardrails and confidence-tier modifiers
# -----------------------
def build_prompt_strict(retrieved_evidence: List[dict],
                        metadata: dict,
                        pre_summary: str,
                        case_type: str,
                        retrieval_conf: float,
                        request_tightness: str = "normal") -> str:
    """
    request_tightness: "normal" | "tight" (tight => strictest, minimal speculation)
    """
    # guardrails: do not hallucinate numeric measurements or anatomy not in evidence
    evidence_block = "\n\n".join([f"FILE: {e['filename']}\nSNIPPET: {e['snippet']}" for e in retrieved_evidence]) or "No retrieved evidence."
    guard = (
        "IMPORTANT GUARDRAILS (apply STRICTLY):\n"
        "- DO NOT INVENT OR FABRICATE ANY NUMERICAL MEASUREMENTS (mm, degrees, cm). If measurements are not explicitly present in the retrieved evidence or metadata, state 'Exact dimensions not available from provided data'.\n"
        "- DO NOT ASSERT PRESENCE OF ANATOMY/STRUCTURES/LESIONS unless clearly supported in the retrieved evidence. If uncertain, state 'limited by available data'.\n"
        "- CITE the filenames used for each claim (list filenames in Evidence used section).\n"
        "- Use conservative clinical language; recommend clinical/radiologist review when uncertain.\n"
    )
    # confidence tier modifier
    if retrieval_conf is None:
        retrieval_conf = 0.0
    if retrieval_conf < 0.4:
        tier_instr = "Retrieval confidence is LOW (<0.4). Limit the report to high-level observations only and recommend human review. Do not make diagnostic judgments."
    elif retrieval_conf < 0.7:
        tier_instr = "Retrieval confidence is MODERATE (0.4-0.7). Be cautious: present findings with qualifiers and recommend confirmatory review."
    else:
        tier_instr = "Retrieval confidence is HIGH (>=0.7). You may present detailed, evidence-based findings but still avoid fabricating numbers."

    tight_note = ""
    if request_tightness == "tight":
        tight_note = "This request is set to TIGHT mode: be especially conservative, avoid any speculative language, avoid numeric estimates."

    prompt = (
        f"You are a conservative dental CBCT reporting assistant.\n\n"
        f"{guard}\n\n"
        f"{tier_instr}\n\n"
        f"{tight_note}\n\n"
        f"PRE-SUMMARY: {pre_summary}\n\n"
        f"RETRIEVED EVIDENCE:\n{evidence_block}\n\n"
        f"PATIENT METADATA:\n{json.dumps(metadata, indent=2)}\n\n"
        f"CASE TYPE: {case_type}\n\n"
        "TASK:\n"
        "Produce a concise structured CBCT report with sections:\n"
        "1) Study Type\n2) Clinical Question\n3) Key Findings (bulleted, evidence-backed)\n"
        "4) Impression(s) — numbered, conservative language\n"
        "5) Recommendations (if any)\n\n"
        "- Add an 'Evidence used' section listing filenames used.\n"
        "- Add a final short 'Confidence statement' (one sentence) describing overall certainty.\n"
        "- MAX 350 words.\n"
        "- If you cannot support a claim, say so.\n"
    )
    return prompt


# -----------------------
# LLM call wrapper
# -----------------------
def call_openrouter_model(model: str, system: str, user: str, max_tokens: int = 800, temperature: float = 0.0):
    try:
        res = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return res.choices[0].message.content, res
    except Exception as e:
        raise RuntimeError(f"OpenRouter API Error: {e}")


# -----------------------
# Consistency checker (critic) — second-pass LLM
# -----------------------
def consistency_check_and_repair(initial_report: str, metadata: dict, evidence: List[dict], case_type: str, retrieval_conf: float) -> Tuple[str, str]:
    """
    Returns (critic_notes, repaired_report_or_original)
    The critic checks for hallucinations and contradictions and attempts to produce a corrected report (or notes).
    """
    evidence_block = "\n\n".join([f"{e['filename']}: {e['snippet']}" for e in evidence]) or "No retrieved evidence"
    critic_prompt = (
        "You are a medical QA assistant (critic). Review the REPORT below for any of the following issues:\n"
        "- Fabricated numeric measurements (mm/cm/degrees) not present in evidence\n"
        "- Statements about anatomy/lesions not supported by evidence\n"
        "- Self-contradictions or unsupported definitive diagnoses\n"
        "- Missing 'Evidence used' section\n\n"
        "If issues are found, list them clearly (bullet list). Then produce a corrected report that removes fabricated items and inserts conservative qualifiers. "
        "If no issues are found, reply 'OK' as the first line and then restate the report unchanged.\n\n"
        f"EVIDENCE:\n{evidence_block}\n\n"
        f"METADATA:\n{json.dumps(metadata, indent=2)}\n\n"
        f"CASE TYPE: {case_type}\n\n"
        "REPORT TO REVIEW:\n"
        f"{initial_report}\n\n"
        "Return JSON with two keys: critic_notes (string) and repaired_report (string)."
    )
    try:
        content, raw = call_openrouter_model(OPENROUTER_MODEL_CRITIC, "You are an expert medical critic.", critic_prompt, max_tokens=600, temperature=0.0)
        # attempt to parse JSON from model; if model returns JSON, parse; otherwise, wrap as critic notes
        try:
            parsed = json.loads(content)
            critic_notes = parsed.get("critic_notes", "")
            repaired = parsed.get("repaired_report", initial_report)
        except Exception:
            # not JSON -> put the whole content into critic_notes and return original
            critic_notes = content
            # if critic content begins with OK, keep original; if it includes "corrected report", try to extract
            if content.strip().lower().startswith("ok"):
                repaired = initial_report
            else:
                # fallback: ask model to produce a corrected version by appending instruction
                repair_prompt = critic_prompt + "\n\nNow produce only the corrected_report string (no JSON)."
                repair_content, _ = call_openrouter_model(OPENROUTER_MODEL_CRITIC, "Repair assistant", repair_prompt, max_tokens=600, temperature=0.0)
                repaired = repair_content if repair_content else initial_report
        return critic_notes, repaired
    except Exception as e:
        return f"Critic failed: {e}", initial_report


# -----------------------
# Safety checks (pre-report)
# -----------------------
def safety_checks(metadata: dict, dicom_path: str):
    warnings = []
    mod = str(metadata.get("Modality", "")).upper()
    if mod not in ("CT","CBCT","CBCT"):
        warnings.append("Modality is not clearly CT/CBCT. Verify study.")
    # check series size
    dn = os.path.dirname(dicom_path)
    if os.path.isdir(dn):
        cnt = sum(1 for _,_,files in os.walk(dn) for fn in files if fn.lower().endswith(".dcm"))
        if cnt > 2000:
            warnings.append(f"Large series detected ({cnt} slices). Consider subsampling.")
    return warnings


# -----------------------
# Response model (v5)
# -----------------------
class ReportResponseV5(BaseModel):
    metadata: dict
    case_type: str
    pre_summary: str
    evidence_used: list
    retrieved_files: list
    retrieval_scores: list
    retrieval_confidence: float
    warnings: list
    report: str            # initial draft
    critic_notes: str      # critic output or ''
    final_report: str      # repaired final (may equal initial if critic OK)


# -----------------------
# Main endpoint
# -----------------------
@app.post("/generate_report/", response_model=ReportResponseV5)
async def generate_report(file: UploadFile = File(...), tight: bool = False):
    """
    Generate a CBCT report.
    Query param tight (bool) - if True, forces the tightest no-speculation mode.
    """
    uploaded = await file.read()
    dicom_path = save_and_detect_dicom(uploaded, file.filename)

    metadata = extract_dicom_metadata(dicom_path)
    summary_tokens = build_metadata_tokens(metadata, file.filename)

    # quick retrieval to help classification
    retrieved_texts, retrieved_files, scores = retrieve_similar(summary_tokens, top_k=6)
    # hybrid case type using retrieval signal
    case_type = hybrid_case_type(metadata, file.filename, retrieved_files)

    pre_summary = generate_pre_summary(metadata, file.filename, retrieved_files, case_type)
    warnings = safety_checks(metadata, dicom_path)

    # evidence mapping for top-k (limit 4)
    top_texts = retrieved_texts[:4]
    top_files = retrieved_files[:4]
    top_scores = scores[:4]
    evidence = make_evidence_map(top_texts, top_files)

    # compute retrieval confidence (normalized)
    mean_dist = float(np.mean(top_scores)) if top_scores else 1.0
    retrieval_conf = max(0.0, 1.0 - (mean_dist / (mean_dist + 1.0)))

    # build strict prompt with tier + tightness
    tightness = "tight" if tight else "normal"
    prompt = build_prompt_strict(evidence, metadata, pre_summary, case_type, retrieval_conf, request_tightness=tightness)
    system_msg = "You are a conservative dental CBCT reporting assistant. Follow guardrails strictly."

    # LLM initial draft
    try:
        report_text, raw = call_openrouter_model(OPENROUTER_MODEL, system_msg, prompt)
    except Exception as e:
        raise HTTPException(500, f"OpenRouter error: {e}")

    # critic pass
    critic_notes, repaired = consistency_check_and_repair(report_text, metadata, evidence, case_type, retrieval_conf)

    # final safety: if retrieval_conf low and repaired contains excessive claims, ensure conservative
    # (the critic should already handle this)

    return ReportResponseV5(
        metadata=metadata,
        case_type=case_type,
        pre_summary=pre_summary,
        evidence_used=evidence,
        retrieved_files=top_files,
        retrieval_scores=top_scores,
        retrieval_confidence=round(float(retrieval_conf), 3),
        warnings=warnings,
        report=report_text,
        critic_notes=critic_notes,
        final_report=repaired
    )


# -----------------------
# PDF endpoint (reuses generate_pdf - updated earlier)
# -----------------------
@app.post("/generate_pdf/")
async def generate_pdf_api(data: dict = Body(...)):
    """
    Expects keys: report (final_report preferred), metadata, case_type, retrieval_confidence, retrieved_files, warnings, pre_summary, critic_notes, evidence_used
    """
    report = data.get("final_report") or data.get("report") or ""
    metadata = data.get("metadata", {})
    case_type = data.get("case_type", "")
    retrieval_conf = data.get("retrieval_confidence", None)
    retrieved_files = data.get("retrieved_files", [])
    warnings = data.get("warnings", [])
    pre_summary = data.get("pre_summary", "")
    critic_notes = data.get("critic_notes", "")

    # default logo from uploaded image if present
    logo_candidate = "/mnt/data/38545cc0-a030-42c3-91b1-7a4d30faefe9.png"
    logo_path = logo_candidate if os.path.exists(logo_candidate) else None

    # call pdf generator (it will accept extra fields; update it to show pre_summary/critic if you like)
    pdf_bytes = generate_pdf(
        report_text=report,
        metadata=metadata,
        case_type=case_type,
        retrieval_confidence=retrieval_conf,
        retrieved_files=retrieved_files,
        warnings=warnings,
        logo_path=logo_path
    )

    return StreamingResponse(io.BytesIO(pdf_bytes), media_type="application/pdf")


# -----------------------
# Admin: rebuild index
# -----------------------
@app.post("/admin/rebuild_index")
def rebuild_index():
    global faiss_index, rag_mapping
    faiss_index, rag_mapping = build_index()
    return {"status": "ok", "count": len(rag_mapping["texts"])}
