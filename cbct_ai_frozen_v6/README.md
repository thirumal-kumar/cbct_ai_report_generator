[README.md](https://github.com/user-attachments/files/23746868/README.md)
# 🦷 CBCT AI Report Generator  
Automated CBCT DICOM Analysis + Radiologist-Style Reporting  
_Backend: FastAPI (SSE Streaming) • Frontend: Streamlit UI_

---

## 📌 Overview  
**CBCT AI Report Generator** is an offline, lightweight system designed to:

- Load CBCT **DICOM (`.dcm`) or ZIP DICOM folders**
- Display **axial, coronal, sagittal previews**
- Perform **essential radiographic measurements**
- Stream progress in real-time to the UI using **Server-Sent Events (SSE)**
- Produce structured **radiologist-style text reports**
- Export reports in **PDF** and **DOCX** formats

This tool is built for clinicians, radiologists, and researchers who require **fast CBCT assessment without cloud dependencies**.

---

## ✨ Key Features

### 🔍 1. Robust CBCT DICOM Loader
- Supports **multiframe CBCT** (Carestream, Planmeca, etc.)
- Extracts voxel metadata (slice thickness, SOP class)
- Saves 3-plane previews (axial/coronal/sagittal)

### 📊 2. Automated Measurements
Includes rule-based, deterministic estimations:

- HU mean / min / max  
- ROI HU sampling  
- Ridge height detection  
- Periapical low-density candidates  
- Measurement warnings for missing metadata  

### 🧠 3. Condition Detector
Suggests case type:

- **Full skull**
- **Maxilla**
- **Mandible**

(Used only to shape the reporting layout.)

### 📝 4. Structured Radiology Report
Outputs clinically formatted sections:

- **Scan Details**  
- **Image Quality**
- **Teeth & Periapical Evaluation**
- **Sinuses / TMJ / Canal / Airway**
- **Bone & Ridge Observations**
- **Impression**
- **Recommendations**

### ⚡ 5. True Streaming Pipeline
The backend streams each stage:

- `uploaded`
- `loading_cbct`
- `loaded_cbct`
- `detector`
- `measurements`
- `report_structured`
- `docx_ready`
- `pdf_ready`
- `complete`

No silent processing — ideal for clinical transparency.

### 📄 6. Exportable Reports
Reports automatically saved to:

```
static/results/<FILENAME>_report.pdf
static/results/<FILENAME>_report.docx
```

---

## 🧱 Project Structure

```
cbct_ai_report_generator/
│
├── app_agentic_stream.py      # FastAPI backend (SSE enabled)
├── chat_ui_v4_stream.py       # Streamlit frontend
├── dicom_reader.py            # CBCT volume loader
├── cbct_measurements.py       # Measurement heuristics
├── condition_detector.py      # Basic condition classifier
├── report_builder.py          # Assembles structured output
│
├── static/
│   ├── previews/              # Generated preview PNGs
│   └── results/               # DOCX/PDF final reports
│
├── samples/                   # Small sample case (optional)
└── README.md
```

---

## 🚀 Installation

### 1. Create Conda environment
```bash
conda create -n cbct_env python=3.11
conda activate cbct_env
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the App

### **Start backend (FastAPI):**
```bash
uvicorn app_agentic_stream:app --host 127.0.0.1 --port 8000 --reload
```

### **Start frontend (Streamlit):**
```bash
streamlit run chat_ui_v4_stream.py
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|---------|--------|-------------|
| `/generate_report_stream/` | POST | Upload CBCT & stream step-by-step progress |
| `/list_conditions` | GET | Returns selectable case types |

---

## 📄 Output Report Format

Each report includes:

- Modality + metadata  
- Image quality summary  
- Structured findings  
- Clinically useful impression  
- Follow-up recommendations  
- Radiologist signature block  

---

## 🧭 Roadmap  
Planned enhancements:

- [ ] 3D segmentation of teeth / canal / sinus  
- [ ] Accurate periapical lesion detection  
- [ ] Implant planning with distance maps  
- [ ] Guided surgical planning export  
- [ ] LLM-augmented clinical summarizer  

---

## 🤝 Contributing  
PRs are welcome.  
For large changes, open an issue first to discuss your ideas.

---

## 📜 License  
MIT License.  
⚠️ *This tool is intended for research and educational purposes only.*  
