# chat_ui.py
import streamlit as st
import requests
import base64
import json

FASTAPI_URL = "http://127.0.0.1:8000/qa/"

st.set_page_config(page_title="CBCT Dental Agent", layout="wide")

# --- Dental UI Theme ---
st.markdown("""
<style>
body { background-color: #f4faff; }
.chat-container {
    background: white;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #dbe7ff;
    margin-bottom: 10px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
.question-box {
    color: #004a8f;
    font-weight: 600;
}
.answer-box {
    margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)

st.title("🦷 CBCT Agent – Interactive Chat")

uploaded = st.file_uploader(
    "Upload CBCT DICOM (.dcm), ZIP, folder, or PDF",
    type=["dcm", "zip", "pdf"]
)

# Store file in session_state so user can ask multiple questions
if uploaded:
    st.session_state["uploaded_file"] = uploaded

# Ask question box
question = st.text_input("Ask any dental/CBCT question:")

if st.button("Ask") and question:
    if "uploaded_file" not in st.session_state:
        st.error("Please upload a CBCT file first.")
    else:
        file = st.session_state["uploaded_file"]

        # Prepare payload
        files = {"file": (file.name, file.getvalue())}

        # First send file to /generate_report OR store locally if agent needs path
        # For our agentic QA → send to FastAPI qa/ with default PDF path
        payload = {
            "question": question,
            "case_file_path": "/mnt/data/cbct_report.pdf"   # Safe default sample
        }

        res = requests.post(
            FASTAPI_URL,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"}
        )

        if res.status_code != 200:
            st.error(f"Error: {res.text}")
        else:
            data = res.json()
            st.markdown(
                f"""
                <div class="chat-container">
                    <div class="question-box">❓ {question}</div>
                    <div class="answer-box">💬 {data['answer']}</div>
                    <div style="font-size:12px;color:#999;">Confidence: {data['confidence']:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
