import streamlit as st
import requests
import json
from pathlib import Path

# -----------------------
# CONFIG
# -----------------------
FASTAPI_QA_URL = "http://127.0.0.1:8000/qa/"
UPLOAD_DIR = Path("uploaded_cases")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="CBCT Agent – Interactive Chat",
    layout="wide",
)

# -----------------------
# SESSION STATE INIT
# -----------------------
if "uploaded_file_path" not in st.session_state:
    st.session_state["uploaded_file_path"] = None

if "question_input" not in st.session_state:
    st.session_state["question_input"] = ""

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# -----------------------
# UTILS
# -----------------------
def color_from_confidence(c: float) -> str:
    if c >= 0.70:
        return "#e6fff0"    # greenish
    if c >= 0.40:
        return "#fffbe6"    # yellowish
    return "#fff0f0"        # reddish


def chat_bubble(question, answer, confidence):
    bg = color_from_confidence(confidence)
    return f"""
    <div style="padding:16px; margin-bottom:12px; background:{bg};
                border-radius:10px; border:1px solid #ddd;">
        <div style="font-size:15px;"><b>Q:</b> {question}</div>
        <div style="font-size:15px; margin-top:6px;"><b>A:</b> {answer}</div>
        <div style="margin-top:8px; font-size:12px; opacity:0.7;">
            Confidence: {confidence:.2f}
        </div>
    </div>
    """


# -----------------------
# HEADER
# -----------------------
st.markdown(
    """
    <h2 style='color:#004e89; font-weight:700;'>🦷 CBCT Agent – Interactive Chat</h2>
    <p style='font-size:16px;'>
    Upload a CBCT DICOM, ZIP, folder or PDF and ask any dental or radiology question.
    </p>
    """,
    unsafe_allow_html=True,
)


# -----------------------
# LAYOUT: LEFT → UPLOAD PANEL, RIGHT → CHAT PANEL
# -----------------------
left, right = st.columns([1, 2])

# -----------------------
# LEFT SIDE – Case Upload
# -----------------------
with left:
    st.subheader("📁 Case Upload")

    uploaded_file = st.file_uploader(
        "Upload CBCT file (.dcm, ZIP, PDF)",
        type=["dcm", "zip", "pdf"],
        help="Max 200MB",
    )

    if uploaded_file:
        file_path = UPLOAD_DIR / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        st.session_state["uploaded_file_path"] = str(file_path)
        st.success(f"Uploaded: {uploaded_file.name}")

    if st.session_state["uploaded_file_path"]:
        st.info(f"Active Case: {st.session_state['uploaded_file_path']}")


# -----------------------
# RIGHT SIDE – Chat Interface
# -----------------------
with right:

    st.subheader("💬 Ask Your CBCT Question")

    q = st.text_input(
        "Question",
        key="question_input",
        placeholder="Example: What are the findings? Is the bone height adequate?",
    )

    ask = st.button("Ask", use_container_width=True)

    if ask:
        if not st.session_state["uploaded_file_path"]:
            st.error("Please upload a CBCT file before asking a question.")
        else:
            with st.spinner("Analyzing CBCT..."):

                payload = {
                    "question": q,
                    "case_file_path": st.session_state["uploaded_file_path"]
                }

                try:
                    res = requests.post(FASTAPI_QA_URL, json=payload)
                except Exception:
                    st.error("Backend unreachable. Please start FastAPI.")
                    st.stop()

            if res.status_code == 200:
                out = res.json()

                bubble = chat_bubble(
                    out["question"], out["answer"], out["confidence"]
                )
                st.session_state["chat_history"].append(bubble)

            else:
                st.error("Backend returned an error.")


    # CHAT HISTORY
    st.subheader("📜 Chat History")
    for item in st.session_state["chat_history"]:
        st.markdown(item, unsafe_allow_html=True)
