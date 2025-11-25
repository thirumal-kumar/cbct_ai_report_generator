# chat_ui_v3.py — Streamlit chat UI (clean clinical layout)
import streamlit as st
import requests
import datetime
import json

API_URL = "http://127.0.0.1:8000/qa/"

st.set_page_config(page_title="CBCT Agent – Chat (v3)", layout="wide")

# Theme colors (clean dental)
PRIMARY = "#0e83d5"   # blue
ACCENT = "#2fc4c4"    # aqua
BG = "#ffffff"

st.markdown(
    f"""
    <style>
    .chat-box {{ background: {BG}; border-radius: 10px; padding: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
    .user-bubble {{ background: #e6f0ff; color: #000; padding:10px; border-radius:10px; }}
    .assistant-bubble {{ background: #f7fcff; color: #000; padding:12px; border-radius:10px; border-left:4px solid {PRIMARY}; }}
    .meta {{ color: #666; font-size:12px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🦷 CBCT Agent — Clinical Chat (v3)")

left, right = st.columns([0.48, 0.52])

with left:
    st.header("Case Panel")
    uploaded = st.file_uploader("Upload CBCT (.dcm, .zip, .pdf) (optional)", type=["dcm", "zip", "pdf"])
    if uploaded:
        st.success(f"Selected: {uploaded.name}")
        # optional: you can implement upload to backend / preview later

    st.markdown("---")
    st.subheader("Prompt options")
    include_meta = st.checkbox("Include metadata in prompt (if available)", value=True)
    deeper = st.checkbox("Request deeper analysis (slower)", value=False)

with right:
    st.header("Clinical Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    q = st.text_input("Enter clinical question", key="question_input")
    send = st.button("Send")

    if send and q.strip():
        # add user message
        st.session_state.messages.append({
            "role": "user",
            "text": q.strip(),
            "time": datetime.datetime.now().strftime("%H:%M:%S")
        })

        payload = {"question": q.strip()}
        try:
            r = requests.post(API_URL, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            st.error(f"Agent error: {e}")
            data = None

        if data:
            # Normalize fields exist
            short = data.get("short_answer", "")
            observations = data.get("observations", [])
            reasoning = data.get("reasoning", "")
            recommendations = data.get("recommendations", [])
            confidence = data.get("confidence", 0.0)
            evidence = data.get("evidence", [])
            model_src = data.get("model_meta", "unknown")

            st.session_state.messages.append({
                "role": "assistant",
                "short": short,
                "observations": observations,
                "reasoning": reasoning,
                "recommendations": recommendations,
                "confidence": confidence,
                "evidence": evidence,
                "model": model_src,
                "time": datetime.datetime.now().strftime("%H:%M:%S")
            })

    # conversation render
    st.subheader("Conversation")
    for msg in st.session_state.messages[::-1]:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-bubble chat-box'><b>You</b> <span class='meta'> {msg['time']}</span><br/>{msg['text']}</div>", unsafe_allow_html=True)
        else:
            # assistant
            st.markdown(f"<div class='assistant-bubble chat-box'><b>CBCT-AI</b> <span class='meta'> {msg['time']} — model: {msg.get('model','?')}</span></div>", unsafe_allow_html=True)
            # summary
            if msg.get("short"):
                st.markdown(f"**Summary:** {msg['short']}")
            # observations
            if msg.get("observations"):
                st.markdown("**Key observations:**")
                for ob in msg["observations"]:
                    st.write(f"- {ob}")
            # reasoning paragraph
            if msg.get("reasoning"):
                st.markdown("**Reasoning:**")
                st.write(msg["reasoning"])
            # recommendations
            if msg.get("recommendations"):
                st.markdown("**Recommendations:**")
                for r in msg["recommendations"]:
                    st.write(f"- {r}")
            # confidence
            st.caption(f"Confidence: {msg.get('confidence',0.0):.3f}")

            # evidence expander (no IDs)
            with st.expander("View retrieved evidence"):
                for e in msg.get("evidence", []):
                    st.write(e)

    st.markdown("---")
    st.caption("Tip: ask focused clinical questions (e.g., 'Is bone sufficient for implant at 36?')")

