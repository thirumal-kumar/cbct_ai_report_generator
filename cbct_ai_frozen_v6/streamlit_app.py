import streamlit as st
import requests

BACKEND = "http://127.0.0.1:8000"

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
    st.info("Uploading study and generating evidence-aware report (this may take 20–60s)...")

    files = {
        "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
    }

    params = {
        "tight": "true" if tight_mode else "false"
    }

    try:
        r = requests.post(
            f"{BACKEND}/generate_report/",
            files=files,
            params=params,
            timeout=300
        )
    except Exception as e:
        st.error(f"Request failed: {e}")
        st.stop()

    if r.status_code != 200:
        st.error(f"Backend error {r.status_code}: {r.text}")
        st.stop()

    data = r.json()
    st.success("Report generated successfully")

    # -------------------------------
    # REPORT
    # -------------------------------
    st.subheader("Final CBCT Report")
    st.write(data["final_report"])

    # -------------------------------
    # METADATA
    # -------------------------------
    with st.expander("Patient / Study Metadata"):
        st.json(data["metadata"])

    # -------------------------------
    # EVIDENCE FILE NAMES (OPTIONAL)
    # -------------------------------
    if "retrieved_files" in data and data["retrieved_files"]:
        with st.expander("Evidence Sources Used"):
            for f in data["retrieved_files"]:
                st.write(f"- {f}")

    # -------------------------------
    # PDF DOWNLOAD
    # -------------------------------
    st.subheader("Download PDF Report")

    pdf = requests.post(
        f"{BACKEND}/generate_pdf/",
        json=data,
        timeout=120
    )

    if pdf.status_code == 200:
        st.download_button(
            "Download PDF",
            pdf.content,
            file_name="CBCT_Report.pdf",
            mime="application/pdf"
        )
    else:
        st.error("PDF generation failed")
