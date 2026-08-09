import streamlit as st

from services.screening_service import ScreeningService
from utils.logger import setup_logger

setup_logger()

st.title("AI Resume Screening Assistant")

jd_file = st.file_uploader(
    "Upload Job Description",
    type=["pdf", "docx", "txt"],
)

resume_files = st.file_uploader(
    "Upload Resumes",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

if st.button("Analyze Resumes"):

    if jd_file and resume_files:

        # Save files temporarily
        # Then call ScreeningService.analyze()

        st.success(
            "Resume screening completed."
        )