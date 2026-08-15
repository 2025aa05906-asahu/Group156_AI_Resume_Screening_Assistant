import os
import tempfile

import streamlit as st

from services.screening_service import ScreeningService
from utils.logger import setup_logger

setup_logger()

st.set_page_config(
    page_title="AI Resume Screening Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("AI Resume Screening Assistant")
st.write(
    "Upload a Job Description and one or more resumes "
    "to rank candidates based on semantic similarity."
)

jd_file = st.file_uploader(
    "Upload Job Description",
    type=["pdf", "docx", "txt"],
)

resume_files = st.file_uploader(
    "Upload Resumes",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

if st.button("Analyze Resumes", type="primary"):

    if not jd_file:
        st.error("Please upload a Job Description.")

    elif not resume_files:
        st.error("Please upload at least one resume.")

    else:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:

                jd_path = os.path.join(
                    temp_dir,
                    jd_file.name,
                )

                with open(jd_path, "wb") as file:
                    file.write(jd_file.getbuffer())

                resume_paths = []

                for resume_file in resume_files:
                    resume_path = os.path.join(
                        temp_dir,
                        resume_file.name,
                    )

                    with open(resume_path, "wb") as file:
                        file.write(resume_file.getbuffer())

                    resume_paths.append(resume_path)

                results_df = ScreeningService.analyze(
                    jd_path,
                    resume_paths,
                )

            if results_df.empty:
                st.warning("No candidates could be processed.")
            else:
                st.success("Resume screening completed successfully.")

                display_df = results_df.copy()

                display_df.insert(
                    0,
                    "Rank",
                    range(1, len(display_df) + 1),
                )

                display_df["matched_skills"] = display_df["matched_skills"].apply(
                    lambda skills: (
                        ", ".join(skills) if isinstance(skills, list) else str(skills)
                    )
                )

                display_df["missing_skills"] = display_df["missing_skills"].apply(
                    lambda skills: (
                        ", ".join(skills) if isinstance(skills, list) else str(skills)
                    )
                )

                st.subheader("Ranked Candidates")

                st.dataframe(
                    display_df,
                    use_container_width=True,
                )

        except ValueError as exc:
            st.error(f"Input error: {exc}")

        except RuntimeError as exc:
            st.error(f"Processing error: {exc}")

        except Exception:
            st.error(
                "An unexpected error occurred while processing "
                "the resumes. Please check the application logs."
            )
