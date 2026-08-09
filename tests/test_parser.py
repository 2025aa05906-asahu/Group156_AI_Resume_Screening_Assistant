from preprocessing import ResumeParser
from preprocessing import JDParser


# ------------ Resume ------------

resume_path = "data/resumes/sample_resume.docx"

try:

    resume_text = ResumeParser.extract_text(resume_path)

    print("=" * 60)
    print("RESUME")
    print("=" * 60)

    print(resume_text[:500])

except Exception as e:

    print("Resume Error:", e)


# ------------ Job Description ------------

jd_path = "data/job_descriptions/sample_job_description.docx"

try:

    jd_text = JDParser.extract_text(jd_path)

    print("=" * 60)
    print("JOB DESCRIPTION")
    print("=" * 60)

    print(jd_text[:500])

except Exception as e:

    print("JD Error:", e)