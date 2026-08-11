from pathlib import Path

import pytest

from preprocessing import JDParser, ResumeParser


# Get the project root directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Directory containing test input files.
TEST_FILES = PROJECT_ROOT / "test_files"


def test_resume_parser():
    """Test that the resume parser extracts text successfully."""

    resume_path = TEST_FILES / "resume_1.txt"

    resume_text = ResumeParser.extract_text(
        str(resume_path)
    )

    # The parser should return text.
    assert isinstance(resume_text, str)

    # The extracted resume should not be empty.
    assert len(resume_text.strip()) > 0


def test_job_description_parser():
    """Test that the job-description parser extracts text successfully."""

    jd_path = TEST_FILES / "job_description.txt"

    jd_text = JDParser.extract_text(
        str(jd_path)
    )

    # The parser should return text.
    assert isinstance(jd_text, str)

    # The extracted job description should not be empty.
    assert len(jd_text.strip()) > 0


def test_multiple_resume_files():
    """Test that multiple resume files can be parsed."""

    resume_files = [
        TEST_FILES / "resume_1.txt",
        TEST_FILES / "resume_2.txt",
        TEST_FILES / "resume_3.txt",
    ]

    for resume_path in resume_files:
        resume_text = ResumeParser.extract_text(
            str(resume_path)
        )

        assert isinstance(resume_text, str)
        assert len(resume_text.strip()) > 0


def test_resume_parser_missing_file():
    """Test that a missing resume file is handled correctly."""

    missing_path = TEST_FILES / "does_not_exist.txt"

    with pytest.raises((FileNotFoundError, ValueError, RuntimeError, OSError)):
        ResumeParser.extract_text(
            str(missing_path)
        )