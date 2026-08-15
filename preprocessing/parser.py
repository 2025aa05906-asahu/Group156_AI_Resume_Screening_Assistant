import logging
import os

import pdfplumber
from docx import Document

logger = logging.getLogger(__name__)


class ResumeParser:
    """Extract text from resume files."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

    @staticmethod
    def extract_text(file_path: str) -> str:
        if not file_path:
            raise ValueError("Resume file path cannot be empty.")

        extension = os.path.splitext(file_path)[1].lower()

        if extension not in ResumeParser.SUPPORTED_EXTENSIONS:
            logger.error("Unsupported resume format: %s", extension)
            raise ValueError(
                "Unsupported file format. "
                "Only PDF, DOCX and TXT files are supported."
            )

        logger.info("Parsing resume: %s", file_path)

        try:
            if extension == ".pdf":
                text = ResumeParser.extract_pdf(file_path)
            elif extension == ".docx":
                text = ResumeParser.extract_docx(file_path)
            else:
                text = ResumeParser.extract_txt(file_path)

            if not text.strip():
                logger.warning("No text extracted from resume: %s", file_path)

            return text.strip()

        except Exception as exc:
            logger.error("Failed to parse resume: %s", exc)
            raise RuntimeError(f"Unable to parse resume: {exc}") from exc

    @staticmethod
    def extract_pdf(file_path: str) -> str:
        pages = []

        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)

        return "\n".join(pages)

    @staticmethod
    def extract_docx(file_path: str) -> str:
        document = Document(file_path)

        return "\n".join(
            paragraph.text.strip()
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        )

    @staticmethod
    def extract_txt(file_path: str) -> str:
        with open(
            file_path,
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as file:
            return file.read()


class JDParser(ResumeParser):
    """Extract text from Job Description files."""

    @staticmethod
    def extract_text(file_path: str) -> str:
        logger.info("Parsing Job Description: %s", file_path)
        return ResumeParser.extract_text(file_path)
