import os

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
}


def validate_file(file_path: str) -> bool:
    """Validate uploaded file extension."""

    if not file_path:
        raise ValueError("File path cannot be empty.")

    extension = os.path.splitext(file_path)[1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {extension}")

    return True
