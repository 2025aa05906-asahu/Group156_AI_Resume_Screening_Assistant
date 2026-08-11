import pytest

from utils.validator import validate_file


def test_valid_pdf_file():
    """A PDF file should be accepted."""
    assert validate_file("sample.pdf") is True


def test_valid_docx_file():
    """A DOCX file should be accepted."""
    assert validate_file("sample.docx") is True


def test_valid_txt_file():
    """A TXT file should be accepted."""
    assert validate_file("sample.txt") is True


def test_uppercase_extension_is_accepted():
    """File extensions should be handled case-insensitively."""
    assert validate_file("sample.PDF") is True
    assert validate_file("sample.DOCX") is True
    assert validate_file("sample.TXT") is True


def test_empty_file_path():
    """An empty file path should raise ValueError."""
    with pytest.raises(ValueError, match="File path cannot be empty."):
        validate_file("")


def test_unsupported_file_type():
    """Unsupported file extensions should raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported file type"):
        validate_file("sample.exe")


def test_file_without_extension():
    """A file without an extension should be rejected."""
    with pytest.raises(ValueError, match="Unsupported file type"):
        validate_file("sample")
