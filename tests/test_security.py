import pytest
from fastapi.testclient import TestClient

from api.main import app
from utils.validator import validate_file

client = TestClient(app)


def test_malicious_input_is_handled():
    """Test that suspicious input does not crash the API."""

    response = client.post(
        "/predict",
        json={
            "job_description": "<script>alert('xss')</script> Python developer",
            "resume_text": "../../etc/passwd Python developer",
        },
    )

    # The API should safely handle the input without an internal server error.
    assert response.status_code in (200, 400, 422)


def test_internal_error_details_are_not_exposed():
    """Test that internal exception details are not returned to clients."""

    response = client.post(
        "/predict",
        json={
            "job_description": "Python developer with machine learning",
            "resume_text": "Python developer with machine learning",
        },
    )

    # If an internal error occurs, the API should return a generic message.
    if response.status_code == 500:
        assert response.json()["detail"] == "Internal server error."


def test_unsupported_file_type_is_rejected():
    """Test that unsupported file extensions are rejected."""

    with pytest.raises(ValueError):
        validate_file("malicious.exe")


def test_file_without_extension_is_rejected():
    """Test that files without extensions are rejected."""

    with pytest.raises(ValueError):
        validate_file("resume")


def test_empty_file_path_is_rejected():
    """Test that an empty file path is rejected."""

    with pytest.raises(ValueError):
        validate_file("")


def test_unknown_api_endpoint_is_not_exposed():
    """Test that undefined API endpoints return 404."""

    response = client.get("/admin")

    assert response.status_code == 404


def test_invalid_http_method_is_rejected():
    """Test that unsupported HTTP methods are rejected."""

    response = client.get("/predict")

    assert response.status_code == 405


def test_malformed_json_is_rejected():
    """Test that malformed JSON requests are rejected."""

    response = client.post(
        "/predict",
        content='{"job_description": "Python developer",',
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422