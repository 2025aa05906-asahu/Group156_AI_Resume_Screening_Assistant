import numpy as np
import pytest
from services.similarity_service import SimilarityModel


def test_similarity_identical_vectors():
    """Test that identical vectors produce a perfect match."""

    vector = np.array([1.0, 0.0, 0.0])

    score, category = SimilarityModel.calculate_similarity(
        vector,
        vector,
    )

    assert score == 100.0
    assert category == "Excellent Match"


def test_similarity_dimension_mismatch():
    """Test that vectors with different dimensions are rejected."""

    v1 = np.array([1.0, 0.0])

    v2 = np.array([1.0, 0.0, 0.0])

    with pytest.raises(ValueError):
        SimilarityModel.calculate_similarity(
            v1,
            v2,
        )


def test_similarity_score_range():
    """Test that the similarity score is within the valid range."""

    v1 = np.array([1.0, 0.0, 0.0])

    v2 = np.array([0.0, 1.0, 0.0])

    score, category = SimilarityModel.calculate_similarity(
        v1,
        v2,
    )

    assert 0.0 <= score <= 100.0
    assert isinstance(category, str)
    assert len(category) > 0


def test_similarity_zero_vector():
    """Test that a zero vector is handled without crashing."""

    v1 = np.array([0.0, 0.0, 0.0])

    v2 = np.array([1.0, 0.0, 0.0])

    score, category = SimilarityModel.calculate_similarity(
        v1,
        v2,
    )

    assert isinstance(score, (int, float))
    assert 0.0 <= score <= 100.0
    assert isinstance(category, str)
