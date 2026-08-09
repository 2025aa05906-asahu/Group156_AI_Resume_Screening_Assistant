import numpy as np
import pytest

from services.similarity_service import (
    SimilarityModel,
)


def test_similarity_identical_vectors():

    vector = np.array(
        [1.0, 0.0, 0.0]
    )

    score, category = (
        SimilarityModel.calculate_similarity(
            vector,
            vector,
        )
    )

    assert score == 100.0
    assert category == "Excellent Match"


def test_similarity_dimension_mismatch():

    v1 = np.array([1.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])

    with pytest.raises(ValueError):
        SimilarityModel.calculate_similarity(
            v1,
            v2,
        )