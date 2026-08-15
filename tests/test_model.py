import numpy as np
import pytest
from services.embedding_service import EmbeddingModel
from services.similarity_service import SimilarityModel


def test_embedding_output_is_generated():
    """The embedding model should generate an embedding for valid text."""
    text = "Python developer with machine learning experience."

    embedding = EmbeddingModel.generate_embedding(text)

    assert embedding is not None
    assert len(embedding) > 0


def test_embedding_output_contains_valid_numbers():
    """Generated embeddings should contain finite numeric values."""
    text = "Software engineer with Python and NLP experience."

    embedding = EmbeddingModel.generate_embedding(text)
    embedding_array = np.asarray(embedding)

    assert embedding_array.size > 0
    assert np.all(np.isfinite(embedding_array))


def test_embedding_output_has_expected_dimension():
    """The MiniLM embedding should have a consistent feature dimension."""
    text = "Machine learning engineer with Python experience."

    embedding = EmbeddingModel.generate_embedding(text)
    embedding_array = np.asarray(embedding)

    assert embedding_array.ndim == 1
    assert embedding_array.shape[0] > 0


def test_embedding_is_consistent_for_same_input():
    """The same input should produce the same embedding."""
    text = "Machine learning engineer with Python experience."

    embedding_1 = EmbeddingModel.generate_embedding(text)
    embedding_2 = EmbeddingModel.generate_embedding(text)

    np.testing.assert_allclose(
        np.asarray(embedding_1),
        np.asarray(embedding_2),
        rtol=1e-5,
        atol=1e-6,
    )


def test_similarity_returns_valid_score_and_category():
    """Similarity should return a valid percentage and match category."""
    resume = "Python developer with machine learning experience."
    job_description = "Looking for a Python and machine learning developer."

    resume_embedding = EmbeddingModel.generate_embedding(resume)
    jd_embedding = EmbeddingModel.generate_embedding(job_description)

    percentage, category = SimilarityModel.calculate_similarity(
        resume_embedding,
        jd_embedding,
    )

    assert 0.0 <= percentage <= 100.0

    assert category in {
        "Excellent Match",
        "Good Match",
        "Average Match",
        "Low Match",
    }


def test_identical_embeddings_produce_perfect_match():
    """Identical embeddings should produce a 100% similarity score."""
    text = "Python machine learning engineer."

    embedding = EmbeddingModel.generate_embedding(text)

    percentage, category = SimilarityModel.calculate_similarity(
        embedding,
        embedding,
    )

    assert percentage == pytest.approx(100.0)
    assert category == "Excellent Match"


def test_empty_text_is_rejected():
    """Empty input text should raise ValueError."""
    with pytest.raises(ValueError, match="Input text cannot be empty."):
        EmbeddingModel.generate_embedding("")


def test_none_embeddings_are_rejected():
    """None embeddings should raise ValueError."""
    with pytest.raises(
        ValueError,
        match="Embeddings cannot be None.",
    ):
        SimilarityModel.calculate_similarity(None, None)


def test_mismatched_embedding_dimensions_are_rejected():
    """Embeddings with different dimensions should raise ValueError."""
    embedding_1 = np.array([1.0, 2.0, 3.0])
    embedding_2 = np.array([1.0, 2.0])

    with pytest.raises(
        ValueError,
        match="Resume and JD embeddings must have the same dimensions.",
    ):
        SimilarityModel.calculate_similarity(
            embedding_1,
            embedding_2,
        )
