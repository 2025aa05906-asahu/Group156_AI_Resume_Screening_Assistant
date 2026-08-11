import pytest

from services.ranking_service import RankingService


def test_empty_resume_list():
    """Test ranking when no resumes are provided."""

    result = RankingService.rank_candidates(
        "Python developer with SQL experience",
        [],
    )

    assert result == []


def test_empty_job_description():
    """Test that an empty job description raises ValueError."""

    with pytest.raises(ValueError):
        RankingService.rank_candidates(
            "",
            [],
        )


def test_rank_candidates_returns_expected_fields(monkeypatch):
    """Test that ranking returns all required candidate fields."""

    def mock_embedding(text):
        return [1.0, 0.0, 0.0]

    def mock_similarity(resume_embedding, jd_embedding):
        return 80.0, "Good Match"

    monkeypatch.setattr(
        "services.ranking_service.EmbeddingModel.generate_embedding",
        mock_embedding,
    )

    monkeypatch.setattr(
        "services.ranking_service.SimilarityModel.calculate_similarity",
        mock_similarity,
    )

    resumes = [
        {
            "name": "candidate1.txt",
            "text": "Python developer with SQL experience",
        }
    ]

    result = RankingService.rank_candidates(
        "Python developer with SQL experience",
        resumes,
    )

    assert len(result) == 1

    candidate = result[0]

    assert candidate["name"] == "candidate1.txt"
    assert candidate["score"] == 80.0
    assert candidate["category"] == "Good Match"
    assert "matched_skills" in candidate
    assert "missing_skills" in candidate


def test_candidates_are_sorted_by_score(monkeypatch):
    """Test that candidates are ranked from highest to lowest score."""

    def mock_embedding(text):
        return [1.0, 0.0, 0.0]

    scores = {
        "candidate1.txt": (60.0, "Average Match"),
        "candidate2.txt": (90.0, "Excellent Match"),
        "candidate3.txt": (75.0, "Good Match"),
    }

    def mock_similarity(resume_embedding, jd_embedding):
        # The actual candidate is identified through the embedding
        # returned by mock_embedding.
        return resume_embedding

    def mock_embedding_with_score(text):
        if "candidate 1" in text.lower():
            return scores["candidate1.txt"]
        if "candidate 2" in text.lower():
            return scores["candidate2.txt"]
        return scores["candidate3.txt"]

    def mock_similarity_with_score(resume_embedding, jd_embedding):
        return resume_embedding

    monkeypatch.setattr(
        "services.ranking_service.EmbeddingModel.generate_embedding",
        mock_embedding_with_score,
    )

    monkeypatch.setattr(
        "services.ranking_service.SimilarityModel.calculate_similarity",
        mock_similarity_with_score,
    )

    resumes = [
        {
            "name": "candidate1.txt",
            "text": "Candidate 1 Python developer",
        },
        {
            "name": "candidate2.txt",
            "text": "Candidate 2 Python developer",
        },
        {
            "name": "candidate3.txt",
            "text": "Candidate 3 Python developer",
        },
    ]

    result = RankingService.rank_candidates(
        "Python developer",
        resumes,
    )

    scores_in_result = [
        candidate["score"]
        for candidate in result
    ]

    assert scores_in_result == sorted(
        scores_in_result,
        reverse=True,
    )


def test_empty_resume_is_skipped(monkeypatch):
    """Test that resumes with empty text are skipped."""

    monkeypatch.setattr(
        "services.ranking_service.EmbeddingModel.generate_embedding",
        lambda text: [1.0, 0.0, 0.0],
    )

    monkeypatch.setattr(
        "services.ranking_service.SimilarityModel.calculate_similarity",
        lambda resume_embedding, jd_embedding: (
            80.0,
            "Good Match",
        ),
    )

    resumes = [
        {
            "name": "empty_resume.txt",
            "text": "",
        },
        {
            "name": "valid_resume.txt",
            "text": "Python developer with SQL experience",
        },
    ]

    result = RankingService.rank_candidates(
        "Python developer",
        resumes,
    )

    assert len(result) == 1
    assert result[0]["name"] == "valid_resume.txt"


def test_ranking_score_is_within_valid_range(monkeypatch):
    """Test that ranking scores are between 0 and 100."""

    monkeypatch.setattr(
        "services.ranking_service.EmbeddingModel.generate_embedding",
        lambda text: [1.0, 0.0, 0.0],
    )

    monkeypatch.setattr(
        "services.ranking_service.SimilarityModel.calculate_similarity",
        lambda resume_embedding, jd_embedding: (
            85.0,
            "Excellent Match",
        ),
    )

    resumes = [
        {
            "name": "candidate.txt",
            "text": "Python developer",
        }
    ]

    result = RankingService.rank_candidates(
        "Python developer",
        resumes,
    )

    assert 0.0 <= result[0]["score"] <= 100.0
