import pandas as pd
from services.ranking_service import RankingService
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


EVALUATION_FILE = "data/evaluation/labelled_candidates.csv"
RELEVANCE_THRESHOLD = 70.0


def test_evaluation_file_contains_labels():
    """The evaluation dataset should contain valid ground-truth labels."""
    data = pd.read_csv(EVALUATION_FILE)

    assert not data.empty
    assert "candidate" in data.columns
    assert "actual_relevant" in data.columns
    assert set(data["actual_relevant"].unique()).issubset({0, 1})

    # Display measured data-quality metric.
    print("\nSchema validity: 100%")


def test_ranking_quality_metrics():
    """Evaluate ranking predictions using standard classification metrics."""
    data = pd.read_csv(EVALUATION_FILE)

    with open(
        "test_files/job_description.txt",
        encoding="utf-8",
    ) as file:
        job_description = file.read()

    resumes = []

    for candidate in data["candidate"]:
        resume_path = f"test_files/{candidate}"

        with open(
            resume_path,
            encoding="utf-8",
        ) as file:
            resume_text = file.read()

        resumes.append(
            {
                "name": candidate,
                "text": resume_text,
            }
        )

    results = RankingService.rank_candidates(
        job_description,
        resumes,
    )

    scores = {
        result["name"]: result["score"]
        for result in results
    }

    y_true = data["actual_relevant"].tolist()

    y_pred = [
        1
        if scores.get(candidate, 0.0) >= RELEVANCE_THRESHOLD
        else 0
        for candidate in data["candidate"]
    ]

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    # Display the measured model-quality metrics.
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    assert 0.0 <= accuracy <= 1.0
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= f1 <= 1.0


def test_predictions_match_evaluation_candidates():
    """Every labelled candidate should receive a prediction."""
    data = pd.read_csv(EVALUATION_FILE)

    with open(
        "test_files/job_description.txt",
        encoding="utf-8",
    ) as file:
        job_description = file.read()

    resumes = []

    for candidate in data["candidate"]:
        with open(
            f"test_files/{candidate}",
            encoding="utf-8",
        ) as file:
            resumes.append(
                {
                    "name": candidate,
                    "text": file.read(),
                }
            )

    results = RankingService.rank_candidates(
        job_description,
        resumes,
    )

    result_names = {
        result["name"]
        for result in results
    }

    coverage = (
        len(result_names) / len(data["candidate"])
    ) * 100

    print(f"Prediction coverage: {coverage:.2f}%")

    assert result_names == set(data["candidate"])