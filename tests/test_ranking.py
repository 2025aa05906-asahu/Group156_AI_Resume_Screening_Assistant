from unittest.mock import patch

from services.ranking_service import (
    RankingService,
)


def test_empty_resume_list():

    result = RankingService.rank_candidates(
        "Python developer with SQL experience",
        [],
    )

    assert result == []


def test_empty_job_description():

    try:
        RankingService.rank_candidates(
            "",
            [],
        )

        assert False

    except ValueError:
        assert True