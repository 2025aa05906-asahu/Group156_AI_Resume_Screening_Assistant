import logging
import os
from typing import List

import pandas as pd

from preprocessing.parser import JDParser, ResumeParser
from services.ranking_service import RankingService

logger = logging.getLogger(__name__)


class ScreeningService:
    """Main application service for resume screening."""

    @staticmethod
    def analyze(
        jd_file_path: str,
        resume_file_paths: List[str],
    ) -> pd.DataFrame:

        logger.info("Starting resume screening process.")

        if not jd_file_path:
            raise ValueError("Job Description file is required.")

        if not resume_file_paths:
            raise ValueError("At least one resume is required.")

        jd_text = JDParser.extract_text(
            jd_file_path
        )

        resumes = []

        for resume_path in resume_file_paths:
            resumes.append(
                {
                    "name": os.path.basename(resume_path),
                    "text": ResumeParser.extract_text(
                        resume_path
                    ),
                }
            )

        results = RankingService.rank_candidates(
            jd_text,
            resumes,
        )

        logger.info(
            "Resume screening completed successfully."
        )

        return pd.DataFrame(results)