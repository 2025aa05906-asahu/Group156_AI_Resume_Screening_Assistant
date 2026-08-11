import logging
from typing import Dict, List

from preprocessing.skill_extraction import SkillExtractor
from preprocessing.text_preprocessor import TextPreprocessor
from services.embedding_service import EmbeddingModel
from services.similarity_service import SimilarityModel


logger = logging.getLogger(__name__)


class RankingService:
    """Rank resumes against a Job Description."""

    @staticmethod
    def rank_candidates(
        job_description: str,
        resumes: List[Dict],
    ) -> List[Dict]:
        """Rank candidate resumes against a job description."""

        if not job_description or not job_description.strip():
            raise ValueError(
                "Job Description cannot be empty."
            )

        if not resumes:
            logger.warning(
                "No resumes supplied."
            )
            return []

        logger.info(
            "Ranking %d candidates.",
            len(resumes),
        )

        jd_clean = TextPreprocessor.preprocess(
            job_description
        )

        jd_embedding = (
            EmbeddingModel.generate_embedding(
                jd_clean
            )
        )

        jd_skills = set(
            SkillExtractor.extract_skills(
                job_description
            )
        )

        ranked_candidates = []

        for resume in resumes:
            candidate_name = resume.get(
                "name",
                "Unknown Candidate",
            )

            resume_text = resume.get(
                "text",
                "",
            )

            if not resume_text.strip():
                logger.warning(
                    "Skipping empty resume: %s",
                    candidate_name,
                )
                continue

            resume_clean = (
                TextPreprocessor.preprocess(
                    resume_text
                )
            )

            resume_embedding = (
                EmbeddingModel.generate_embedding(
                    resume_clean
                )
            )

            score, category = (
                SimilarityModel.calculate_similarity(
                    resume_embedding,
                    jd_embedding,
                )
            )

            resume_skills = set(
                SkillExtractor.extract_skills(
                    resume_text
                )
            )

            matched_skills = sorted(
                resume_skills.intersection(
                    jd_skills
                )
            )

            missing_skills = sorted(
                jd_skills.difference(
                    resume_skills
                )
            )

            ranked_candidates.append(
                {
                    "name": candidate_name,
                    "score": score,
                    "category": category,
                    "matched_skills": matched_skills,
                    "missing_skills": missing_skills,
                }
            )

        ranked_candidates.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        logger.info(
            "Candidate ranking completed."
        )

        return ranked_candidates