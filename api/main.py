from fastapi import FastAPI, HTTPException

from models.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from preprocessing.skill_extraction import SkillExtractor
from preprocessing.text_preprocessor import TextPreprocessor
from services.embedding_service import EmbeddingModel
from services.similarity_service import SimilarityModel
from utils.logger import setup_logger


setup_logger()

app = FastAPI(
    title="AI Resume Screening Assistant API",
    description=(
        "REST API for semantic resume screening "
        "using Sentence Transformers."
    ),
    version="1.0.0",
)


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health_check():
    """Check API availability."""

    return {
        "status": "healthy"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(request: PredictionRequest):
    """Compare a resume against a Job Description."""

    try:
        jd_clean = TextPreprocessor.preprocess(
            request.job_description
        )

        resume_clean = TextPreprocessor.preprocess(
            request.resume_text
        )

        jd_embedding = (
            EmbeddingModel.generate_embedding(
                jd_clean
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

        jd_skills = set(
            SkillExtractor.extract_skills(
                request.job_description
            )
        )

        resume_skills = set(
            SkillExtractor.extract_skills(
                request.resume_text
            )
        )

        matched_skills = sorted(
            jd_skills.intersection(
                resume_skills
            )
        )

        missing_skills = sorted(
            jd_skills.difference(
                resume_skills
            )
        )

        return {
            "similarity_score": score,
            "category": category,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Internal server error.",
        ) from exc