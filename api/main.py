import logging

from fastapi import FastAPI, HTTPException

from models.schemas import HealthResponse, PredictionRequest, PredictionResponse
from preprocessing.skill_extraction import SkillExtractor
from preprocessing.text_preprocessor import TextPreprocessor
from services.embedding_service import EmbeddingModel
from services.similarity_service import SimilarityModel
from utils.logger import setup_logger

setup_logger()

logger = logging.getLogger(__name__)


app = FastAPI(
    title="AI Resume Screening Assistant API",
    description=(
        "REST API for semantic resume screening using "
        "Sentence Transformers. The API compares a candidate "
        "resume with a job description and identifies matched "
        "and missing technical skills."
    ),
    version="1.0.0",
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check whether the Resume Screening API is available.",
)
def health_check():
    """Check API availability."""

    logger.info("Health check requested")

    return {"status": "healthy"}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Resume Screening Prediction",
    description=(
        "Compare a candidate resume against a job description "
        "using semantic similarity and identify matched and "
        "missing technical skills."
    ),
)
def predict(request: PredictionRequest):
    """Compare a resume against a Job Description."""

    logger.info("Prediction request received")

    try:
        # Preprocess job description and resume
        jd_clean = TextPreprocessor.preprocess(request.job_description)

        resume_clean = TextPreprocessor.preprocess(request.resume_text)

        # Generate embeddings
        jd_embedding = EmbeddingModel.generate_embedding(jd_clean)

        resume_embedding = EmbeddingModel.generate_embedding(resume_clean)

        # Calculate semantic similarity
        score, category = SimilarityModel.calculate_similarity(
            resume_embedding,
            jd_embedding,
        )

        # Extract skills
        jd_skills = set(SkillExtractor.extract_skills(request.job_description))

        resume_skills = set(SkillExtractor.extract_skills(request.resume_text))

        # Find matched and missing skills
        matched_skills = sorted(jd_skills.intersection(resume_skills))

        missing_skills = sorted(jd_skills.difference(resume_skills))

        # Log successful prediction
        logger.info(
            "Prediction completed successfully | " "category=%s | score=%.4f",
            category,
            score,
        )

        return {
            "similarity_score": score,
            "category": category,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
        }

    except ValueError as exc:
        logger.warning(
            "Invalid prediction request: %s",
            exc,
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception:
        logger.exception("Unexpected error during prediction")

        raise HTTPException(
            status_code=500,
            detail="Internal server error.",
        )
