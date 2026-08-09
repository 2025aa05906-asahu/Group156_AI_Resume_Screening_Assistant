from typing import List

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Input schema for resume screening."""

    job_description: str = Field(
        ...,
        min_length=10,
        description="Job Description text",
    )

    resume_text: str = Field(
        ...,
        min_length=10,
        description="Candidate resume text",
    )


class PredictionResponse(BaseModel):
    """Output schema for resume screening."""

    similarity_score: float
    category: str
    matched_skills: List[str]
    missing_skills: List[str]


class HealthResponse(BaseModel):
    status: str