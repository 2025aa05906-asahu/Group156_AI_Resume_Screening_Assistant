from typing import List

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Input schema for resume screening."""

    job_description: str = Field(
        ...,
        min_length=10,
        description="Job Description text",
        examples=["Python developer with experience in FastAPI and machine learning"],
    )

    resume_text: str = Field(
        ...,
        min_length=10,
        description="Candidate resume text",
        examples=["Python developer with experience in FastAPI and machine learning"],
    )


class PredictionResponse(BaseModel):
    """Output schema for resume screening."""

    similarity_score: float = Field(
        ...,
        description=(
            "Semantic similarity score between the resume " "and job description"
        ),
        examples=[85.5],
    )

    category: str = Field(
        ...,
        description="Resume matching category based on the similarity score",
        examples=["Excellent Match"],
    )

    matched_skills: List[str] = Field(
        ...,
        description=(
            "Technical skills found in both the job description " "and resume"
        ),
        examples=[["python", "fastapi", "machine learning"]],
    )

    missing_skills: List[str] = Field(
        ...,
        description=(
            "Technical skills required by the job description "
            "but missing from the resume"
        ),
        examples=[["docker", "aws"]],
    )


class HealthResponse(BaseModel):
    """Response schema for API health check."""

    status: str = Field(
        ...,
        description="Current API health status",
        examples=["healthy"],
    )
