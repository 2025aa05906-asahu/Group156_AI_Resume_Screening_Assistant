import logging

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class SimilarityModel:
    """Calculate cosine similarity between embeddings."""

    @staticmethod
    def calculate_similarity(
        resume_embedding,
        jd_embedding,
    ):
        if resume_embedding is None or jd_embedding is None:
            raise ValueError("Embeddings cannot be None.")

        resume_embedding = np.asarray(resume_embedding)
        jd_embedding = np.asarray(jd_embedding)

        if resume_embedding.shape != jd_embedding.shape:
            raise ValueError(
                "Resume and JD embeddings must have "
                "the same dimensions."
            )

        score = cosine_similarity(
            resume_embedding.reshape(1, -1),
            jd_embedding.reshape(1, -1),
        )[0][0]

        score = max(0.0, min(1.0, float(score)))

        percentage = round(score * 100, 2)

        if percentage >= 85:
            category = "Excellent Match"
        elif percentage >= 70:
            category = "Good Match"
        elif percentage >= 50:
            category = "Average Match"
        else:
            category = "Low Match"

        logger.info(
            "Similarity calculated: %.2f%%",
            percentage,
        )

        return percentage, category