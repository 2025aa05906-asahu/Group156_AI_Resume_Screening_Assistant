import logging
import re
from typing import List

logger = logging.getLogger(__name__)


COMMON_SKILLS = {
    "python",
    "java",
    "c",
    "c++",
    "javascript",
    "typescript",
    "sql",
    "mysql",
    "postgresql",
    "mongodb",
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "git",
    "github",
    "linux",
    "tensorflow",
    "pytorch",
    "machine learning",
    "deep learning",
    "nlp",
    "data science",
    "pandas",
    "numpy",
    "scikit-learn",
    "flask",
    "fastapi",
    "streamlit",
    "html",
    "css",
    "react",
    "node.js",
}


SKILL_PATTERNS = {
    skill: re.compile(
        r"\b" + re.escape(skill) + r"\b",
        re.IGNORECASE,
    )
    for skill in COMMON_SKILLS
}


class SkillExtractor:
    """Extract predefined technical skills."""

    @staticmethod
    def extract_skills(text: str) -> List[str]:
        if not text:
            logger.warning("Empty text supplied for skill extraction.")
            return []

        text = re.sub(r"\s+", " ", text.lower()).strip()

        found_skills = [
            skill
            for skill, pattern in SKILL_PATTERNS.items()
            if pattern.search(text)
        ]

        logger.info(
            "Extracted %d technical skills.",
            len(found_skills),
        )

        return sorted(found_skills)