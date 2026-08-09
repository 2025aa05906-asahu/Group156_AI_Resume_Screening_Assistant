import logging
import re
import string

import spacy

logger = logging.getLogger(__name__)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError as exc:
    raise RuntimeError(
        "spaCy model 'en_core_web_sm' is not installed."
    ) from exc


class TextPreprocessor:
    """Clean and normalize resume/JD text."""

    @staticmethod
    def clean_text(text: str) -> str:
        if text is None:
            raise ValueError("Text cannot be None.")

        if not text.strip():
            logger.warning("Empty text received for preprocessing.")
            return ""

        text = text.lower()

        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"\S+@\S+", "", text)
        text = re.sub(r"\d+", " ", text)

        text = text.translate(
            str.maketrans("", "", string.punctuation)
        )

        text = re.sub(r"\s+", " ", text)

        return text.strip()

    @staticmethod
    def preprocess(text: str) -> str:
        logger.info("Starting text preprocessing.")

        cleaned_text = TextPreprocessor.clean_text(text)

        if not cleaned_text:
            return ""

        doc = nlp(cleaned_text)

        tokens = []

        for token in doc:
            if (
                token.is_stop
                or token.is_punct
                or token.is_space
                or not token.is_alpha
            ):
                continue

            lemma = token.lemma_.strip()

            if lemma:
                tokens.append(lemma)

        result = " ".join(tokens)

        logger.info("Text preprocessing completed.")

        return result