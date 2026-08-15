import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Wrapper around the Sentence Transformer model."""

    _model = None

    MODEL_NAME = "all-MiniLM-L6-v2"

    @classmethod
    def load_model(cls):
        """Load the Sentence Transformer model."""

        if cls._model is None:
            try:
                logger.info(
                    "Loading Sentence Transformer model: %s",
                    cls.MODEL_NAME,
                )

                cls._model = SentenceTransformer(cls.MODEL_NAME)

                logger.info("Embedding model loaded successfully.")

            except Exception as exc:
                logger.error(
                    "Failed to load embedding model: %s",
                    exc,
                )

                raise RuntimeError("Unable to load embedding model.") from exc

        return cls._model

    @classmethod
    def generate_embedding(cls, text: str):
        """Generate a semantic embedding for the input text."""

        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")

        model = cls.load_model()

        logger.info("Generating text embedding.")

        return model.encode(
            text.strip(),
            convert_to_tensor=False,
            show_progress_bar=False,
        )
