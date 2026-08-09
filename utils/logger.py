import logging
import os


def setup_logger():
    """Configure application logging."""

    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
        handlers=[
            logging.FileHandler(
                "logs/application.log"
            ),
            logging.StreamHandler(),
        ],
    )