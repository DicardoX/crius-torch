import logging

# Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

__all__ = [
    "check_installation",
]


def check_installation():
    logger.info("Crius-torch is loaded successfully.")
