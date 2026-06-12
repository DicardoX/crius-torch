from crius.global_envs import logger

__all__ = [
    "check_installation",
]


def check_installation():
    logger.info("Crius-torch is loaded successfully.")
