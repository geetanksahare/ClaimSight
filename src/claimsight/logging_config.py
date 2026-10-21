import logging
import sys

_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logger once with stdout stream handler."""
    global _configured
    if _configured:
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    if not root_logger.handlers:
        root_logger.addHandler(handler)
    else:
        root_logger.handlers = [handler]

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger after ensuring root logging is configured."""
    if not _configured:
        configure_logging()
    return logging.getLogger(name)
