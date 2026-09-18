import logging
import re

_SECRET_KEYS = re.compile(r"(api[_-]?key|password|token|secret|authorization)\s*[:=]\s*[^,\s]+", re.I)
_BASE64 = re.compile(r"(?:data:image/[^;]+;base64,|[A-Za-z0-9+/]{200,}={0,2})")


class SafeFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        rendered = _BASE64.sub("[REDACTED_IMAGE_DATA]", rendered)
        return _SECRET_KEYS.sub(r"\1=[REDACTED]", rendered)


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(SafeFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level.upper())
    return logger
