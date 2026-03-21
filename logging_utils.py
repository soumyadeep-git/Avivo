"""
Logging helpers for structured application logs.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any


class JsonLogFormatter(logging.Formatter):
    """Serialize log records into compact JSON for production readability."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        context = getattr(record, "context", None)
        if isinstance(context, dict):
            payload.update(context)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=True)


def configure_logging(level: str) -> None:
    """Configure root logging once using a JSON formatter."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level.upper())
    root_logger.addHandler(handler)


def log_event(logger: logging.Logger, level: int, message: str, **context: Any) -> None:
    """Emit structured logs with additional context."""
    logger.log(level, message, extra={"context": context})
