"""Structured logging configuration using structlog."""

from __future__ import annotations

import logging
import sys

import structlog


def _renderer(use_json: bool):
    """Pick the renderer for the given environment."""
    if use_json:
        return structlog.processors.JSONRenderer(sort_keys=True)
    return structlog.dev.ConsoleRenderer()


def configure_logging(level: int = logging.INFO, json_output: bool = False) -> None:
    """
    Configure structlog-based logging.

    Args:
        level: Root log level.
        json_output: Emit JSON lines instead of the pretty console renderer
            (recommended for production / containerized environments).
    """
    timestamper = structlog.processors.TimeStamper(fmt="iso")

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.stdlib.add_logger_name,
            timestamper,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _renderer(json_output),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Make stdlib logging (sqlalchemy, uvicorn, ...) route through structlog
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=_renderer(json_output),
        foreign_pre_chain=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.stdlib.add_logger_name,
            timestamper,
            structlog.processors.StackInfoRenderer(),
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = []
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet noisy third-party loggers
    for noisy in ("uvicorn.access", "asyncio", "aiosqlite"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structured logger bound to the calling module."""
    return structlog.get_logger(name)
