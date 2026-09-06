# ABSOLUTE PATH: src/ipomdp/telemetry/logger.py
# ==============================================================================
# STRUCTURED JSONL & CONSOLE LOGGING PIPELINE
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Exception Traceback Serialization:
#    - Captures record.exc_info using self.formatException() to ensure full stack
#      traces are serialized into machine-readable JSONL files.
# ==============================================================================

from datetime import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """Formats log records as JSON lines for structured machine parsing."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "telemetry"):
            log_obj["telemetry"] = record.telemetry

        return json.dumps(log_obj)


def setup_logger(name: str, log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """Initializes console and JSONL log handlers."""
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(level)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Console Handler
    c_handler = logging.StreamHandler(stream=sys.stdout)
    c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

    # Machine-Readable JSONL File Handler
    f_handler = logging.FileHandler(Path(log_dir) / f"{name}.jsonl")
    f_handler.setFormatter(JSONFormatter())
    logger.addHandler(f_handler)

    return logger


def log_telemetry(logger: logging.Logger, msg: str, telemetry: Dict[str, Any], level: int = logging.INFO):
    """Logs explicit dictionary telemetry to structured JSON formatter."""
    logger.log(level, msg, extra={"telemetry": telemetry})
