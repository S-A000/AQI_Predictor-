from __future__ import annotations

import json
import logging
import queue
import sys
import threading
from datetime import datetime, timezone
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path
from typing import Any, Dict
from contextvars import ContextVar

# ==========================================================
# Context Variables (For Request / Correlation IDs)
# ==========================================================
# Defaults to "SYSTEM" if not set during a request/pipeline run
request_id_var: ContextVar[str] = ContextVar("request_id", default="SYSTEM")
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="SYSTEM")

# ==========================================================
# Configuration & Paths
# ==========================================================
LOG_DIRECTORY = Path("logs")
LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)

MAX_BYTES = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5
LOG_LEVEL = logging.INFO

# Logger names (or dotted-path suffixes) whose records belong in api.log.
# A record matches if its logger name is exactly one of these, or ends
# with ".<name>" — so both `get_logger("api")` and
# `get_logger(__name__)` from a module like `src.ingestion.weather_client`
# are routed correctly.
API_LOGGER_NAMES: tuple[str, ...] = ("api", "weather_client", "aqi_client")

# ==========================================================
# Custom Formatters
# ==========================================================

class ColorFormatter(logging.Formatter):
    """Adds ANSI colors to console output based on log level."""
    
    GREY = "\x1b[38;20m"
    BLUE = "\x1b[34;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    FORMAT_STR = (
        "%(asctime)s | %(levelname)-8s | "
        "[CorrID: %(correlation_id)s] | %(name)-20s | %(message)s"
    )

    COLORS = {
        logging.DEBUG: GREY,
        logging.INFO: BLUE,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD_RED
    }

    def format(self, record: logging.LogRecord) -> str:
        # Inject context variables
        record.request_id = request_id_var.get()
        record.correlation_id = correlation_id_var.get()
        
        log_fmt = self.COLORS.get(record.levelno, self.RESET) + self.FORMAT_STR + self.RESET
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


class JSONFormatter(logging.Formatter):
    """Formats log records as JSON for structured file logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id_var.get(),
            "correlation_id": correlation_id_var.get(),
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_obj)

# ==========================================================
# Custom Filters
# ==========================================================

class APILogFilter(logging.Filter):
    """
    Restricts log records to those emitted by API-related loggers.

    A record passes if its logger name is exactly one of
    ``logger_names``, or is a dotted sub-logger of one of them
    (e.g. ``"src.ingestion.weather_client"`` matches ``"weather_client"``).
    This lets ``api_handler`` receive only API traffic (Weather API,
    AQI API, HTTP requests/responses/retries/failures) without
    duplicating every log record into ``api.log``.
    """

    def __init__(self, logger_names: tuple[str, ...] = API_LOGGER_NAMES) -> None:
        super().__init__()
        self._logger_names = logger_names

    def filter(self, record: logging.LogRecord) -> bool:
        name = record.name
        return any(
            name == keyword or name.endswith(f".{keyword}")
            for keyword in self._logger_names
        )

# ==========================================================
# Logger Factory
# ==========================================================

class LoggerFactory:
    """
    Singleton Factory to configure the root logger with a QueueListener
    for thread-safe and async logging.
    """
    _configured = False
    _listener: QueueListener | None = None
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def configure(cls) -> None:
        # Fast path: already configured, no lock needed.
        if cls._configured:
            return

        # Slow path: guard actual (re)configuration so concurrent
        # get_logger() calls from multiple threads at startup can never
        # both pass the check above and each build their own
        # queue/listener/handlers.
        with cls._lock:
            if cls._configured:
                return

            # Create central queue for async logging
            log_queue: queue.Queue = queue.Queue(-1)

            # ------------------------------
            # Console Handler (Colored)
            # ------------------------------
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(ColorFormatter())
            console_handler.setLevel(LOG_LEVEL)

            # ------------------------------
            # File Handlers (JSON)
            # ------------------------------
            def create_file_handler(filename: str, level: int) -> RotatingFileHandler:
                handler = RotatingFileHandler(
                    LOG_DIRECTORY / filename,
                    maxBytes=MAX_BYTES,
                    backupCount=BACKUP_COUNT,
                    encoding="utf-8"
                )
                handler.setFormatter(JSONFormatter())
                handler.setLevel(level)
                return handler

            pipeline_handler = create_file_handler("pipeline.log", logging.INFO)

            api_handler = create_file_handler("api.log", logging.INFO)
            # Only API-related loggers (api, weather_client, aqi_client,
            # and their dotted submodule paths) reach api.log.
            api_handler.addFilter(APILogFilter())

            error_handler = create_file_handler("error.log", logging.ERROR)
            error_handler.setLevel(logging.ERROR)  # Only captures ERROR and above

            # ------------------------------
            # Queue Listener Setup
            # ------------------------------
            # The listener runs on a separate background thread and writes to files/console.
            # respect_handler_level=True means each handler's own setLevel()
            # (and now, each handler's own filters) still apply — so
            # error_handler only ever sees ERROR/CRITICAL records, and
            # api_handler only ever sees records from API-related loggers.
            cls._listener = QueueListener(
                log_queue,
                console_handler,
                pipeline_handler,
                api_handler,
                error_handler,
                respect_handler_level=True
            )
            cls._listener.start()

            # ------------------------------
            # Root Logger Setup
            # ------------------------------
            root_logger = logging.getLogger()
            root_logger.setLevel(LOG_LEVEL)

            # Remove any existing handlers to prevent duplicates
            root_logger.handlers.clear()

            # Add ONLY the QueueHandler to the root logger
            queue_handler = QueueHandler(log_queue)
            root_logger.addHandler(queue_handler)

            cls._configured = True

    @classmethod
    def stop_listener(cls) -> None:
        """
        Safely stop the listener before application shutdown.

        Idempotent and thread-safe: calling this multiple times, or
        calling it when the factory was never configured, is a no-op
        after the first successful stop. ``QueueListener.stop()``
        enqueues a sentinel and joins the listener thread, which
        guarantees every record enqueued *before* this call has already
        been handed to every handler (and therefore written/flushed)
        before the method returns — no queued log is lost.
        """
        with cls._lock:
            if cls._listener is None:
                return

            listener = cls._listener
            cls._listener = None

            # Blocks until the background thread has drained the queue
            # and processed every record enqueued before the sentinel.
            listener.stop()

            # Flush and close every handler explicitly so buffered file
            # writes are guaranteed to hit disk and file descriptors are
            # released cleanly.
            for handler in listener.handlers:
                try:
                    handler.flush()
                    handler.close()
                except Exception:  # noqa: BLE001
                    # Never let a handler failure block shutdown.
                    pass

            # Detach the queue handler from the root logger and reset
            # state so a subsequent get_logger() call can safely
            # reconfigure logging from a clean slate (useful for tests
            # and for graceful restart scenarios).
            root_logger = logging.getLogger()
            root_logger.handlers.clear()

            cls._configured = False

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        cls.configure()
        return logging.getLogger(name)

# ==========================================================
# Public Functions
# ==========================================================

def get_logger(name: str) -> logging.Logger:
    """
    Return application logger.
    
    Example:
    --------
    logger = get_logger(__name__)
    logger.info("Pipeline started")
    """
    return LoggerFactory.get_logger(name)

def shutdown_logger() -> None:
    """Call this gracefully when the application shuts down."""
    LoggerFactory.stop_listener()