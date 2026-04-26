import logging
import sys
from pathlib import Path


LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def configure_logging(log_dir, log_name="run.log"):
    """Configure root logging to console and one file path."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_name

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    logging.disable(logging.NOTSET)
    for logger in logging.Logger.manager.loggerDict.values():
        if isinstance(logger, logging.Logger):
            logger.disabled = False

    formatter = logging.Formatter(LOG_FORMAT)
    console_handlers = [
        handler
        for handler in root.handlers
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
    ]
    if console_handlers:
        console_handlers[0].setFormatter(formatter)
        console_handlers[0]._project_console = True
    elif not any(getattr(handler, "_project_console", False) for handler in root.handlers):
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        console._project_console = True
        root.addHandler(console)

    for handler in list(root.handlers):
        if getattr(handler, "_project_file", False):
            root.removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler._project_file = True
    root.addHandler(file_handler)
    return log_path
