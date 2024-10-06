import logging

from rich.logging import RichHandler
from rich.highlighter import JSONHighlighter


class CustomFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: "[bright_green][%(asctime)s][blue] | %(levelname)s[white] | %(message)s",
        logging.INFO: "[bright_green][%(asctime)s][blue] | %(levelname)s[white]  | %(message)s",
        logging.WARNING: "[bright_green][%(asctime)s][blue] | %(levelname)s[white] | [bright_yellow]%(message)s",
        logging.ERROR: "[bright_green][%(asctime)s][blue] | %(levelname)s[white] | [bright_red]%(message)s",
        logging.CRITICAL: "[bright_green][%(asctime)s][blue] | %(levelname)s[white] | [blue]%(message)s",
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%H:%M:%S")
        return formatter.format(record)


logger = logging.getLogger("my_logger")
logger.handlers.clear()

ch = RichHandler(
    level=0, show_level=False, show_time=False, markup=True, rich_tracebacks=True, omit_repeated_times=False
)
ch.setLevel(logging.INFO)
ch.setFormatter(CustomFormatter())
ch.highlighter = JSONHighlighter()

logger.addHandler(ch)
logger.setLevel(logging.INFO)
