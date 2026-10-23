"""Console output for solves when the application has not configured logging.

pddlstream reports progress through loggers named after its modules (all under 'pddlstream'): INFO for progress,
WARNING for problems. An application that configures logging controls this output as usual (for example,
logging.getLogger('pddlstream').setLevel(logging.INFO) to see progress). Otherwise, the solvers print their INFO
messages to stdout while they run, as pddlstream always has.
"""

import contextlib
import logging
import sys

LOGGER_NAME = 'pddlstream'


class ConsoleHandler(logging.Handler):
    """Writes messages to whatever sys.stdout is when they are logged, so code that silences stdout (Verbose) also
    silences them. Warnings start with 'Warning! '."""

    def emit(self, record):
        try:
            message = self.format(record)
            if record.levelno >= logging.WARNING:
                message = 'Warning! ' + message
            sys.stdout.write(message + '\n')
        except Exception:
            self.handleError(record)


@contextlib.contextmanager
def console_output():
    """Prints pddlstream's INFO and higher messages to stdout, unless logging is configured. Also a decorator."""
    logger = logging.getLogger(LOGGER_NAME)
    if logger.hasHandlers():  # The application configured logging, or an outer call already prints
        yield
        return
    handler, level = ConsoleHandler(), logger.level
    logger.addHandler(handler)
    if level == logging.NOTSET:
        logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)
