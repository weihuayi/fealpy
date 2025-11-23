
import logging


class StringListHandler(logging.Handler):
    """A handler that collects logs into a list.

    Attributes:
        log_list (list): log message list.
    """
    FORMAT = '%(asctime)s %(levelname)s - %(message)s'
    FORMATTER = logging.Formatter(FORMAT, datefmt="%d-%m-%Y %H:%M:%S")

    def __init__(self, log_list: list[str], /):
        """
        Args:
            log_list (list): The list to store message.
        """
        super().__init__()
        self.log_list = log_list # refer
        self.setFormatter(self.FORMATTER)

    def emit(self, record):
        """Output log message.

        Args:
            record (LogRecord): Log recording object.
        """
        try:
            message = self.format(record)
            self.log_list.append(message)
        except Exception:
            self.handleError(record)
