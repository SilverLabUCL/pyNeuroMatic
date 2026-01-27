# -*- coding: utf-8 -*-
"""
Centralized history logging for pyNeuroMatic.

Provides NMHistory, a wrapper around Python's logging module that maintains
an in-memory history buffer and colorama-colored console output, analogous
to Igor Pro's history command window.

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.

If you use this software in your research, please cite:
Rothman JS and Silver RA (2018) NeuroMatic: An Integrated Open-Source
Software Toolkit for Acquisition, Analysis and Simulation of
Electrophysiological Data. Front. Neuroinform. 12:14.
doi: 10.3389/fninf.2018.00014

Copyright (c) 2026 The Silver Lab, University College London.
Licensed under MIT License - see LICENSE file for details.

Original NeuroMatic: https://github.com/SilverLabUCL/NeuroMatic
Website: https://github.com/SilverLabUCL/pyNeuroMatic
Paper: https://doi.org/10.3389/fninf.2018.00014
"""
from __future__ import annotations

import datetime
import logging
from collections import deque

from colorama import Fore


class NMHistoryBufferHandler(logging.Handler):
    """Logging handler that stores records in a fixed-size in-memory buffer.

    Each record is stored as a dict with keys: date, level, treepath, message.
    When the buffer is full, the oldest entry is discarded.
    """

    def __init__(self, maxlen: int = 10000) -> None:
        super().__init__()
        self._buffer: deque[dict[str, str]] = deque(maxlen=maxlen)

    def emit(self, record: logging.LogRecord) -> None:
        entry = {
            "date": str(datetime.datetime.now()),
            "level": record.levelname,
            "treepath": getattr(record, "treepath", ""),
            "message": record.getMessage(),
        }
        self._buffer.append(entry)

    @property
    def buffer(self) -> list[dict[str, str]]:
        """Return a copy of the buffer as a list."""
        return list(self._buffer)

    def clear(self) -> None:
        """Clear all entries from the buffer."""
        self._buffer.clear()


class NMConsoleHandler(logging.Handler):
    """Logging handler that prints to console with colorama colors.

    Replicates the output format of nmu.history(): treepath prefix,
    optional title, and red coloring for WARNING/ERROR levels.
    """

    def emit(self, record: logging.LogRecord) -> None:
        tp = getattr(record, "treepath", "")
        title = getattr(record, "title", "")
        msg = record.getMessage()

        if tp and tp.upper() != "NONE":
            h = tp + ": " + msg
        else:
            h = msg

        if title:
            h = title + ": " + h

        if record.levelno >= logging.WARNING:
            print(Fore.RED + h + Fore.BLACK)
        else:
            print(h)


class NMHistory:
    """Centralized history log for pyNeuroMatic.

    Wraps Python's logging module to provide an in-memory history buffer
    and colorama-colored console output. Analogous to Igor Pro's history
    command window.

    Access via NMManager.history property.

    :param buffer_size: maximum number of entries in the history buffer.
    :type buffer_size: int
    :param quiet: if True, suppress console output (buffer still records).
    :type quiet: bool
    """

    def __init__(
        self,
        buffer_size: int = 10000,
        quiet: bool = False,
    ) -> None:
        self._logger = logging.getLogger("pyneuromatic")
        self._logger.setLevel(logging.DEBUG)

        # prevent duplicate handlers if NMHistory is created multiple times
        self._logger.handlers.clear()

        # in-memory buffer handler - always captures everything
        self._buffer_handler = NMHistoryBufferHandler(maxlen=buffer_size)
        self._buffer_handler.setLevel(logging.DEBUG)
        self._logger.addHandler(self._buffer_handler)

        # console handler - respects quiet flag
        self._console_handler = NMConsoleHandler()
        if quiet:
            self._console_handler.setLevel(logging.CRITICAL + 1)
        else:
            self._console_handler.setLevel(logging.DEBUG)
        self._logger.addHandler(self._console_handler)

    def log(
        self,
        message: str,
        title: str = "",
        tp: str = "",
        level: int = logging.INFO,
        quiet: bool = False,
    ) -> str:
        """Log a message to the history buffer and optionally to console.

        :param message: the message to log.
        :type message: str
        :param title: message title (e.g. 'ALERT' or 'ERROR').
        :type title: str
        :param tp: treepath string (e.g. 'nm.project0.folder0').
        :type tp: str
        :param level: Python logging level (e.g. logging.INFO).
        :type level: int
        :param quiet: if True, suppress console output for this call only.
        :type quiet: bool
        :return: formatted history string.
        :rtype: str
        """
        # build return string matching current nmu.history() format
        if tp and tp.upper() != "NONE":
            h = tp + ": " + message
        else:
            h = message
        if title:
            h = title + ": " + h

        # temporarily raise console handler level if quiet for this call
        old_level = None
        if quiet:
            old_level = self._console_handler.level
            self._console_handler.setLevel(logging.CRITICAL + 1)

        self._logger.log(
            level,
            message,
            extra={"treepath": tp, "title": title},
        )

        if old_level is not None:
            self._console_handler.setLevel(old_level)

        return h

    @property
    def buffer(self) -> list[dict[str, str]]:
        """Return a copy of the history buffer.

        Each entry is a dict with keys: date, level, treepath, message.
        """
        return self._buffer_handler.buffer

    @property
    def buffer_size(self) -> int:
        """Return the maximum buffer size."""
        return self._buffer_handler._buffer.maxlen or 0

    def clear(self) -> None:
        """Clear the history buffer."""
        self._buffer_handler.clear()

    def history_print(self, last_n: int = 0) -> None:
        """Print history buffer contents to console.

        :param last_n: number of most recent entries to print (0 for all).
        :type last_n: int
        """
        entries = self.buffer
        if last_n > 0:
            entries = entries[-last_n:]
        for entry in entries:
            tp = entry.get("treepath", "")
            msg = entry.get("message", "")
            level = entry.get("level", "")

            if tp:
                line = tp + ": " + msg
            else:
                line = msg

            if level in ("WARNING", "ERROR"):
                line = level + ": " + line
                print(Fore.RED + line + Fore.BLACK)
            else:
                print(line)

    @property
    def quiet(self) -> bool:
        """True if console output is suppressed."""
        return self._console_handler.level > logging.CRITICAL

    @quiet.setter
    def quiet(self, value: bool) -> None:
        if value:
            self._console_handler.setLevel(logging.CRITICAL + 1)
        else:
            self._console_handler.setLevel(logging.DEBUG)
