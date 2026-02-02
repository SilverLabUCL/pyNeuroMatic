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
import inspect
import logging
from collections import deque

from colorama import Fore

import pyneuromatic.core.nm_preferences as nmp


class NMHistoryBufferHandler(logging.Handler):
    """Logging handler that stores records in a fixed-size in-memory buffer.

    Each record is stored as a dict with keys: date, level, path, message.
    When the buffer is full, the oldest entry is discarded.
    """

    def __init__(self, maxlen: int = 10000) -> None:
        super().__init__()
        self._buffer: deque[dict[str, str]] = deque(maxlen=maxlen)

    def emit(self, record: logging.LogRecord) -> None:
        entry = {
            "date": str(datetime.datetime.now()),
            "level": record.levelname,
            "path": getattr(record, "path", ""),
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

    Replicates the output format of history(): path prefix,
    optional title, and red coloring for WARNING/ERROR levels.
    """

    def emit(self, record: logging.LogRecord) -> None:
        path = getattr(record, "path", "")
        title = getattr(record, "title", "")
        msg = record.getMessage()

        if path and path.upper() != "NONE":
            h = path + ": " + msg
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
    and colorama-colored console output.

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
        self._logger.propagate = False  # prevent duplicate output to root logger

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
        path: str = "",
        level: int = logging.INFO,
        quiet: bool = False,
    ) -> str:
        """Log a message to the history buffer and optionally to console.

        :param message: the message to log.
        :type message: str
        :param title: message title (e.g. 'ALERT' or 'ERROR').
        :type title: str
        :param path: path string (e.g. 'nm.project0.folder0').
        :type path: str
        :param level: Python logging level (e.g. logging.INFO).
        :type level: int
        :param quiet: if True, suppress console output for this call only.
        :type quiet: bool
        :return: formatted history string.
        :rtype: str
        """
        # build return string matching current history() format
        if path and path.upper() != "NONE":
            h = path + ": " + message
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
            extra={"path": path, "title": title},
        )

        if old_level is not None:
            self._console_handler.setLevel(old_level)

        return h

    @property
    def buffer(self) -> list[dict[str, str]]:
        """Return a copy of the history buffer.

        Each entry is a dict with keys: date, level, path, message.
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
            path = entry.get("path", "")
            msg = entry.get("message", "")
            level = entry.get("level", "")

            if path:
                line = path + ": " + msg
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


# Module-level history instance (set by NMManager.__init__ via set_history())
_nm_history: NMHistory | None = None


def set_history(history_instance: NMHistory | None) -> None:
    """Register the active NMHistory instance.

    Called by NMManager.__init__() to connect the centralized
    history log to the history() function.

    :param history_instance: NMHistory instance or None.
    :type history_instance: NMHistory | None
    """
    global _nm_history
    _nm_history = history_instance


def history_change_str(
    param_name: str,
    old_value: object,
    new_value: object,
) -> str:
    """Create history text for variables that have changed.

    :param param_name: name of parameter that has changed.
    :type param_name: str
    :param old_value: old value of parameter.
    :type old_value: object
    :param new_value: new value of parameter.
    :type new_value: object
    :return: history string containing the change.
    :rtype: str
    """
    if not isinstance(param_name, str):
        param_name = str(param_name)
    if not isinstance(old_value, str):
        if old_value is None:
            old_value = "None"
        else:
            old_value = "'%s'" % old_value
    if not isinstance(new_value, str):
        if new_value is None:
            new_value = "None"
        else:
            new_value = "'%s'" % new_value
    h = "changed " + param_name + " from " + old_value + " to " + new_value
    return h


def history(
    message: str,
    title: str = "",
    path: str | None = None,
    frame: int = 1,
    red: bool = False,
    quiet: bool = False,
) -> str:
    """Print message to NM history.

    This function checks nmp.QUIET internally, so callers don't need to.

    :param message: message to print.
    :type message: str
    :param title: message title (e.g. 'ALERT' or 'ERROR').
    :type title: str
    :param path: function path, pass '' for default or 'NONE' for none.
    :type path: str | None
    :param frame: inspect frame # for creating path.
    :type frame: int
    :param red: True to print red, False to print black.
    :type red: bool
    :param quiet: True to not print message, False to print.
    :type quiet: bool
    :return: history string.
    :rtype: str
    """
    # Check global QUIET flag
    if nmp.QUIET:
        quiet = True

    if not isinstance(message, str):
        return ""
    if not isinstance(frame, int) or frame < 0:
        frame = 1
    if path is None:
        path = get_path(inspect.stack(), frame=frame)
    elif not isinstance(path, str):
        path = ""

    # determine log level from title/red
    if isinstance(title, str) and title.upper() == "ERROR":
        level = logging.ERROR
    elif isinstance(title, str) and title.upper() == "ALERT" or red:
        level = logging.WARNING
    else:
        level = logging.INFO

    # delegate to NMHistory if available
    if _nm_history is not None:
        return _nm_history.log(
            message, title=title, path=path, level=level, quiet=quiet
        )

    # fallback: original print() behavior (before NMManager is created)
    if path:
        h = path + ": " + message
    else:
        h = message
    if isinstance(title, str) and len(title) > 0:
        h = title + ": " + h
    if not quiet:
        if red:
            print(Fore.RED + h + Fore.BLACK)
        else:
            print(h)
    return h


def alert(
    message: str,
    path: str | None = None,
    quiet: bool = False,
) -> str:
    """Log an alert message.

    Convenience wrapper for history() with ALERT title.

    :param message: message to print.
    :type message: str
    :param path: function path.
    :type path: str | None
    :param quiet: True to not print message.
    :type quiet: bool
    :return: history string.
    :rtype: str
    """
    return history(message, title="ALERT", path=path, red=True, quiet=quiet)


def error(
    message: str,
    path: str | None = None,
    quiet: bool = False,
) -> str:
    """Log an error message.

    Convenience wrapper for history() with ERROR title.

    :param message: message to print.
    :type message: str
    :param path: function path.
    :type path: str | None
    :param quiet: True to not print message.
    :type quiet: bool
    :return: history string.
    :rtype: str
    """
    return history(message, title="ERROR", path=path, red=True, quiet=quiet)


def _get_frame_from_stack(stack: list, frame: int = 1) -> inspect.FrameType | None:
    """Extract and validate a frame from the stack.

    :param stack: stack list from inspect.stack().
    :type stack: list
    :param frame: frame index to extract.
    :type frame: int
    :return: frame object or None if invalid.
    :rtype: inspect.FrameType | None
    """
    if not stack:
        return None
    if not isinstance(frame, int) or frame < 0:
        frame = 1
    if len(stack) <= frame:
        return None
    f = stack[frame].frame
    if not inspect.isframe(f):
        return None
    return f


def get_path(
    stack: list,
    frame: int = 1,
    package: str = "nm",
) -> str:
    """Create function ancestry path.

    :param stack: stack list from inspect.stack().
    :type stack: list
    :param frame: frame index for creating path.
    :type frame: int
    :param package: package prefix, e.g. 'nm'.
    :type package: str
    :return: path string like 'nm.ClassName.method_name'.
    :rtype: str
    """
    path = [package] if package else []
    c = get_class_from_stack(stack, frame=frame)
    m = get_method_from_stack(stack, frame=frame)
    if c:
        path.append(c)
    if m:
        path.append(m)
    return ".".join(path)


def get_class_from_stack(
    stack: list,
    frame: int = 1,
    module: bool = False,
) -> str:
    """Extract class name from stack.

    :param stack: stack list from inspect.stack().
    :type stack: list
    :param frame: frame index for creating path.
    :type frame: int
    :param module: if True, include module name as prefix.
    :type module: bool
    :return: class name (e.g. 'NMObject' or 'nm_object.NMObject').
    :rtype: str
    """
    f = _get_frame_from_stack(stack, frame=frame)
    if f is None:
        return ""
    if "self" not in f.f_locals:
        return ""
    cls = f.f_locals["self"].__class__
    if module:
        return cls.__module__ + "." + cls.__name__
    return cls.__name__


def get_method_from_stack(
    stack: list,
    frame: int = 1,
) -> str:
    """Extract method name from stack.

    :param stack: stack list from inspect.stack().
    :type stack: list
    :param frame: frame index for creating path.
    :type frame: int
    :return: method name.
    :rtype: str
    """
    f = _get_frame_from_stack(stack, frame=frame)
    if f is None:
        return ""
    return f.f_code.co_name
