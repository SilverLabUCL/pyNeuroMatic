# -*- coding: utf-8 -*-
"""
NMSettings - General pyNeuroMatic settings, saved between sessions.

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

from pathlib import Path

from pyneuromatic.tools.nm_tool_config import NMToolConfig

# How tables show NaN values
TABLE_NAN_TEXT = "NaN"
TABLE_NAN_GREY = "NaN (grey)"
TABLE_NAN_BLANK = "blank"


# The Plot's trace line styles and symbols (#465)
TRACE_LINES = ("solid", "dash", "dot", "dash-dot", "none")
TRACE_SYMBOLS = ("none", "circle", "square", "triangle", "diamond", "plus", "cross", "star")
# While a tool runs, show each epoch: no/off, yes+wait, or yes+ask (#468)
RUN_DISPLAYS = ("off", "each", "ask")


class NMSettings(NMToolConfig):
    """General settings for the whole application (not one tool).

    Same schema-based format as the tool configs (validated values,
    ``to_dict()`` / ``from_dict()``, TOML ``save()`` / ``load()``). The user's
    settings live in :meth:`user_path`; :meth:`load_user` reads them (e.g. at
    start-up) and remembers the file, so :meth:`save_user` writes changes
    back. Settings that were never loaded from a file are not saved, so
    scripts and tests leave the user's file alone.

    Parameters:
        table_nan: How tables show NaN values: ``"NaN"``, ``"NaN (grey)"``
            (shown in light grey) or ``"blank"`` (an empty cell).
        confirm_quit: Ask before quitting when the session has data or
            history to lose.
        ask_import_options: Before importing a file, ask what to import
            (folder name, prefix, channels, epochs, X range).
        plot_trace_color, plot_trace_width, plot_trace_line,
        plot_trace_symbol, plot_trace_symbol_size: How the Plot draws the
            current trace: colour (e.g. "#000000"), line width (pixels),
            line style and symbol (see ``TRACE_LINES``, ``TRACE_SYMBOLS``).
        plot_overlay_color, plot_overlay_width: How it draws overlay traces.
        run_display: While a tool runs, show each epoch in the Plot:
            ``"off"`` (fastest), ``"each"`` (then wait *run_delay*) or
            ``"ask"`` (then ask to continue).
        run_delay: Seconds to wait after showing each epoch ("each").
        stats_digits: Significant digits of Stats values shown (Stats1's
            Details, Stats2's tables and Details); copying keeps them all.
    """

    _TOML_TYPE = "nm_settings"
    _schema = {
        "table_nan": {"type": str, "default": TABLE_NAN_TEXT,
                      "choices": [TABLE_NAN_TEXT, TABLE_NAN_GREY, TABLE_NAN_BLANK]},
        "confirm_quit": {"type": bool, "default": True},
        "ask_import_options": {"type": bool, "default": True},
        # The Plot's traces (#465)
        "plot_trace_color": {"type": str, "default": "#000000"},
        "plot_trace_width": {"type": float, "default": 2.0, "min": 0.5, "max": 10.0},
        "plot_trace_line": {"type": str, "default": "solid", "choices": list(TRACE_LINES)},
        "plot_trace_symbol": {"type": str, "default": "none", "choices": list(TRACE_SYMBOLS)},
        "plot_trace_symbol_size": {"type": float, "default": 6.0, "min": 1.0, "max": 30.0},
        "plot_overlay_color": {"type": str, "default": "#78909c"},
        "plot_overlay_width": {"type": float, "default": 1.0, "min": 0.5, "max": 10.0},
        # Tool runs: show each epoch as it's analysed (#468)
        "run_display": {"type": str, "default": "off", "choices": list(RUN_DISPLAYS)},
        "run_delay": {"type": float, "default": 0.0, "min": 0.0, "max": 10.0},
        # Stats values shown (#471)
        "stats_digits": {"type": int, "default": 6, "min": 2, "max": 10},
    }

    def __init__(self) -> None:
        super().__init__()
        object.__setattr__(self, "_path", None)  # the file to save to, if loaded

    def _validate(self, key: str, value: object) -> object:
        # A whole number for a width or size, e.g. plot_trace_width = 2
        if (self._schema[key]["type"] is float and isinstance(value, int)
                and not isinstance(value, bool)):
            value = float(value)
        return super()._validate(key, value)

    @staticmethod
    def user_path() -> Path:
        """The user's settings file: ~/.pyneuromatic/settings.toml."""
        return Path.home() / ".pyneuromatic" / "settings.toml"

    @property
    def path(self) -> Path | None:
        """The settings file changes are saved to (None if not loaded)."""
        return self._path

    def load_user(self, path: str | Path | None = None) -> bool:
        """Read the user's settings (if the file exists) and save changes
        there from now on. Returns True if a file was read."""
        path = Path(path) if path is not None else self.user_path()
        object.__setattr__(self, "_path", path)
        if not path.exists():
            return False
        loaded = type(self).load(path)
        for key in self._schema:
            setattr(self, key, getattr(loaded, key))
        return True

    def save_user(self) -> Path | None:
        """Write the settings to the file they were loaded from, if any."""
        if self._path is None:
            return None
        self._path.parent.mkdir(parents=True, exist_ok=True)
        return self.save(self._path)


__all__ = ["NMSettings", "TABLE_NAN_BLANK", "TABLE_NAN_GREY", "TABLE_NAN_TEXT",
           "RUN_DISPLAYS", "TRACE_LINES", "TRACE_SYMBOLS"]
