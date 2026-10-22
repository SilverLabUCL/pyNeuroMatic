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
    """

    _TOML_TYPE = "nm_settings"
    _schema = {
        "table_nan": {"type": str, "default": TABLE_NAN_TEXT,
                      "choices": [TABLE_NAN_TEXT, TABLE_NAN_GREY, TABLE_NAN_BLANK]},
    }

    def __init__(self) -> None:
        super().__init__()
        object.__setattr__(self, "_path", None)  # the file to save to, if loaded

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


__all__ = ["NMSettings", "TABLE_NAN_BLANK", "TABLE_NAN_GREY", "TABLE_NAN_TEXT"]
