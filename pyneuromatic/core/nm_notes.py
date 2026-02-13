# -*- coding: utf-8 -*-
"""
NMNotes module.

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


class NMNotes:
    """Timestamped append-only notes.

    Each note is stored as a dict with 'date' (ISO timestamp) and 'note' (text).
    Used by NMFolder (session notes) and NMData (transformation logs).
    """

    def __init__(self) -> None:
        self._entries: list[dict] = []

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NMNotes):
            return NotImplemented
        return self._entries == other._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def __getitem__(self, index):
        return self._entries[index]

    @property
    def note(self) -> str:
        """Return the text of the most recent note, or empty string if none."""
        if self._entries:
            return self._entries[-1].get("note", "")
        return ""

    @note.setter
    def note(self, text: str) -> None:
        """Add a new note with the given text."""
        self.add(text)

    def add(self, text: str) -> None:
        """Add a timestamped note.

        :param text: The note text to add
        :type text: str
        """
        if text is None:
            return
        if not isinstance(text, str):
            text = str(text)
        entry = {
            "date": datetime.datetime.now().isoformat(" ", "seconds"),
            "note": text,
        }
        self._entries.append(entry)

    def clear(self) -> None:
        """Clear all notes."""
        self._entries = []

    def print_all(self) -> None:
        """Print all notes to stdout."""
        for n in self._entries:
            date = n.get("date", "")
            text = n.get("note", "")
            print(f"{date}  {text}")

    @staticmethod
    def ok(notes: list[dict]) -> bool:
        """Validate notes format.

        :param notes: List of note dicts to validate
        :type notes: list[dict]
        :return: True if format is valid
        :rtype: bool
        """
        if not isinstance(notes, list):
            return False
        for n in notes:
            if not isinstance(n, dict):
                return False
            keys = n.keys()
            if len(keys) != 2 or "date" not in keys or "note" not in keys:
                return False
            if not isinstance(n["date"], str) or not isinstance(n["note"], str):
                return False
        return True
