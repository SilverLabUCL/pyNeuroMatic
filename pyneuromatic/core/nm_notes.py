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

import copy
import datetime


class NMNotes:
    """Timestamped notes.

    Each note is stored as a dict with 'date' (ISO timestamp) and 'note' (text),
    and 'readonly': True for a note written by pyNeuroMatic itself (e.g. how a
    run's results were made, a transform log), which can't be edited or
    removed (#505). Used by NMFolder (session notes) and NMData (transformation
    logs). A note's text can be edited (its date stays) and a note removed (#501).
    """

    def __init__(self) -> None:
        self._entries: list[dict] = []
        # Whose notes, for a command's path (#510): (an NMObject, None), or
        # (its NMSets or NMGroups, the set's name or group's number). Set by
        # the owner each time its notes are asked for, so names are current
        self._owner: tuple | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NMNotes):
            return NotImplemented
        return self._entries == other._entries

    def __deepcopy__(self, memo: dict) -> NMNotes:
        """A copy of the notes, not of their owner (a copy's owner sets its own)."""
        result = NMNotes()
        result._entries = copy.deepcopy(self._entries, memo)
        return result

    def set_owner(self, owner, key=None) -> NMNotes:
        """Whose notes these are (see :attr:`command_path`); returns them."""
        self._owner = (owner, key)
        return self

    @property
    def command_path(self) -> str | None:
        """These notes as written in a command (#510), e.g.
        "nm.folders['Demo'].dataseries['Record'].epochs['E0'].notes" or
        "nm.folders['Demo'].data.sets.notes('Mine')"; None if their owner
        isn't in the manager (a command couldn't find them)."""
        if self._owner is None:
            return None
        from pyneuromatic.core.nm_data import nm_ref
        import pyneuromatic.core.nm_command_history as nmch
        owner, key = self._owner
        if key is not None:  # a set's or group's: its NMSets / NMGroups
            path = getattr(owner, "_nm_cmd_path", None)
            if path is None:
                return None
            return "%s.%s.notes(%r)" % (nmch.get_command_history().nm_name, path, key)
        ref = nm_ref(owner)
        if ref == "None":  # an epoch or channel: through its data series
            series = getattr(owner, "_parent", None)
            series_ref = nm_ref(series) if series is not None else "None"
            kind = ("epochs" if owner in getattr(series, "epochs", {}).values()
                    else "channels" if owner in getattr(series, "channels", {}).values()
                    else None)
            if series_ref == "None" or kind is None:
                return None
            ref = "%s.%s[%r]" % (series_ref, kind, owner.name)
        return ref + ".notes"

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

    def add(self, text: str, readonly: bool = False) -> None:
        """Add a timestamped note.

        :param text: The note text to add
        :type text: str
        :param readonly: True for a note written by pyNeuroMatic (a record of
            what it did): it can't be edited or removed.
        """
        if text is None:
            return
        if not isinstance(text, str):
            text = str(text)
        entry = {
            "date": datetime.datetime.now().isoformat(" ", "seconds"),
            "note": text,
        }
        if readonly:
            entry["readonly"] = True
        self._entries.append(entry)

    def is_readonly(self, index: int) -> bool:
        """True if note *index* was written by pyNeuroMatic (read-only)."""
        return bool(self._entries[index].get("readonly", False))

    def edit(self, index: int, text: str) -> None:
        """Change note *index*'s text; its date (when written) stays.

        :raises IndexError: If there is no note *index*.
        :raises ValueError: If note *index* is read-only.
        """
        entry = self._entries[index]
        if entry.get("readonly"):
            raise ValueError("note %d is read-only" % index)
        entry["note"] = text if isinstance(text, str) else str(text)

    def remove(self, index: int) -> dict:
        """Remove note *index*, returning it.

        :raises IndexError: If there is no note *index*.
        :raises ValueError: If note *index* is read-only.
        """
        if self._entries[index].get("readonly"):
            raise ValueError("note %d is read-only" % index)
        return self._entries.pop(index)

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
            keys = set(n.keys())
            if not {"date", "note"} <= keys <= {"date", "note", "readonly"}:
                return False
            if not isinstance(n["date"], str) or not isinstance(n["note"], str):
                return False
            if not isinstance(n.get("readonly", False), bool):
                return False
        return True
