# -*- coding: utf-8 -*-
"""
NMToolConfig - Schema-based configuration base class for NMTool subclasses.

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

import sys
from pathlib import Path

# TOML compatibility layer (same pattern as nm_workspace.py)
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None  # type: ignore

try:
    import tomli_w
except ImportError:
    tomli_w = None  # type: ignore


class NMToolConfig:
    """Schema-based configuration base class for NMTool subclasses.

    Each subclass declares ``_schema`` — a class-level dict mapping parameter
    names to a spec dict — and ``_TOML_TYPE`` (a unique string identifying the
    config type in saved files).  The base class then provides:

    - Validated attribute setting (``__setattr__``) with clear error messages.
    - ``to_dict()`` / ``from_dict()`` serialisation.
    - ``save()`` / ``load()`` TOML round-trip with a ``[pyneuromatic]`` header.

    Spec dict keys per parameter:

    - ``"type"`` *(required)*: Expected Python type (``bool``, ``int``,
      ``float``, ``str``).
    - ``"default"`` *(required)*: Value used when the config is constructed.
    - ``"min"`` *(optional)*: Inclusive lower bound for numeric types.
    - ``"max"`` *(optional)*: Inclusive upper bound for numeric types.
    - ``"choices"`` *(optional)*: List of allowed values.

    Example subclass::

        class NMToolStatsConfig(NMToolConfig):
            _TOML_TYPE = "stats_config"
            _schema = {
                "ignore_nans": {"type": bool, "default": True},
                "alpha":        {"type": float, "default": 0.05,
                                 "min": 0.0, "max": 1.0},
            }
    """

    _TOML_TYPE: str = "tool_config"   # override in each subclass
    _TOML_VERSION: str = "1"
    _schema: dict = {}                 # override in each subclass

    def __init__(self) -> None:
        for key, spec in self._schema.items():
            object.__setattr__(self, key, spec["default"])

    def __repr__(self) -> str:
        params = ", ".join(
            "%s=%r" % (k, getattr(self, k)) for k in self._schema
        )
        return "%s(%s)" % (self.__class__.__name__, params)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __setattr__(self, key: str, value: object) -> None:
        if key.startswith("_"):
            object.__setattr__(self, key, value)
            return
        if key not in self._schema:
            raise AttributeError(
                "%s: unknown config parameter '%s'"
                % (self.__class__.__name__, key)
            )
        value = self._validate(key, value)
        object.__setattr__(self, key, value)

    def _validate(self, key: str, value: object) -> object:
        """Validate *value* against the schema spec for *key*.

        Raises:
            TypeError: Wrong type (also rejects bool when int/float expected,
                since bool is a subclass of int in Python).
            ValueError: Out of range or not in allowed choices.
        """
        spec = self._schema[key]
        expected = spec["type"]
        # bool is a subclass of int — reject it when int/float is expected
        if isinstance(value, bool) and expected is not bool:
            raise TypeError(
                "config '%s': expected %s, got bool" % (key, expected.__name__)
            )
        if not isinstance(value, expected):
            raise TypeError(
                "config '%s': expected %s, got %s"
                % (key, expected.__name__, type(value).__name__)
            )
        if "min" in spec and value < spec["min"]:  # type: ignore[operator]
            raise ValueError(
                "config '%s': %s < minimum %s" % (key, value, spec["min"])
            )
        if "max" in spec and value > spec["max"]:  # type: ignore[operator]
            raise ValueError(
                "config '%s': %s > maximum %s" % (key, value, spec["max"])
            )
        if "choices" in spec and value not in spec["choices"]:
            raise ValueError(
                "config '%s': %r not in %s" % (key, value, spec["choices"])
            )
        return value

    def to_dict(self) -> dict:
        """Serialise the config to a plain dict with a ``[pyneuromatic]`` header."""
        result: dict = {
            "pyneuromatic": {
                "type": self._TOML_TYPE,
                "version": self._TOML_VERSION,
            }
        }
        for key in self._schema:
            result[key] = getattr(self, key)
        return result

    @classmethod
    def from_dict(cls, d: dict) -> "NMToolConfig":
        """Reconstruct a config from a dict produced by ``to_dict()``.

        Unknown keys are silently ignored.  Each recognised key is validated
        via ``__setattr__`` before being stored.

        Args:
            d: Dict — typically loaded from a TOML file.

        Returns:
            New instance of *cls* with values from *d*.
        """
        obj = cls()
        for key in cls._schema:
            if key in d:
                setattr(obj, key, d[key])
        return obj

    def save(self, filepath: str | Path) -> Path:
        """Save the config to a TOML file.

        Args:
            filepath: Destination path (e.g. ``"stats.toml"``).

        Returns:
            Resolved Path of the saved file.

        Raises:
            RuntimeError: If ``tomli_w`` is not installed.
        """
        if tomli_w is None:
            raise RuntimeError(
                "TOML writing not available. Install 'tomli-w' package."
            )
        filepath = Path(filepath)
        with open(filepath, "wb") as f:
            tomli_w.dump(self.to_dict(), f)
        return filepath

    @classmethod
    def load(cls, filepath: str | Path) -> "NMToolConfig":
        """Load a config from a TOML file saved by ``save()``.

        Validates the ``[pyneuromatic]`` header before constructing the object.

        Args:
            filepath: Path to the ``.toml`` file.

        Returns:
            New instance of *cls* populated from the file.

        Raises:
            RuntimeError: If ``tomllib``/``tomli`` is not available.
            FileNotFoundError: If the file does not exist.
            ValueError: If the file's ``type`` does not match ``_TOML_TYPE``.
        """
        if tomllib is None:
            raise RuntimeError(
                "TOML support not available. "
                "Install 'tomli' package for Python < 3.11"
            )
        filepath = Path(filepath)
        with open(filepath, "rb") as f:
            d = tomllib.load(f)
        meta = d.get("pyneuromatic", {})
        if meta.get("type") != cls._TOML_TYPE:
            raise ValueError(
                "%s is not a %s config file" % (filepath.name, cls._TOML_TYPE)
            )
        return cls.from_dict(d)


