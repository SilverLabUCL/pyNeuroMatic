# -*- coding: utf-8 -*-
"""
NMStatWin and NMStatWinContainer: stat window definition and container.

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
import math
import sys
from pathlib import Path
from typing import Any

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

from pyneuromatic.tools.nm_stat_func import (
    NMStatFunc,
    FUNC_NAMES_BSLN,
    _stat_func_from_dict,
)
from pyneuromatic.core.nm_command_history import add_nm_command, records_command
from pyneuromatic.tools.nm_stat_utilities import stat
from pyneuromatic.core.nm_data import NMData
import pyneuromatic.core.nm_history as nmh
import pyneuromatic.core.nm_configurations as nmc
import pyneuromatic.core.nm_utilities as nmu


class NMStatWin:
    """Stat measurement window: x-range, function and baseline.

    Lightweight class (does not inherit NMObject) following the NMScaleY
    pattern. Each window defines one stat measurement pipeline:

    - ``func`` — the stat function (e.g., ``{"name": "mean"}`` or
      ``{"name": "risetime+", "p0": 10, "p1": 90}``)
    - ``xbgn`` / ``xend`` — x-boundaries of the main stat region
    - ``bsln_on`` / ``bsln_func`` / ``bsln_xbgn`` / ``bsln_xend`` — optional
      baseline stat subtracted from (or required by) the main stat

    Filters and transforms are set on the channel (``NMChannel.transform``),
    not the window: Stats measures the channel as the plot shows it.

    Call ``compute(data)`` to run the pipeline and retrieve result dicts.

    Args:
        name: Window name (alphanumeric + underscores, default ``"NMStatWin0"``).
        win: Optional dict to initialise window parameters via ``_win_set()``.
            Accepted keys: ``on``, ``func``, ``xbgn``, ``xend``,
            ``bsln_on``, ``bsln_func``, ``bsln_xbgn``, ``bsln_xend``.
    """

    def __init__(
        self,
        name: str = "NMStatWin0",
        win: dict[str, object] | None = None,
        nm_path: str = "stats.windows",
    ) -> None:
        if not isinstance(name, str):
            raise TypeError(nmu.type_error_str(name, "name", "string"))
        if not name or not nmu.name_ok(name):
            raise ValueError("name: %s" % name)
        self._name = name
        self._nm_path = nm_path

        self.__on = True
        self.__func: NMStatFunc | None = None
        self.__xbgn = -math.inf
        self.__xend = math.inf
        self.__results: list[dict[str, Any]] = []  # [ {}, {} ...] list of dictionaries

        # baseline
        self.__bsln_on = False
        self.__bsln_func: dict[str, Any] = {}
        self.__bsln_xbgn = -math.inf
        self.__bsln_xend = math.inf

        if win is None:
            pass  # ok
        elif isinstance(win, dict):
            self._win_set(win, quiet=True)
        else:
            e = nmu.type_error_str(win, "win", "dictionary")
            raise TypeError(e)

    def __eq__(self, other: object) -> bool:
        """Return True if other is an NMStatWin with the same to_dict()."""
        if not isinstance(other, NMStatWin):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    @property
    def name(self) -> str:
        """Window name string."""
        return self._name

    def to_dict(self) -> dict:
        """Serialize this stat window to a dict."""
        return {
            "name": self._name,
            "on": self.__on,
            "func": self.__func.to_dict() if self.__func else {},
            "xbgn": self.__xbgn,
            "xend": self.__xend,
            "bsln_on": self.__bsln_on,
            "bsln_func": self.__bsln_func,
            "bsln_xbgn": self.__bsln_xbgn,
            "bsln_xend": self.__bsln_xend
        }

    def _win_set(
        self,
        win: dict[str, object],
        quiet: bool = nmc.QUIET
    ) -> None:
        """Set multiple window parameters from a dict.

        Accepted keys: ``on``, ``func``, ``xbgn``, ``xend``,
        ``bsln_on``, ``bsln_func``, ``bsln_xbgn``, ``bsln_xend``.
        The key ``"name"`` is silently ignored (name is set via constructor).
        The old key ``"transform"`` is ignored with a warning: filters and
        transforms are now set on the channel.
        """
        if not isinstance(win, dict):
            e = nmu.type_error_str(win, "win", "dictionary")
            raise TypeError(e)
        for k, v in win.items():
            if not isinstance(k, str):
                e = nmu.type_error_str(k, "key", "string")
                raise TypeError(e)
            k = k.lower()
            if k == "name":
                continue  # name is set via constructor, not _win_set
            elif k == "on":
                self._on_set(v, quiet=True)  # type: ignore[arg-type]
            elif k == "func":
                self._func_set(v, quiet=True)  # type: ignore[arg-type]
            elif k == "xbgn":
                self._x_set("xbgn", v, quiet=True)  # type: ignore[arg-type]
            elif k == "xend":
                self._x_set("xend", v, quiet=True)  # type: ignore[arg-type]
            elif k == "transform":
                if v:  # saved by an older version
                    nmh.history(
                        "window transforms are no longer supported and were "
                        "ignored; set them on the channel (NMChannel.transform)",
                        title="WARNING", path=self._name, red=True,
                    )
            elif k == "bsln_on":
                self._bsln_on_set(v, quiet=True)  # type: ignore[arg-type]
            elif k == "bsln_func":
                self._bsln_func_set(v, quiet=True)  # type: ignore[arg-type]
            elif k == "bsln_xbgn":
                self._x_set("bsln_xbgn", v, quiet=True)  # type: ignore[arg-type]
            elif k == "bsln_xend":
                self._x_set("bsln_xend", v, quiet=True)  # type: ignore[arg-type]
            else:
                raise KeyError("unknown key '%s'" % k)
        return None

    @property
    def on(self) -> bool:
        """True if this window is active and will be included in compute()."""
        return self.__on

    @on.setter
    @records_command
    def on(self, on: bool) -> None:
        self._on_set(on)
        add_nm_command("%s[%r].on = %r" % (self._nm_path, self._name, self.__on))
        return None

    def _on_set(self, on: bool, quiet: bool = nmc.QUIET) -> None:
        """Set the on flag; raises TypeError if not bool."""
        if not isinstance(on, bool):
            e = nmu.type_error_str(on, "on", "boolean")
            raise TypeError(e)
        self.__on = on
        return None

    @property
    def func(self) -> dict:
        """The stat function as a dict (e.g. ``{"name": "mean"}``), or ``{}``."""
        return self.__func.to_dict() if self.__func else {}

    @func.setter
    @records_command
    def func(self, func: dict | str) -> None:
        self._func_set(func)
        add_nm_command("%s[%r].func = %r" % (self._nm_path, self._name, self._func_arg()))
        return None

    def _func_arg(self) -> dict | str | None:
        """The func as a command sets it: its name alone (e.g. 'mean'),
        else its dict (e.g. {'name': 'risetime+', 'p0': 10.0, 'p1': 90.0})."""
        func = self.func
        if not func:
            return None
        return func["name"] if list(func) == ["name"] else func

    def _func_set(
        self,
        func: dict | str | None,
        quiet: bool = nmc.QUIET
    ) -> None:
        """Set the stat function from a dict, string name, or None.

        If ``func`` is a dict without a ``"name"`` key and a func is already
        set, the current func name is reused so parameters can be updated
        in-place (e.g. ``w.func = {"p0": 10, "p1": 90}``).
        """
        if func is None or (isinstance(func, dict) and len(func) == 0):
            self.__func = None
            return None
        if isinstance(func, dict) and "name" not in func:
            if self.__func is not None:
                # allows updating func parameters without passing name
                func["name"] = self.__func.name
            else:
                raise KeyError("missing func key 'name'")
        self.__func = _stat_func_from_dict(func)
        return None

    @property
    def xbgn(self) -> float:
        """Left x-boundary of the main stat window (default ``-inf``)."""
        return self.__xbgn

    @xbgn.setter
    @records_command
    def xbgn(self, xbgn: float) -> None:
        self._x_set("xbgn", xbgn)
        add_nm_command("%s[%r].xbgn = %r" % (self._nm_path, self._name, self.__xbgn))
        return None

    def _x_set(
        self,
        xname: str,  # e.g. "xbgn" or "bsln_xbgn"
        x: float,
        quiet: bool = nmc.QUIET
    ) -> None:
        """Set an x-boundary by name.

        Args:
            xname: One of ``"xbgn"``, ``"xend"``, ``"bsln_xbgn"``, ``"bsln_xend"``.
            x: The x value. Infinite values are coerced to ``-inf`` for xbgn
                variants and ``+inf`` for xend variants. NaN is rejected.
        """
        x = float(x)  # might raise type error
        if math.isnan(x):
            raise ValueError(xname + ": %s" % x)
        n = xname.lower()
        if n not in ("xbgn", "xend", "bsln_xbgn", "bsln_xend"):
            raise ValueError("xname: %s" % xname)
        if math.isinf(x):
            x = -math.inf if n.endswith("xbgn") else math.inf
        if n == "xbgn":
            self.__xbgn = x
        elif n == "xend":
            self.__xend = x
        elif n == "bsln_xbgn":
            self.__bsln_xbgn = x
        elif n == "bsln_xend":
            self.__bsln_xend = x
        return None

    @property
    def xend(self) -> float:
        """Right x-boundary of the main stat window (default ``+inf``)."""
        return self.__xend

    @xend.setter
    @records_command
    def xend(self, xend: float) -> None:
        self._x_set("xend", xend)
        add_nm_command("%s[%r].xend = %r" % (self._nm_path, self._name, self.__xend))
        return None

    @property
    def bsln_on(self) -> bool:
        """True if the baseline stat is enabled."""
        return self.__bsln_on

    @bsln_on.setter
    @records_command
    def bsln_on(self, on: bool) -> None:
        self._bsln_on_set(on)
        add_nm_command("%s[%r].bsln_on = %r" % (self._nm_path, self._name, self.__bsln_on))
        return None

    def _bsln_on_set(self, on: bool, quiet: bool = nmc.QUIET) -> None:
        """Set the bsln_on flag; raises TypeError if not bool."""
        if not isinstance(on, bool):
            e = nmu.type_error_str(on, "on", "boolean")
            raise TypeError(e)
        self.__bsln_on = on
        return None

    @property
    def bsln_func(self) -> dict:
        """The baseline function as a dict (e.g. ``{"name": "mean"}``), or ``{}``."""
        return self.__bsln_func

    @bsln_func.setter
    @records_command
    def bsln_func(self, func: dict | str) -> None:
        self._bsln_func_set(func)
        add_nm_command("%s[%r].bsln_func = %r"
                       % (self._nm_path, self._name, self.__bsln_func.get("name")))
        return None

    def _bsln_func_set(
        self,
        func: dict | str | None,
        quiet: bool = nmc.QUIET
    ) -> None:
        """Set the baseline function from a dict, string name, or None.

        Only names in ``FUNC_NAMES_BSLN`` are accepted: median, mean,
        mean+var, mean+std, mean+sem.
        """
        if func is None:
            self.__bsln_func.clear()
            return None
        if isinstance(func, dict):
            if len(func) == 0:
                self.__bsln_func.clear()
                return None
            if "name" not in func:
                e = "missing func key 'name'"
                raise KeyError(e)
            func_name = func["name"]
        elif isinstance(func, str):
            func_name = func
        else:
            e = nmu.type_error_str(func, "func", "dictionary, string or None")
            raise TypeError(e)

        if func_name is None:
            self.__bsln_func.clear()
            return None
        if not isinstance(func_name, str):
            e = nmu.type_error_str(func_name, "func_name", "string")
            raise TypeError(e)

        found = False
        for f in FUNC_NAMES_BSLN:
            if f.lower() == func_name.lower():
                found = True
                break
        if not found:
            raise ValueError("func_name: %s" % func_name)

        self.__bsln_func.clear()
        self.__bsln_func.update({"name": func_name.lower()})
        return None

    @property
    def bsln_xbgn(self) -> float:
        """Left x-boundary of the baseline window (default ``-inf``)."""
        return self.__bsln_xbgn

    @bsln_xbgn.setter
    @records_command
    def bsln_xbgn(self, xbgn: float) -> None:
        self._x_set("bsln_xbgn", xbgn)
        add_nm_command("%s[%r].bsln_xbgn = %r" % (self._nm_path, self._name, self.__bsln_xbgn))
        return None

    @property
    def bsln_xend(self) -> float:
        """Right x-boundary of the baseline window (default ``+inf``)."""
        return self.__bsln_xend

    @bsln_xend.setter
    @records_command
    def bsln_xend(self, xend: float) -> None:
        self._x_set("bsln_xend", xend)
        add_nm_command("%s[%r].bsln_xend = %r" % (self._nm_path, self._name, self.__bsln_xend))
        return None

    @records_command
    def reset(self) -> None:
        """Return to the default settings, as a new window has them: on, no
        function, X begin / X end unbounded, and the baseline off (no
        function, unbounded). The name and the last results are kept."""
        self.__on = True
        self.__func = None
        self.__xbgn = -math.inf
        self.__xend = math.inf
        self.__bsln_on = False
        self.__bsln_func = {}
        self.__bsln_xbgn = -math.inf
        self.__bsln_xend = math.inf
        add_nm_command("%s[%r].reset()" % (self._nm_path, self._name))
        return None

    @property
    def results(self) -> list[dict]:
        """List of result dicts from the most recent ``compute()`` call."""
        return self.__results

    def _run_stat(self, data, func, id_str, xbgn, xend, ignore_nans, **extra):
        """Create a result dict, append it to results, and call stat().

        Args:
            data: NMData object to analyse.
            func: Stat function dict (e.g. ``{"name": "max"}``).
            id_str: Label stored in the result as ``"id"`` (e.g. ``"bsln"``,
                ``"main"``, or a func name for intermediate pipeline steps).
            xbgn: Left x-boundary for this stat call.
            xend: Right x-boundary for this stat call.
            ignore_nans: If True, exclude NaN values.
            **extra: Additional key-value pairs stored in the result dict
                (e.g. ``p0``, ``p1``, ``warning``).

        Returns:
            The result dict after stat() has populated it in-place.
        """
        r: dict[str, Any] = {"win": self.name, "id": id_str}
        r.update(extra)
        r["func"] = func
        r["xbgn"] = xbgn
        r["xend"] = xend
        self.__results.append(r)
        stat(data, func, xbgn=xbgn, xend=xend, ignore_nans=ignore_nans, results=r)
        return r

    def compute(
        self,
        data: NMData,
        ignore_nans: bool = False,
        quiet: bool = nmc.QUIET,
        note: bool = True,
    ) -> list:
        """Run the stat computation on data.

        Runs the baseline stat if ``bsln_on`` is True, then delegates to
        ``self.__func.compute()``. *data* is measured as it is (the tools
        pass the channel's transformed copy, ``NMChannel.transformed``).

        Args:
            data: NMData object to analyse.
            ignore_nans: If True, exclude NaN values from computations.
            quiet: If True, the note isn't printed (it is still logged).
            note: If False, no note is written to the history at all (e.g.
                the Plot's preview of the measurement, #463).

        Returns:
            List of result dicts from each stat call in the pipeline.
            Returns an empty list if ``data`` is None or no func is set.
        """
        self.__results = []

        if data is None or self.__func is None:
            return self.__results

        if not isinstance(ignore_nans, bool):
            ignore_nans = True

        bsln_result: dict[str, Any] = {}

        if self.__bsln_on:
            self.__func.validate_baseline(self.__bsln_func.get("name"))
            bsln_result = self._run_stat(
                data, self.__bsln_func.copy(), "bsln",
                self.__bsln_xbgn, self.__bsln_xend, ignore_nans
            )
        elif self.__func.needs_baseline:
            raise RuntimeError(
                "func '%s' requires baseline" % self.__func.name
            )

        self.__func.compute(
            data, self.__xbgn, self.__xend, ignore_nans,
            self._run_stat, bsln_result
        )

        if note:
            text = "compute func=%s, xbgn=%s, xend=%s" % (
                self.__func.name, self.__xbgn, self.__xend)
            if self.__bsln_on:
                text += ", bsln=%s %s to %s" % (
                    self.__bsln_func.get("name"), self.__bsln_xbgn, self.__bsln_xend)
            nmh.history(text, path=self._name, quiet=quiet)

        return self.__results


class NMStatWinContainer:
    """Ordered container of NMStatWin objects with auto-naming.

    Windows are created via ``new()`` and named ``{prefix}0``, ``{prefix}1``,
    etc. ``selected_name`` is the window currently shown for editing.

    Args:
        name_prefix: Prefix for auto-generated window names (default ``"w"``).
    """

    def __init__(
        self,
        name_prefix: str = "w",
        nm_path: str = "stats.windows",
    ) -> None:
        self._prefix = name_prefix
        self._nm_path = nm_path
        self._windows: dict[str, NMStatWin] = {}
        self._count = 0
        self._selected_name: str | None = None

    @property
    def selected_name(self) -> str | None:
        """Name of the window shown for editing (None if there are none)."""
        return self._selected_name

    @selected_name.setter
    @records_command
    def selected_name(self, name: str | None) -> None:
        if name is not None and name not in self._windows:
            raise KeyError("no stat window '%s'" % name)
        if name == self._selected_name:
            return  # unchanged: nothing to record
        self._selected_name = name
        add_nm_command("%s.selected_name = %r" % (self._nm_path, name))

    @records_command
    def new(self, select: bool = False, quiet: bool = nmc.QUIET) -> NMStatWin:
        """Create, register, and return a new NMStatWin with an auto-name.

        Args:
            select: If True, select it (the first window is always selected).
        """
        name = "%s%d" % (self._prefix, self._count)
        self._count += 1
        w = NMStatWin(name=name, nm_path=self._nm_path)
        self._windows[name] = w
        if select or self._selected_name is None:
            self._selected_name = name
        # The command doesn't name the window (it is named for you)
        nmh.history("new window %s" % name, path=self._nm_path, quiet=quiet)
        add_nm_command("%s.new(%s)" % (self._nm_path, "select=True" if select else ""))
        return w

    @records_command
    def delete_last(self, quiet: bool = nmc.QUIET) -> str:
        """Delete the last window (e.g. w2), so the names stay in sequence;
        the next ``new()`` window takes its name again. The first window is
        kept: there is always one. If the last window was selected, the one
        before it is selected.

        Returns:
            The name of the window deleted.

        Raises:
            RuntimeError: If there is only one window (or none).
        """
        if len(self._windows) <= 1:
            raise RuntimeError("the first stat window can't be deleted")
        name = list(self._windows)[-1]
        del self._windows[name]
        number = name[len(self._prefix):]
        if name.startswith(self._prefix) and number.isdigit():
            self._count = int(number)  # new() makes it again
        if self._selected_name == name:
            self._selected_name = list(self._windows)[-1]
        nmh.history("deleted window %s" % name, path=self._nm_path, quiet=quiet)
        add_nm_command("%s.delete_last()" % self._nm_path)
        return name

    def __iter__(self):
        """Iterate over all NMStatWin objects in insertion order."""
        return iter(self._windows.values())

    def __len__(self):
        """Return the number of windows in the container."""
        return len(self._windows)

    def __getitem__(self, name: str) -> NMStatWin:
        """Return the NMStatWin with the given name."""
        return self._windows[name]

    def __contains__(self, name: str) -> bool:
        """Return True if a window with the given name exists."""
        return name in self._windows

    _TOML_TYPE = "stat_windows"
    _TOML_VERSION = "1"

    def to_dict(self) -> dict:
        """Serialize the container and all windows to a plain dict.

        The returned dict includes a ``[pyneuromatic]`` metadata header so
        that saved TOML files are self-describing.  None values are omitted
        from window dicts (TOML does not support null); ``selected_name`` is
        omitted when None for the same reason.
        """
        def _strip_none(d: dict) -> dict:
            return {k: v for k, v in d.items() if v is not None}

        result: dict = {
            "pyneuromatic": {
                "type": self._TOML_TYPE,
                "version": self._TOML_VERSION,
            },
            "prefix": self._prefix,
            "windows": [_strip_none(w.to_dict()) for w in self._windows.values()],
        }
        if self.selected_name is not None:
            result["selected_name"] = self.selected_name
        return result

    @classmethod
    def from_dict(cls, d: dict) -> "NMStatWinContainer":
        """Reconstruct a container from a dict produced by ``to_dict()``.

        Args:
            d: Dict with keys ``"prefix"``, ``"selected_name"``, and
                ``"windows"`` (list of window dicts).

        Returns:
            New NMStatWinContainer populated with NMStatWin objects.
        """
        prefix = d.get("prefix", "w")
        container = cls(name_prefix=prefix)
        for wd in d.get("windows", []):
            name = wd.get("name", "%s%d" % (prefix, container._count))
            w = NMStatWin(name=name, win=wd)
            container._windows[name] = w
            container._count += 1
        container._selected_name = d.get("selected_name")
        return container

    def save(self, filepath: str | Path) -> Path:
        """Save the container to a TOML file.

        Args:
            filepath: Destination path (e.g. ``"my_windows.toml"``).

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
    def load(cls, filepath: str | Path) -> "NMStatWinContainer":
        """Load a container from a TOML file saved by ``save()``.

        Args:
            filepath: Path to the ``.toml`` file.

        Returns:
            New NMStatWinContainer populated from the file.

        Raises:
            RuntimeError: If ``tomllib``/``tomli`` is not available.
            FileNotFoundError: If the file does not exist.
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
                "%s is not a pyNeuroMatic stat windows file" % filepath.name
            )
        return cls.from_dict(d)
