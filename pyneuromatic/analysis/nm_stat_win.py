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
import copy
import math
from typing import Any

from pyneuromatic.analysis.nm_stat_func import (
    NMStatFunc,
    FUNC_NAMES_BSLN,
    _stat_func_from_dict,
)
from pyneuromatic.analysis.nm_stat_utilities import stat
from pyneuromatic.core.nm_data import NMData
import pyneuromatic.core.nm_history as nmh
import pyneuromatic.core.nm_preferences as nmp
from pyneuromatic.core.nm_transform import (
    NMTransform,
    apply_transforms,
    _transform_from_dict,
)
import pyneuromatic.core.nm_utilities as nmu


class NMStatWin:
    """Stat measurement window: x-range, function, baseline, and transforms.

    Lightweight class (does not inherit NMObject) following the NMScaleY
    pattern. Each window defines one stat measurement pipeline:

    - ``func`` — the stat function (e.g., ``{"name": "mean"}`` or
      ``{"name": "risetime+", "p0": 10, "p1": 90}``)
    - ``x0`` / ``x1`` — x-boundaries of the main stat region
    - ``bsln_on`` / ``bsln_func`` / ``bsln_x0`` / ``bsln_x1`` — optional
      baseline stat subtracted from (or required by) the main stat
    - ``transform`` — optional list of NMTransform operations applied to a
      copy of the data before computing

    Call ``compute(data)`` to run the pipeline and retrieve result dicts.

    Args:
        parent: Optional parent object reference.
        name: Window name (alphanumeric + underscores, default ``"NMStatWin0"``).
        win: Optional dict to initialise window parameters via ``_win_set()``.
            Accepted keys: ``on``, ``func``, ``x0``, ``x1``, ``transform``,
            ``bsln_on``, ``bsln_func``, ``bsln_x0``, ``bsln_x1``.
    """

    def __init__(
        self,
        parent: object | None = None,
        name: str = "NMStatWin0",
        win: dict[str, object] | None = None,
    ) -> None:
        self._parent = parent
        if not isinstance(name, str):
            raise TypeError(nmu.type_error_str(name, "name", "string"))
        if not name or not nmu.name_ok(name):
            raise ValueError("name: %s" % name)
        self._name = name

        self.__on = True
        self.__func: NMStatFunc | None = None
        self.__x0 = -math.inf
        self.__x1 = math.inf
        self.__transform: list[NMTransform] | None = None
        self.__results: list[dict[str, Any]] = []  # [ {}, {} ...] list of dictionaries

        # baseline
        self.__bsln_on = False
        self.__bsln_func: dict[str, Any] = {}
        self.__bsln_x0 = -math.inf
        self.__bsln_x1 = math.inf

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

    def __deepcopy__(self, memo: dict) -> NMStatWin:
        """Deep-copy all attributes, resetting _parent to None."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for attr, value in self.__dict__.items():
            if attr == "_parent":
                setattr(result, attr, None)
            else:
                setattr(result, attr, copy.deepcopy(value, memo))
        return result

    @property
    def name(self) -> str:
        """Window name string."""
        return self._name

    def copy(self) -> NMStatWin:
        """Return a deep copy of this window."""
        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        """Serialize this stat window to a dict."""
        if self.__transform is not None:
            transform_dicts = [t.to_dict() for t in self.__transform]
        else:
            transform_dicts = None
        return {
            "name": self._name,
            "on": self.__on,
            "func": self.__func.to_dict() if self.__func else {},
            "x0": self.__x0,
            "x1": self.__x1,
            "transform": transform_dicts,
            "bsln_on": self.__bsln_on,
            "bsln_func": self.__bsln_func,
            "bsln_x0": self.__bsln_x0,
            "bsln_x1": self.__bsln_x1
        }

    def _win_set(
        self,
        win: dict[str, object],
        quiet: bool = nmp.QUIET
    ) -> None:
        """Set multiple window parameters from a dict.

        Accepted keys: ``on``, ``func``, ``x0``, ``x1``, ``transform``,
        ``bsln_on``, ``bsln_func``, ``bsln_x0``, ``bsln_x1``.
        The key ``"name"`` is silently ignored (name is set via constructor).
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
            elif k == "x0":
                self._x_set("x0", v, quiet=True)  # type: ignore[arg-type]
            elif k == "x1":
                self._x_set("x1", v, quiet=True)  # type: ignore[arg-type]
            elif k == "transform":
                self._transform_set(v, quiet=True)  # type: ignore[arg-type]
            elif k == "bsln_on":
                self._bsln_on_set(v, quiet=True)  # type: ignore[arg-type]
            elif k == "bsln_func":
                self._bsln_func_set(v, quiet=True)  # type: ignore[arg-type]
            elif k == "bsln_x0":
                self._x_set("bsln_x0", v, quiet=True)  # type: ignore[arg-type]
            elif k == "bsln_x1":
                self._x_set("bsln_x1", v, quiet=True)  # type: ignore[arg-type]
            else:
                raise KeyError("unknown key '%s'" % k)
        nmh.history("set win=%s" % win, path=self._name, quiet=quiet)
        return None

    @property
    def on(self) -> bool:
        """True if this window is active and will be included in compute()."""
        return self.__on

    @on.setter
    def on(self, on: bool) -> None:
        return self._on_set(on)

    def _on_set(self, on: bool, quiet: bool = nmp.QUIET) -> None:
        """Set the on flag; raises TypeError if not bool."""
        if not isinstance(on, bool):
            e = nmu.type_error_str(on, "on", "boolean")
            raise TypeError(e)
        self.__on = on
        nmh.history("set on=%s" % on, path=self._name, quiet=quiet)
        return None

    @property
    def func(self) -> dict:
        """The stat function as a dict (e.g. ``{"name": "mean"}``), or ``{}``."""
        return self.__func.to_dict() if self.__func else {}

    @func.setter
    def func(self, func: dict | str) -> None:
        self._func_set(func)
        return None

    def _func_set(
        self,
        func: dict | str | None,
        quiet: bool = nmp.QUIET
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
        self.__func = _stat_func_from_dict(func, parent=self._parent)
        if self.__func is not None:
            nmh.history(
                "set func=%s" % self.__func.to_dict(),
                path=self._name,
                quiet=quiet,
            )
        return None

    @property
    def x0(self) -> float:
        """Left x-boundary of the main stat window (default ``-inf``)."""
        return self.__x0

    @x0.setter
    def x0(self, x0: float) -> None:
        return self._x_set("x0", x0)

    def _x_set(
        self,
        xname: str,  # e.g. "x0" or "bsln_x0"
        x: float,
        quiet: bool = nmp.QUIET
    ) -> None:
        """Set an x-boundary by name.

        Args:
            xname: One of ``"x0"``, ``"x1"``, ``"bsln_x0"``, ``"bsln_x1"``.
            x: The x value. Infinite values are coerced to ``-inf`` for x0
                variants and ``+inf`` for x1 variants. NaN is rejected.
        """
        x = float(x)  # might raise type error
        if math.isnan(x):
            raise ValueError(xname + ": %s" % x)
        n = xname.lower()
        if n not in ("x0", "x1", "bsln_x0", "bsln_x1"):
            raise ValueError("xname: %s" % xname)
        if math.isinf(x):
            x = -math.inf if n.endswith("x0") else math.inf
        if n == "x0":
            self.__x0 = x
        elif n == "x1":
            self.__x1 = x
        elif n == "bsln_x0":
            self.__bsln_x0 = x
        elif n == "bsln_x1":
            self.__bsln_x1 = x
        nmh.history("set %s=%s" % (n, x), path=self._name, quiet=quiet)
        return None

    @property
    def x1(self) -> float:
        """Right x-boundary of the main stat window (default ``+inf``)."""
        return self.__x1

    @x1.setter
    def x1(self, x1: float) -> None:
        return self._x_set("x1", x1)

    @property
    def transform(self) -> list[NMTransform] | None:
        """List of NMTransform objects applied before computing, or None."""
        return self.__transform

    @transform.setter
    def transform(self, transform_list: list) -> None:
        return self._transform_set(transform_list)

    def _transform_set(
        self,
        transform_list: list[NMTransform] | list[dict] | None,
        quiet: bool = nmp.QUIET
    ) -> None:
        """Set the transform list from NMTransform objects, dicts, or None."""
        if transform_list is None:
            self.__transform = None
            nmh.history("set transform=None", path=self._name, quiet=quiet)
            return None
        if not isinstance(transform_list, list):
            e = nmu.type_error_str(transform_list, "transform_list", "list")
            raise TypeError(e)
        result = []
        for item in transform_list:
            if isinstance(item, NMTransform):
                result.append(item)
            elif isinstance(item, dict):
                result.append(_transform_from_dict(item))
            else:
                e = nmu.type_error_str(
                    item, "transform item", "NMTransform or dict"
                )
                raise TypeError(e)
        self.__transform = result
        nmh.history(
            "set transform=[%s]" % ", ".join(t.type_str for t in result),
            path=self._name,
            quiet=quiet,
        )
        return None

    @property
    def bsln_on(self) -> bool:
        """True if the baseline stat is enabled."""
        return self.__bsln_on

    @bsln_on.setter
    def bsln_on(self, on: bool) -> None:
        return self._bsln_on_set(on)

    def _bsln_on_set(self, on: bool, quiet: bool = nmp.QUIET) -> None:
        """Set the bsln_on flag; raises TypeError if not bool."""
        if not isinstance(on, bool):
            e = nmu.type_error_str(on, "on", "boolean")
            raise TypeError(e)
        self.__bsln_on = on
        nmh.history("set bsln_on=%s" % on, path=self._name, quiet=quiet)
        return None

    @property
    def bsln_func(self) -> dict:
        """The baseline function as a dict (e.g. ``{"name": "mean"}``), or ``{}``."""
        return self.__bsln_func

    @bsln_func.setter
    def bsln_func(self, func: dict | str) -> None:
        self._bsln_func_set(func)
        return None

    def _bsln_func_set(
        self,
        func: dict | str | None,
        quiet: bool = nmp.QUIET
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
        nmh.history(
            "set bsln_func=%s" % self.__bsln_func,
            path=self._name,
            quiet=quiet,
        )
        return None

    @property
    def bsln_x0(self) -> float:
        """Left x-boundary of the baseline window (default ``-inf``)."""
        return self.__bsln_x0

    @bsln_x0.setter
    def bsln_x0(self, x0: float) -> None:
        return self._x_set("bsln_x0", x0)

    @property
    def bsln_x1(self) -> float:
        """Right x-boundary of the baseline window (default ``+inf``)."""
        return self.__bsln_x1

    @bsln_x1.setter
    def bsln_x1(self, x1: float) -> None:
        return self._x_set("bsln_x1", x1)

    @property
    def results(self) -> list[dict]:
        """List of result dicts from the most recent ``compute()`` call."""
        return self.__results

    def _run_stat(self, data, func, id_str, x0, x1, xclip, ignore_nans,
                  **extra):
        """Create a result dict, append it to results, and call stat().

        Args:
            data: NMData object to analyse.
            func: Stat function dict (e.g. ``{"name": "max"}``).
            id_str: Label stored in the result as ``"id"`` (e.g. ``"bsln"``,
                ``"main"``, or a func name for intermediate pipeline steps).
            x0: Left x-boundary for this stat call.
            x1: Right x-boundary for this stat call.
            xclip: If True, clip data to [x0, x1].
            ignore_nans: If True, exclude NaN values.
            **extra: Additional key-value pairs stored in the result dict
                (e.g. ``p0``, ``p1``, ``warning``).

        Returns:
            The result dict after stat() has populated it in-place.
        """
        r: dict[str, Any] = {"win": self.name, "id": id_str}
        r.update(extra)
        r["func"] = func
        r["x0"] = x0
        r["x1"] = x1
        self.__results.append(r)
        stat(data, func, x0=x0, x1=x1, xclip=xclip,
             ignore_nans=ignore_nans, results=r)
        return r

    def compute(
        self,
        data: NMData,
        xclip: bool = False,  # if x0|x1 OOB, clip to data x-scale limits
        ignore_nans: bool = False,
        quiet: bool = nmp.QUIET
    ) -> list:
        """Run the stat computation on data.

        Applies any transforms to a copy of the data (original is never
        mutated), runs the baseline stat if ``bsln_on`` is True, then
        delegates to ``self.__func.compute()``.

        Args:
            data: NMData object to analyse.
            xclip: If True, clip data to [x0, x1] before computing stats.
            ignore_nans: If True, exclude NaN values from computations.
            quiet: If True, suppress history logging.

        Returns:
            List of result dicts from each stat call in the pipeline.
            Returns an empty list if ``data`` is None or no func is set.
        """
        # Apply transforms to a copy of the data (original never mutated)
        if self.__transform and len(self.__transform) > 0:
            if data is not None and data.nparray is not None:
                transformed = apply_transforms(
                    data.nparray, self.__transform, xscale=data.xscale
                )
                data = NMData(
                    name=data.name,
                    nparray=transformed,
                    xscale=data.xscale.to_dict(),
                    yscale=data.yscale.to_dict(),
                )

        self.__results = []

        if data is None or self.__func is None:
            return self.__results

        if not isinstance(xclip, bool):
            xclip = False

        if not isinstance(ignore_nans, bool):
            ignore_nans = True

        bsln_result: dict[str, Any] = {}

        if self.__bsln_on:
            self.__func.validate_baseline(self.__bsln_func.get("name"))
            bsln_result = self._run_stat(
                data, self.__bsln_func.copy(), "bsln",
                self.__bsln_x0, self.__bsln_x1, xclip, ignore_nans
            )
        elif self.__func.needs_baseline:
            raise RuntimeError(
                "func '%s' requires baseline" % self.__func.name
            )

        self.__func.compute(
            data, self.__x0, self.__x1, xclip, ignore_nans,
            self._run_stat, bsln_result
        )

        nmh.history(
            "compute func=%s, x0=%s, x1=%s, n=%d"
            % (self.__func.name, self.__x0, self.__x1, len(self.__results)),
            path=self._name,
            quiet=quiet,
        )

        return self.__results


class NMStatWinContainer:
    """Ordered container of NMStatWin objects with auto-naming.

    Windows are created via ``new()`` and named ``{prefix}0``, ``{prefix}1``,
    etc. The ``selected_name`` attribute tracks which window is currently
    active.

    Args:
        parent: Optional parent object reference.
        name_prefix: Prefix for auto-generated window names (default ``"w"``).
    """

    def __init__(
        self,
        parent: object | None = None,
        name_prefix: str = "w",
    ) -> None:
        self._parent = parent
        self._prefix = name_prefix
        self._windows: dict[str, NMStatWin] = {}
        self._count = 0
        self.selected_name: str | None = None

    def new(self, quiet: bool = nmp.QUIET) -> NMStatWin:
        """Create, register, and return a new NMStatWin with an auto-name."""
        name = "%s%d" % (self._prefix, self._count)
        self._count += 1
        w = NMStatWin(parent=self._parent, name=name)
        self._windows[name] = w
        if self.selected_name is None:
            self.selected_name = name
        nmh.history("new NMStatWin=%s" % name, quiet=quiet)
        return w

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
