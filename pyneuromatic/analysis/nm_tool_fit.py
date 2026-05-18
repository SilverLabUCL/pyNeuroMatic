# -*- coding: utf-8 -*-
"""
NMToolFit: curve-fitting analysis tool.

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

import numpy as np

from pyneuromatic.analysis.nm_tool import NMTool
from pyneuromatic.analysis.nm_tool_config import NMToolConfig
from pyneuromatic.analysis.nm_tool_folder import NMToolFolder
from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_folder import NMFolder
import pyneuromatic.core.nm_command_history as nmch
import pyneuromatic.core.nm_configurations as nmc
import pyneuromatic.core.nm_history as nmh
import pyneuromatic.core.nm_utilities as nmu
from pyneuromatic.analysis.nm_tool_utilities import fit_nmdata

_POLY_FUNC_NAMES: list[str] = ["poly%d" % d for d in range(2, 10)]
_VALID_FUNC_NAMES: frozenset[str] = frozenset(
    {"line", "exp", "exp2", "gauss", "boltzmann"} | set(_POLY_FUNC_NAMES)
)


def _poly_degree(func_name: str) -> int:
    """Extract polynomial degree from a ``'polyN'`` func_name string."""
    return int(func_name[4:])


def _eval_model(func_name: str, result: dict, x: np.ndarray) -> np.ndarray:
    """Evaluate the fitted model at *x* using parameters from *result*.

    Returns an array of NaNs if *func_name* is not recognised.
    Comparison is case-insensitive.
    """
    fn = func_name.lower()
    if fn == "line":
        return result["slope"] * x + result["intercept"]
    if fn.startswith("poly"):
        # polyval expects descending order; result["coefficients"] is ascending
        coeffs_desc = list(reversed(result["coefficients"]))
        return np.polyval(coeffs_desc, x)
    if fn == "exp":
        A, B, X0, Y0 = result["A"], result["Tau"], result["X0"], result["Y0"]
        return A * np.exp(-(x - X0) / B) + Y0
    if fn == "exp2":
        A1, B1, A2, B2 = result["A1"], result["Tau1"], result["A2"], result["Tau2"]
        X0, Y0 = result["X0"], result["Y0"]
        return A1 * np.exp(-(x - X0) / B1) + A2 * np.exp(-(x - X0) / B2) + Y0
    if fn == "gauss":
        A, mu, sg, Y0 = result["A"], result["Mu"], result["Sigma"], result["Y0"]
        return A * np.exp(-0.5 * ((x - mu) / sg) ** 2) + Y0
    if fn == "boltzmann":
        A, V50, K, Y0 = result["A"], result["V50"], result["K"], result["Y0"]
        return A / (1.0 + np.exp(-(x - V50) / K)) + Y0
    return np.full_like(x, float("nan"), dtype=float)


class NMToolFitConfig(NMToolConfig):
    """Configuration for NMToolFit.

    Parameters:
        func_name: Fitting function — ``"line"`` (default),
            ``"poly2"``–``"poly9"`` (polynomial of that degree),
            ``"exp"``, or ``"gauss"``.
        xbgn: Fit window start (x-units). Default ``-inf``.
        xend: Fit window end (x-units). Default ``+inf``.
        maxfev: Maximum function evaluations for nonlinear fits. Default 10000.
        ignore_nans: If True (default), exclude NaN values before fitting.
        overwrite: Reuse existing toolfolder instead of creating a new one.
            Default True.
        results_to_history: Print fit results to history log. Default False.
        results_to_cache: Save results dict to folder.toolresults. Default True.
        results_to_numpy: Write FT_ NMData arrays to a toolfolder. Default True.
    """

    _TOML_TYPE = "fit_config"
    _schema = {
        "func_name":          {"type": str,   "default": "line",
                               "choices": ["line"] + _POLY_FUNC_NAMES + ["exp", "exp2", "gauss", "boltzmann"]},
        "xbgn":                 {"type": float, "default": -math.inf},
        "xend":                 {"type": float, "default":  math.inf},
        "maxfev":             {"type": int,   "default": 10000, "min": 1},
        "ignore_nans":        {"type": bool,  "default": True},
        "overwrite":          {"type": bool,  "default": True},
        "results_to_history":  {"type": bool,  "default": False},
        "results_to_cache":    {"type": bool,  "default": True},
        "results_to_numpy":    {"type": bool,  "default": True},
        "results_errors":      {"type": bool,  "default": False},
        "results_residuals":   {"type": bool,  "default": False},
        "x_origin":            {"type": float, "default": 0.0},
        "results_fit_array":   {"type": bool,  "default": False},
        "results_fit_npts":    {"type": int,   "default": 0, "min": 0},
    }


class NMToolFit(NMTool):
    """Curve-fitting tool for NMData arrays.

    Fits a mathematical function to each epoch's NMData array over the
    specified x-range and writes per-epoch result arrays (``FT_`` prefix)
    to a toolfolder.

    Supported functions (``func_name``):

    * ``"line"``              — ``y = M * x + B``
      (M = slope, B = intercept)
    * ``"poly2"``–``"poly9"`` — ``y = C0 + C1*x + … + Cn*x^n``
      (Ck = coefficient of x^k)
    * ``"exp"``               — ``y = A * exp(-(x - X0) / Tau) + Y0``
      (A = amplitude, Tau = decay time constant, X0 = fixed x-origin, Y0 = y-offset)
    * ``"exp2"``              — ``y = A1 * exp(-(x - X0) / Tau1) + A2 * exp(-(x - X0) / Tau2) + Y0``
      (Tau1 ≤ Tau2 after fitting; X0 = fixed x-origin)
    * ``"gauss"``             — ``y = A * exp(-0.5 * ((x - mu) / sigma)^2) + Y0``
      (A = amplitude, mu = mean, sigma = std dev, Y0 = y-offset)
    * ``"boltzmann"``         — ``y = A / (1 + exp(-(x - V50) / K)) + Y0``
      (A = amplitude, V50 = midpoint, K = slope factor, Y0 = y-offset)

    Initial parameters for nonlinear fits (``exp``, ``gauss``) are
    auto-estimated from the data; supply *p0* to override.  ``X0`` in the
    exp model is a fixed constant set via ``x_origin`` (not fitted).
    Per-point standard deviations for weighted fitting can be supplied via
    *sigma*.

    Attributes:
        func_name: Active fitting function. Default ``"line"``.
        xbgn: Fit window start. Default ``-inf``.
        xend: Fit window end. Default ``+inf``.
        maxfev: Maximum function evaluations for nonlinear fits. Default 10000.
        p0: Initial parameter dict for ``exp``/``gauss`` fits, or ``None``
            for auto-estimation. Not TOML-serializable.
        sigma: Per-point standard deviations array for weighted fitting,
            or ``None``. Not TOML-serializable.
        results_errors: When True, writes ``FT_err_*`` arrays of one-sigma
            parameter uncertainties to the toolfolder. Default False.
        results_residuals: When True, writes per-epoch ``FT_resid_*`` arrays
            (data − model) to the toolfolder. Default False.
        results_fit_array: When True, writes per-epoch ``Fit_*`` arrays
            containing the model evaluated at the fit x-values. Default False.
        results_fit_npts: Number of points for the fit array. 0 (default)
            uses the same x-points as the original data. Any positive integer
            generates a uniformly-spaced grid from x[0] to x[-1].
        x_origin: Fixed x-offset (X0) for the ``exp`` model: ``A*exp(-(x-X0)/Tau)+Y0``.
            Default 0.0. Has no effect on other fit functions.
        param_names: Optional dict remapping default parameter names to
            user-defined names for output arrays.  Keys are default names
            (e.g. ``"A"``, ``"B"``, ``"Tau"``, ``"Y0"``, ``"Mu"``, ``"Sigma"``);
            values are replacement names.  Partial overrides are supported.
            Not TOML-serializable.
    """

    def __init__(self) -> None:
        super().__init__(name="fit")
        self._config = NMToolFitConfig()

        self._ignore_nans = self._config.ignore_nans
        self._overwrite = self._config.overwrite
        self._results_to_history = self._config.results_to_history
        self._results_to_cache = self._config.results_to_cache
        self._results_to_numpy = self._config.results_to_numpy

        self._xbgn = self._config.xbgn
        self._xend = self._config.xend

        self.__func_name: str = "line"
        self.__maxfev: int = self._config.maxfev
        self.__x_origin: float = self._config.x_origin

        self.__results_errors: bool = self._config.results_errors
        self.__results_residuals: bool = self._config.results_residuals
        self.__results_fit_array: bool = self._config.results_fit_array
        self.__results_fit_npts: int = self._config.results_fit_npts

        # Function-specific / not TOML-serializable — not in config
        self.__p0: dict | None = None
        self.__sigma: np.ndarray | None = None
        self.__param_names: dict | None = None

        # Internal run state — reset by run_init()
        self._fit_results: list[dict] = []
        self._epoch_names: list[str] = []
        self._toolfolder: NMToolFolder | None = None

    # ------------------------------------------------------------------
    # Properties

    @property
    def func_name(self) -> str:
        """Fitting function: ``'line'``, ``'poly2'``–``'poly9'``, ``'exp'``, or ``'gauss'``."""
        return self.__func_name

    @func_name.setter
    def func_name(self, value: str) -> None:
        self._func_name_set(value)

    def _func_name_set(self, value: str, quiet: bool = nmc.QUIET) -> None:
        if not isinstance(value, str):
            raise TypeError(nmu.type_error_str(value, "func_name", "string"))
        if value not in _VALID_FUNC_NAMES:
            raise ValueError(
                "func_name must be one of %s, got %r"
                % (sorted(_VALID_FUNC_NAMES), value)
            )
        self.__func_name = value
        nmh.history("set func_name=%r" % self.__func_name, quiet=quiet)
        nmch.add_nm_command("%s.func_name = %r" % (self._name, self.__func_name))

    @property
    def maxfev(self) -> int:
        """Maximum function evaluations for nonlinear fits (exp/gauss). Default 10000."""
        return self.__maxfev

    @maxfev.setter
    def maxfev(self, value: int) -> None:
        self._maxfev_set(value)

    def _maxfev_set(self, value: int, quiet: bool = nmc.QUIET) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(nmu.type_error_str(value, "maxfev", "int"))
        if value < 1:
            raise ValueError("maxfev must be >= 1, got %d" % value)
        self.__maxfev = value
        nmh.history("set maxfev=%d" % self.__maxfev, quiet=quiet)
        nmch.add_nm_command("%s.maxfev = %r" % (self._name, self.__maxfev))

    @property
    def p0(self) -> dict | None:
        """Initial parameter estimates for nonlinear fits (exp/gauss), or None."""
        return self.__p0

    @p0.setter
    def p0(self, value: dict | None) -> None:
        if value is not None and not isinstance(value, dict):
            raise TypeError(nmu.type_error_str(value, "p0", "dict or None"))
        self.__p0 = value
        nmh.history("set p0=%r" % self.__p0, quiet=nmc.QUIET)
        nmch.add_nm_command("%s.p0 = %r" % (self._name, self.__p0))

    @property
    def sigma(self) -> np.ndarray | None:
        """Per-point standard deviations for weighted fitting, or None."""
        return self.__sigma

    @sigma.setter
    def sigma(self, value: np.ndarray | None) -> None:
        if value is not None and not isinstance(value, np.ndarray):
            raise TypeError(nmu.type_error_str(value, "sigma", "numpy ndarray or None"))
        self.__sigma = value
        nmh.history("set sigma (len=%s)" % (len(value) if value is not None else None),
                    quiet=nmc.QUIET)
        nmch.add_nm_command("%s.sigma = <array>" % self._name)

    @property
    def x_origin(self) -> float:
        """Fixed x-offset (X0) for the ``exp`` model. Default 0.0."""
        return self.__x_origin

    @x_origin.setter
    def x_origin(self, value: float) -> None:
        self._x_origin_set(value)

    def _x_origin_set(self, value: float, quiet: bool = nmc.QUIET) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "x_origin", "float"))
        if math.isnan(value):
            raise ValueError("x_origin cannot be NaN")
        self.__x_origin = float(value)
        nmh.history("set x_origin=%g" % self.__x_origin, quiet=quiet)
        nmch.add_nm_command("%s.x_origin = %r" % (self._name, self.__x_origin))

    @property
    def param_names(self) -> dict | None:
        """Dict remapping default parameter names to user-defined output names, or None."""
        return self.__param_names

    @param_names.setter
    def param_names(self, value: dict | None) -> None:
        if value is not None and not isinstance(value, dict):
            raise TypeError(nmu.type_error_str(value, "param_names", "dict or None"))
        self.__param_names = value
        nmh.history("set param_names=%r" % self.__param_names, quiet=nmc.QUIET)
        nmch.add_nm_command("%s.param_names = %r" % (self._name, self.__param_names))

    @property
    def results_errors(self) -> bool:
        """Write parameter standard-deviation arrays (``FT_err_*``) to the toolfolder."""
        return self.__results_errors

    @results_errors.setter
    def results_errors(self, value: bool) -> None:
        self._results_errors_set(value)

    def _results_errors_set(self, value: bool, quiet: bool = nmc.QUIET) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "results_errors", "bool"))
        self.__results_errors = value
        nmh.history("set results_errors=%r" % self.__results_errors, quiet=quiet)
        nmch.add_nm_command("%s.results_errors = %r" % (self._name, self.__results_errors))

    @property
    def results_residuals(self) -> bool:
        """Write per-epoch residual arrays (``FT_resid_*``) to the toolfolder."""
        return self.__results_residuals

    @results_residuals.setter
    def results_residuals(self, value: bool) -> None:
        self._results_residuals_set(value)

    def _results_residuals_set(self, value: bool, quiet: bool = nmc.QUIET) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "results_residuals", "bool"))
        self.__results_residuals = value
        nmh.history("set results_residuals=%r" % self.__results_residuals, quiet=quiet)
        nmch.add_nm_command("%s.results_residuals = %r" % (self._name, self.__results_residuals))

    @property
    def results_fit_array(self) -> bool:
        """Write per-epoch ``Fit_*`` model arrays to the toolfolder."""
        return self.__results_fit_array

    @results_fit_array.setter
    def results_fit_array(self, value: bool) -> None:
        self._results_fit_array_set(value)

    def _results_fit_array_set(self, value: bool, quiet: bool = nmc.QUIET) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "results_fit_array", "bool"))
        self.__results_fit_array = value
        nmh.history("set results_fit_array=%r" % self.__results_fit_array, quiet=quiet)
        nmch.add_nm_command("%s.results_fit_array = %r" % (self._name, self.__results_fit_array))

    @property
    def results_fit_npts(self) -> int:
        """Points in fit array. 0 = same x-points as the source data."""
        return self.__results_fit_npts

    @results_fit_npts.setter
    def results_fit_npts(self, value: int) -> None:
        self._results_fit_npts_set(value)

    def _results_fit_npts_set(self, value: int, quiet: bool = nmc.QUIET) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(nmu.type_error_str(value, "results_fit_npts", "int"))
        if value < 0:
            raise ValueError("results_fit_npts must be >= 0, got %d" % value)
        self.__results_fit_npts = value
        nmh.history("set results_fit_npts=%d" % self.__results_fit_npts, quiet=quiet)
        nmch.add_nm_command("%s.results_fit_npts = %r" % (self._name, self.__results_fit_npts))

    # ------------------------------------------------------------------
    # Lifecycle

    def run_init(self) -> bool:
        """Reset internal state before the run loop."""
        self._fit_results = []
        self._epoch_names = []
        self._toolfolder = None
        return True

    def run(self) -> bool:
        """Fit the active function to the current NMData array.

        Returns:
            True on success.
        """
        data = self.data
        if not isinstance(data, NMData):
            raise RuntimeError("no data selected")
        if data.nparray is None:
            return True

        degree = _poly_degree(self.__func_name) if self.__func_name.startswith("poly") else 2
        result = fit_nmdata(
            data,
            func_name=self.__func_name,
            xbgn=self._xbgn,
            xend=self._xend,
            degree=degree,
            x_origin=self.__x_origin,
            p0=self.__p0,
            sigma=self.__sigma,
            maxfev=self.__maxfev,
            ignore_nans=self._ignore_nans,
        )
        self._fit_results.append(result)
        self._epoch_names.append(data.name)
        return True

    def run_finish(self) -> bool:
        """Persist results via enabled output sinks.

        Returns:
            True on success.
        """
        if not self._epoch_names:
            return True
        if self._results_to_history:
            self._write_results_to_history()
        if self._results_to_cache:
            self._write_results_to_cache()
        if self._results_to_numpy:
            self._write_results_to_numpy()
        return True

    # ------------------------------------------------------------------
    # Output sinks

    def _note_str(self) -> str:
        """Build the notes annotation string for output arrays."""
        parts = [
            "NMFit(func_name=%r" % self.__func_name,
            "xbgn=%s" % self._xbgn,
            "xend=%s" % self._xend,
            "n_epochs=%d" % len(self._epoch_names),
        ]
        if self.__func_name in ("exp", "exp2") and self.__x_origin != 0.0:
            parts.append("x_origin=%s" % self.__x_origin)
        if not self._ignore_nans:
            parts.append("ignore_nans=False")
        if self.__p0 is not None:
            parts.append("p0=%r" % self.__p0)
        return ", ".join(parts) + ")"

    def _add_note(self, data: NMData, text: str) -> None:
        notes = getattr(data, "notes", None)
        if notes is not None:
            notes.add(text)

    def _pname(self, default: str) -> str:
        """Return user-defined output name for *default*, or *default* if no mapping."""
        if self.__param_names:
            return self.__param_names.get(default, default)
        return default

    def _write_results_to_numpy(self) -> NMToolFolder | None:
        """Write FT_ NMData arrays to a Fit toolfolder.

        Output arrays depend on ``func_name``:

        * **line**: ``FT_M`` (slope), ``FT_B`` (intercept), ``FT_R2``,
          ``FT_ChiSqr``
        * **poly2–poly9**: ``FT_C0`` … ``FT_Cn``, ``FT_R2``, ``FT_ChiSqr``
        * **exp**: ``FT_A``, ``FT_B``, ``FT_X0``, ``FT_Y0``, ``FT_R2``,
          ``FT_ChiSqr``, ``FT_Converged``
        * **gauss**: ``FT_A``, ``FT_Mu``, ``FT_Sigma``, ``FT_Y0``,
          ``FT_R2``, ``FT_ChiSqr``, ``FT_Converged``

        All parameter array names pass through ``param_names`` so they can be
        customised (e.g. ``param_names={"A": "amplitude", "Y0": "baseline"}``).

        When ``results_errors`` is True, adds ``FT_err_*`` arrays with
        one-standard-deviation parameter uncertainties for each epoch.

        When ``results_residuals`` is True, adds per-epoch ``FT_resid_*``
        arrays containing the fit residuals (data − model) over the fit window.

        When ``results_fit_array`` is True, adds per-epoch ``Fit_*`` arrays
        containing the model evaluated at the fit x-values (or at a uniform
        grid of ``results_fit_npts`` points when that is non-zero).

        Returns:
            The NMToolFolder written to, or None if no folder is set.
        """
        if not isinstance(self.folder, NMFolder):
            return None
        prefix = "Fit_" + self.__func_name.capitalize()
        self._toolfolder = self._make_toolfolder(prefix, overwrite=self._overwrite)
        f = self._toolfolder
        note = self._note_str()
        epoch_scale = {"start": 0.0, "delta": 1.0, "units": "epoch"}

        p = self._pname  # shorthand; applied to parameter names, not metadata

        def _write(name: str, values: list[float]) -> NMData:
            d = f.data.new("FT_" + name,
                           nparray=np.array(values, dtype=float),
                           xscale=epoch_scale)
            self._add_note(d, note)
            return d

        if self.__func_name == "line":
            _write(p("M"),      [r["slope"]     for r in self._fit_results])
            _write(p("B"),      [r["intercept"] for r in self._fit_results])
            _write("R2",        [r["r2"]        for r in self._fit_results])
            _write("ChiSqr",    [r["chi_sqr"]   for r in self._fit_results])
            if self.__results_errors:
                _write("err_" + p("M"), [r["slope_err"]     for r in self._fit_results])
                _write("err_" + p("B"), [r["intercept_err"] for r in self._fit_results])

        elif self.__func_name.startswith("poly"):
            degree = _poly_degree(self.__func_name)
            for k in range(degree + 1):
                _write(p("C%d" % k), [r["coefficients"][k] for r in self._fit_results])
            _write("R2",     [r["r2"]      for r in self._fit_results])
            _write("ChiSqr", [r["chi_sqr"] for r in self._fit_results])
            if self.__results_errors:
                for k in range(degree + 1):
                    _write("err_" + p("C%d" % k),
                           [r["coef_errors"][k] for r in self._fit_results])

        elif self.__func_name == "exp":
            _write(p("A"),      [r["A"]   for r in self._fit_results])
            _write(p("Tau"),    [r["Tau"] for r in self._fit_results])
            _write(p("X0"),     [r["X0"]  for r in self._fit_results])
            _write(p("Y0"),     [r["Y0"]  for r in self._fit_results])
            _write("R2",        [r["r2"]        for r in self._fit_results])
            _write("ChiSqr",    [r["chi_sqr"]   for r in self._fit_results])
            _write("Converged", [float(r["converged"]) for r in self._fit_results])
            if self.__results_errors:
                _write("err_" + p("A"),   [r["A_err"]   for r in self._fit_results])
                _write("err_" + p("Tau"), [r["Tau_err"] for r in self._fit_results])
                _write("err_" + p("Y0"),  [r["Y0_err"]  for r in self._fit_results])

        elif self.__func_name == "exp2":
            _write(p("A1"),     [r["A1"]   for r in self._fit_results])
            _write(p("Tau1"),   [r["Tau1"] for r in self._fit_results])
            _write(p("A2"),     [r["A2"]   for r in self._fit_results])
            _write(p("Tau2"),   [r["Tau2"] for r in self._fit_results])
            _write(p("X0"),     [r["X0"]   for r in self._fit_results])
            _write(p("Y0"),     [r["Y0"]   for r in self._fit_results])
            _write("R2",        [r["r2"]        for r in self._fit_results])
            _write("ChiSqr",    [r["chi_sqr"]   for r in self._fit_results])
            _write("Converged", [float(r["converged"]) for r in self._fit_results])
            if self.__results_errors:
                _write("err_" + p("A1"),   [r["A1_err"]   for r in self._fit_results])
                _write("err_" + p("Tau1"), [r["Tau1_err"] for r in self._fit_results])
                _write("err_" + p("A2"),   [r["A2_err"]   for r in self._fit_results])
                _write("err_" + p("Tau2"), [r["Tau2_err"] for r in self._fit_results])
                _write("err_" + p("Y0"),   [r["Y0_err"]   for r in self._fit_results])

        elif self.__func_name == "gauss":
            _write(p("A"),      [r["A"]     for r in self._fit_results])
            _write(p("Mu"),     [r["Mu"]    for r in self._fit_results])
            _write(p("Sigma"),  [r["Sigma"] for r in self._fit_results])
            _write(p("Y0"),     [r["Y0"]    for r in self._fit_results])
            _write("R2",        [r["r2"]        for r in self._fit_results])
            _write("ChiSqr",    [r["chi_sqr"]   for r in self._fit_results])
            _write("Converged", [float(r["converged"]) for r in self._fit_results])
            if self.__results_errors:
                _write("err_" + p("A"),     [r["A_err"]     for r in self._fit_results])
                _write("err_" + p("Mu"),    [r["Mu_err"]    for r in self._fit_results])
                _write("err_" + p("Sigma"), [r["Sigma_err"] for r in self._fit_results])
                _write("err_" + p("Y0"),    [r["Y0_err"]    for r in self._fit_results])

        elif self.__func_name == "boltzmann":
            _write(p("A"),   [r["A"]   for r in self._fit_results])
            _write(p("V50"), [r["V50"] for r in self._fit_results])
            _write(p("K"),   [r["K"]   for r in self._fit_results])
            _write(p("Y0"),  [r["Y0"]  for r in self._fit_results])
            _write("R2",        [r["r2"]        for r in self._fit_results])
            _write("ChiSqr",    [r["chi_sqr"]   for r in self._fit_results])
            _write("Converged", [float(r["converged"]) for r in self._fit_results])
            if self.__results_errors:
                _write("err_" + p("A"),   [r["A_err"]   for r in self._fit_results])
                _write("err_" + p("V50"), [r["V50_err"] for r in self._fit_results])
                _write("err_" + p("K"),   [r["K_err"]   for r in self._fit_results])
                _write("err_" + p("Y0"),  [r["Y0_err"]  for r in self._fit_results])

        if self.__results_residuals:
            for name, result in zip(self._epoch_names, self._fit_results):
                d = f.data.new(
                    "FT_resid_" + name,
                    nparray=np.array(result["residuals"], dtype=float),
                    xarray=np.array(result["x"], dtype=float),
                )
                self._add_note(d, note)

        if self.__results_fit_array:
            for name, result in zip(self._epoch_names, self._fit_results):
                x_src = np.array(result["x"], dtype=float)
                if self.__results_fit_npts > 0:
                    x_fit = np.linspace(x_src[0], x_src[-1], self.__results_fit_npts)
                    y_fit = _eval_model(self.__func_name, result, x_fit)
                else:
                    x_fit = x_src
                    y_fit = np.array(result["yfit"], dtype=float)
                d = f.data.new(
                    "Fit_" + name,
                    nparray=y_fit,
                    xarray=x_fit,
                )
                self._add_note(d, note)

        f.data.new(
            "FT_epoch_names",
            nparray=np.array(self._epoch_names, dtype=object),
        )
        return f

    def _write_results_to_cache(self) -> None:
        """Save fit results dict to folder.toolresults."""
        if not isinstance(self.folder, NMFolder):
            return
        results = {
            name: result
            for name, result in zip(self._epoch_names, self._fit_results)
        }
        self.folder.toolresults_save("fit", results)

    def _write_results_to_history(self) -> None:
        """Print fit results to the history log."""
        for name, result in zip(self._epoch_names, self._fit_results):
            nmh.history("fit: %s: %s R²=%.4f" % (name, self.__func_name, result["r2"]))
