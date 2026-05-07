# -*- coding: utf-8 -*-
"""
NMToolEvent: spontaneous event detection tool.

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
from pyneuromatic.analysis.nm_tool_utilities import find_events_nmdata
import pyneuromatic.core.nm_math as nm_math

_VALID_ALGORITHMS: frozenset[str] = frozenset(
    {"threshold", "nstdv", "template"}
)
_VALID_POLARITIES: frozenset[str] = frozenset({"negative", "positive"})


class NMToolEventConfig(NMToolConfig):
    """Configuration for NMToolEvent.

    Parameters:
        algorithm: Detection algorithm — ``"threshold"`` (sliding baseline,
            fixed amplitude, default), ``"nstdv"`` (sliding baseline,
            N×stdv), or ``"template"`` (Clements & Bekkers 1997).
        polarity: ``"positive"`` (upward deflections, default) or
            ``"negative"`` (downward deflections).
        threshold: Threshold amplitude above baseline (algorithm="threshold"). Default 10.0.
        nstdv: N×stdv above baseline (algorithm="nstdv"). Default 4.0.
        baseline_avg: Baseline averaging window (x-units, threshold/nstdv).
            Default 4.0.
        baseline_dt: Detection window after t0 (x-units, threshold/nstdv).
            Default 2.0.
        criterion_threshold: Criterion threshold for template matching. Default 4.0.
        template_baseline: Baseline window (x-units) prepended to the template as
            zeros before matching. Detected times are shifted forward by this
            amount to recover the true event onset. Default 0.0.
        onset_search: If True, search backward for event onset. Default True.
        onset_avg: Onset window size (x-units). Default 1.0.
        onset_nstdv: Onset N×stdv. Default 1.0.
        onset_limit: Max backward onset search (x-units). Default 2.0.
        peak_search: If True, search forward for event peak. Default True.
        peak_avg: Peak window size (x-units). Default 1.0.
        peak_nstdv: Peak N×stdv. Default 1.0.
        peak_limit: Max forward peak search (x-units). Default 4.0.
        refractory: Minimum inter-event interval (x-units). Default 0.0.
        x0: Search start (x-units). Default ``-inf``.
        x1: Search end (x-units). Default ``+inf``.
        max_events: Stop after this many accepted events per epoch. 0 means
            no limit (default). Useful for debugging and GUI use.
        overwrite: Reuse existing toolfolder instead of creating a new one.
            Default True.
        results_to_history: Print event counts to history log. Default False.
        results_to_cache: Save results dict to folder.toolresults. Default True.
        results_to_numpy: Write EV_ NMData arrays to a toolfolder. Default True.
    """

    _TOML_TYPE = "event_config"
    _schema = {
        "algorithm":          {"type": str,   "default": "threshold",
                               "choices": ["threshold", "nstdv", "template"]},
        "polarity":           {"type": str,   "default": "positive",
                               "choices": ["negative", "positive"]},
        "threshold":          {"type": float, "default": 10.0,  "min": 0.0},
        "nstdv":              {"type": float, "default": 4.0,   "min": 0.0},
        "baseline_avg":       {"type": float, "default": 4.0,   "min": 0.0},
        "baseline_dt":        {"type": float, "default": 2.0,   "min": 0.0},
        "criterion_threshold": {"type": float, "default": 4.0,   "min": 0.0},
        "template_baseline":  {"type": float, "default": 0.0,   "min": 0.0},
        "onset_search":       {"type": bool,  "default": True},
        "onset_avg":          {"type": float, "default": 1.0,   "min": 0.0},
        "onset_nstdv":        {"type": float, "default": 1.0,   "min": 0.0},
        "onset_limit":        {"type": float, "default": 2.0,   "min": 0.0},
        "peak_search":        {"type": bool,  "default": True},
        "peak_avg":           {"type": float, "default": 1.0,   "min": 0.0},
        "peak_nstdv":         {"type": float, "default": 1.0,   "min": 0.0},
        "peak_limit":         {"type": float, "default": 4.0,   "min": 0.0},
        "refractory":         {"type": float, "default": 0.0,   "min": 0.0},
        "x0":                 {"type": float, "default": -math.inf},
        "x1":                 {"type": float, "default":  math.inf},
        "max_events":         {"type": int,   "default": 0,        "min": 0},
        "overwrite":          {"type": bool,  "default": True},
        "results_to_history": {"type": bool,  "default": False},
        "results_to_cache":   {"type": bool,  "default": True},
        "results_to_numpy":   {"type": bool,  "default": True},
    }


class NMToolEvent(NMTool):
    """Spontaneous event detection tool.

    Detects spontaneous events (e.g. EPSPs, EPSCs) in NMData arrays using
    one of three algorithms, with optional onset and peak refinement based
    on Kudoh & Taguchi (2002) and Clements & Bekkers (1997).

    After a run, per-epoch result arrays (``EV_`` prefix) are written to a
    new Event subfolder. Rejected events (failed onset/peak check) are stored
    in ``EV_reject_`` arrays alongside the accepted ``EV_`` arrays.

    For interactive (GUI) use, :meth:`find_next_event` provides a single-event
    primitive that can be driven event-by-event.

    Attributes:
        algorithm: Detection algorithm. ``"threshold"`` (default), ``"nstdv"``,
            or ``"template"``.
        polarity: ``"positive"`` (default) or ``"negative"``.
        threshold: Amplitude threshold above baseline (algorithm="threshold"). Default 10.0.
        nstdv: N×stdv threshold above baseline (algorithm="nstdv"). Default 4.0.
        baseline_avg: Baseline window size (x-units). Default 4.0.
        baseline_dt: Detection window after t0 (x-units). Default 2.0.
        template: User-supplied 1-D numpy template (algorithm="template").
            Should contain only the event shape — the baseline is prepended
            automatically via *template_baseline*. Not TOML-serializable;
            set directly on the instance.
        criterion_threshold: Criterion threshold for template matching. Default 4.0.
        template_baseline: Baseline window (x-units) prepended to the template as
            zeros before matching. Detected times are shifted forward by this
            amount to recover the true event onset. Default 0.0.
        onset_search: Enable onset refinement. Default True.
        onset_avg: Onset window (x-units). Default 1.0.
        onset_nstdv: Onset N×stdv. Default 1.0.
        onset_limit: Max backward onset search (x-units). Default 2.0.
        peak_search: Enable peak refinement. Default True.
        peak_avg: Peak window (x-units). Default 1.0.
        peak_nstdv: Peak N×stdv. Default 1.0.
        peak_limit: Max forward peak search (x-units). Default 4.0.
        refractory: Minimum inter-event interval (x-units). Default 0.0.
        x0: Search start (x-units). Default ``-inf``.
        x1: Search end (x-units). Default ``+inf``.
        max_events: Stop after this many accepted events per epoch. 0 means
            no limit (default). Useful for debugging and GUI use.
    """

    def __init__(self) -> None:
        super().__init__(name="event")
        self._config = NMToolEventConfig()

        self._overwrite          = self._config.overwrite
        self._results_to_history = self._config.results_to_history
        self._results_to_cache   = self._config.results_to_cache
        self._results_to_numpy   = self._config.results_to_numpy

        self.__algorithm:     str   = "threshold"
        self.__polarity:      str   = "positive"
        self.__threshold:     float = 10.0
        self.__nstdv:         float = 4.0
        self.__baseline_avg:  float = 4.0
        self.__baseline_dt:   float = 2.0
        self.__template:          np.ndarray | None = None
        self.__criterion_threshold: float = 4.0
        self.__template_baseline:   float = 0.0
        self.__onset_search:  bool  = True
        self.__onset_avg:     float = 1.0
        self.__onset_nstdv:   float = 1.0
        self.__onset_limit:   float = 2.0
        self.__peak_search:   bool  = True
        self.__peak_avg:      float = 1.0
        self.__peak_nstdv:    float = 1.0
        self.__peak_limit:    float = 4.0
        self.__refractory:    float = 0.0
        self.__x0:            float = -math.inf
        self.__x1:            float = math.inf
        self.__max_events:    int   = 0

        # Match criterion cache — keyed by id(data.nparray), survives run_init()
        self._match_criterion_cache_id: int | None        = None
        self._match_criterion_cache:    np.ndarray | None = None

        # Internal run state — reset by run_init()
        self._detect_times:    list[list[float]]           = []
        self._onset_times:     list[list[float | None]]    = []
        self._peak_times:      list[list[float | None]]    = []
        self._reject_times:    list[list[float]]           = []
        self._match_criterion: list[np.ndarray | None]     = []
        self._epoch_names:     list[str]                   = []
        self._detected_xunits: str | None                  = None
        self._toolfolder:      NMToolFolder | None         = None

    # ------------------------------------------------------------------
    # Properties

    @property
    def algorithm(self) -> str:
        """Detection algorithm: ``'threshold'``, ``'nstdv'``, or ``'template'``."""
        return self.__algorithm

    @algorithm.setter
    def algorithm(self, value: str) -> None:
        self._algorithm_set(value)

    def _algorithm_set(self, value: str, quiet: bool = nmc.QUIET) -> None:
        if not isinstance(value, str):
            raise TypeError(nmu.type_error_str(value, "algorithm", "string"))
        if value not in _VALID_ALGORITHMS:
            raise ValueError(
                "algorithm must be one of %s, got %r"
                % (sorted(_VALID_ALGORITHMS), value)
            )
        self.__algorithm = value
        nmh.history("set algorithm=%r" % self.__algorithm, quiet=quiet)
        nmch.add_nm_command("%s.algorithm = %r" % (self._name, self.__algorithm))

    @property
    def polarity(self) -> str:
        """Event polarity: ``'negative'`` or ``'positive'``."""
        return self.__polarity

    @polarity.setter
    def polarity(self, value: str) -> None:
        self._polarity_set(value)

    def _polarity_set(self, value: str, quiet: bool = nmc.QUIET) -> None:
        if not isinstance(value, str):
            raise TypeError(nmu.type_error_str(value, "polarity", "string"))
        if value not in _VALID_POLARITIES:
            raise ValueError(
                "polarity must be one of %s, got %r"
                % (sorted(_VALID_POLARITIES), value)
            )
        self.__polarity = value
        nmh.history("set polarity=%r" % self.__polarity, quiet=quiet)
        nmch.add_nm_command("%s.polarity = %r" % (self._name, self.__polarity))

    @property
    def threshold(self) -> float:
        """Amplitude threshold (algorithm='threshold')."""
        return self.__threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold_set(value)

    def _threshold_set(self, value: float, quiet: bool = nmc.QUIET) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "threshold", "float"))
        value = float(value)
        if value < 0:
            raise ValueError("threshold must be >= 0, got %g" % value)
        self.__threshold = value
        nmh.history("set threshold=%g" % self.__threshold, quiet=quiet)
        nmch.add_nm_command("%s.threshold = %r" % (self._name, self.__threshold))

    @property
    def nstdv(self) -> float:
        """N×stdv threshold (algorithm='nstdv')."""
        return self.__nstdv

    @nstdv.setter
    def nstdv(self, value: float) -> None:
        self._nstdv_set(value)

    def _nstdv_set(self, value: float, quiet: bool = nmc.QUIET) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "nstdv", "float"))
        value = float(value)
        if value < 0:
            raise ValueError("nstdv must be >= 0, got %g" % value)
        self.__nstdv = value
        nmh.history("set nstdv=%g" % self.__nstdv, quiet=quiet)
        nmch.add_nm_command("%s.nstdv = %r" % (self._name, self.__nstdv))

    @property
    def baseline_avg(self) -> float:
        """Baseline averaging window size (x-units)."""
        return self.__baseline_avg

    @baseline_avg.setter
    def baseline_avg(self, value: float) -> None:
        self._baseline_avg_set(value)

    def _baseline_avg_set(self, value: float, quiet: bool = nmc.QUIET) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "baseline_avg", "float"))
        value = float(value)
        if value < 0:
            raise ValueError("baseline_avg must be >= 0, got %g" % value)
        self.__baseline_avg = value
        nmh.history("set baseline_avg=%g" % self.__baseline_avg, quiet=quiet)
        nmch.add_nm_command(
            "%s.baseline_avg = %r" % (self._name, self.__baseline_avg)
        )

    @property
    def baseline_dt(self) -> float:
        """Detection window after baseline midpoint (x-units)."""
        return self.__baseline_dt

    @baseline_dt.setter
    def baseline_dt(self, value: float) -> None:
        self._baseline_dt_set(value)

    def _baseline_dt_set(self, value: float, quiet: bool = nmc.QUIET) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "baseline_dt", "float"))
        value = float(value)
        if value <= 0:
            raise ValueError("baseline_dt must be > 0, got %g" % value)
        self.__baseline_dt = value
        nmh.history("set baseline_dt=%g" % self.__baseline_dt, quiet=quiet)
        nmch.add_nm_command(
            "%s.baseline_dt = %r" % (self._name, self.__baseline_dt)
        )

    @property
    def template(self) -> np.ndarray | None:
        """User-supplied event template (numpy array, algorithm='template')."""
        return self.__template

    @template.setter
    def template(self, value: np.ndarray | None) -> None:
        self._template_set(value)

    def _template_set(
        self, value: np.ndarray | None, quiet: bool = nmc.QUIET
    ) -> None:
        if value is not None and not isinstance(value, np.ndarray):
            raise TypeError(
                nmu.type_error_str(value, "template", "numpy.ndarray or None")
            )
        if value is not None and value.ndim != 1:
            raise ValueError("template must be 1-D, got shape %s" % str(value.shape))
        self.__template = value
        n = len(value) if value is not None else 0
        nmh.history("set template (n=%d)" % n, quiet=quiet)
        nmch.add_nm_command("%s.template = <array n=%d>" % (self._name, n))

    @property
    def criterion_threshold(self) -> float:
        """Match-criterion detection threshold (algorithm='template'). Default 4.0."""
        return self.__criterion_threshold

    @criterion_threshold.setter
    def criterion_threshold(self, value: float) -> None:
        self._criterion_threshold_set(value)

    def _criterion_threshold_set(self, value: float, quiet: bool = nmc.QUIET) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "criterion_threshold", "float"))
        value = float(value)
        if value <= 0:
            raise ValueError("criterion_threshold must be > 0, got %g" % value)
        self.__criterion_threshold = value
        nmh.history("set criterion_threshold=%g" % self.__criterion_threshold, quiet=quiet)
        nmch.add_nm_command(
            "%s.criterion_threshold = %r" % (self._name, self.__criterion_threshold)
        )

    @property
    def template_baseline(self) -> float:
        """Baseline window (x-units) prepended to template as zeros before matching.

        The criterion crossing occurs at the start of this baseline window, so
        detected times are shifted forward by *template_baseline* to recover the
        true event onset. Default 0.0 (no baseline prepended).
        """
        return self.__template_baseline

    @template_baseline.setter
    def template_baseline(self, value: float) -> None:
        self._template_baseline_set(value)

    def _template_baseline_set(self, value: float, quiet: bool = nmc.QUIET) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "template_baseline", "float"))
        value = float(value)
        if value < 0:
            raise ValueError("template_baseline must be >= 0, got %g" % value)
        self.__template_baseline = value
        nmh.history("set template_baseline=%g" % self.__template_baseline, quiet=quiet)
        nmch.add_nm_command(
            "%s.template_baseline = %r" % (self._name, self.__template_baseline)
        )

    @property
    def onset_search(self) -> bool:
        """Enable backward onset search after each detection."""
        return self.__onset_search

    @onset_search.setter
    def onset_search(self, value: bool) -> None:
        self._onset_search_set(value)

    def _onset_search_set(self, value: bool, quiet: bool = nmc.QUIET) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "onset_search", "bool"))
        self.__onset_search = value
        nmh.history("set onset_search=%r" % self.__onset_search, quiet=quiet)
        nmch.add_nm_command(
            "%s.onset_search = %r" % (self._name, self.__onset_search)
        )

    @property
    def onset_avg(self) -> float:
        """Onset sliding window size (x-units)."""
        return self.__onset_avg

    @onset_avg.setter
    def onset_avg(self, value: float) -> None:
        self._onset_avg_set(value)

    def _onset_avg_set(self, value: float, quiet: bool = nmc.QUIET) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "onset_avg", "float"))
        value = float(value)
        if value < 0:
            raise ValueError("onset_avg must be >= 0, got %g" % value)
        self.__onset_avg = value
        nmh.history("set onset_avg=%g" % self.__onset_avg, quiet=quiet)
        nmch.add_nm_command("%s.onset_avg = %r" % (self._name, self.__onset_avg))

    @property
    def onset_nstdv(self) -> float:
        """Onset N×stdv."""
        return self.__onset_nstdv

    @onset_nstdv.setter
    def onset_nstdv(self, value: float) -> None:
        self._onset_nstdv_set(value)

    def _onset_nstdv_set(self, value: float, quiet: bool = nmc.QUIET) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "onset_nstdv", "float"))
        value = float(value)
        if value < 0:
            raise ValueError("onset_nstdv must be >= 0, got %g" % value)
        self.__onset_nstdv = value
        nmh.history("set onset_nstdv=%g" % self.__onset_nstdv, quiet=quiet)
        nmch.add_nm_command(
            "%s.onset_nstdv = %r" % (self._name, self.__onset_nstdv)
        )

    @property
    def onset_limit(self) -> float:
        """Max backward onset search distance (x-units)."""
        return self.__onset_limit

    @onset_limit.setter
    def onset_limit(self, value: float) -> None:
        self._onset_limit_set(value)

    def _onset_limit_set(self, value: float, quiet: bool = nmc.QUIET) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "onset_limit", "float"))
        value = float(value)
        if value <= 0:
            raise ValueError("onset_limit must be > 0, got %g" % value)
        self.__onset_limit = value
        nmh.history("set onset_limit=%g" % self.__onset_limit, quiet=quiet)
        nmch.add_nm_command(
            "%s.onset_limit = %r" % (self._name, self.__onset_limit)
        )

    @property
    def peak_search(self) -> bool:
        """Enable forward peak search after each detection."""
        return self.__peak_search

    @peak_search.setter
    def peak_search(self, value: bool) -> None:
        self._peak_search_set(value)

    def _peak_search_set(self, value: bool, quiet: bool = nmc.QUIET) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "peak_search", "bool"))
        self.__peak_search = value
        nmh.history("set peak_search=%r" % self.__peak_search, quiet=quiet)
        nmch.add_nm_command(
            "%s.peak_search = %r" % (self._name, self.__peak_search)
        )

    @property
    def peak_avg(self) -> float:
        """Peak sliding window size (x-units)."""
        return self.__peak_avg

    @peak_avg.setter
    def peak_avg(self, value: float) -> None:
        self._peak_avg_set(value)

    def _peak_avg_set(self, value: float, quiet: bool = nmc.QUIET) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "peak_avg", "float"))
        value = float(value)
        if value < 0:
            raise ValueError("peak_avg must be >= 0, got %g" % value)
        self.__peak_avg = value
        nmh.history("set peak_avg=%g" % self.__peak_avg, quiet=quiet)
        nmch.add_nm_command("%s.peak_avg = %r" % (self._name, self.__peak_avg))

    @property
    def peak_nstdv(self) -> float:
        """Peak N×stdv."""
        return self.__peak_nstdv

    @peak_nstdv.setter
    def peak_nstdv(self, value: float) -> None:
        self._peak_nstdv_set(value)

    def _peak_nstdv_set(self, value: float, quiet: bool = nmc.QUIET) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "peak_nstdv", "float"))
        value = float(value)
        if value < 0:
            raise ValueError("peak_nstdv must be >= 0, got %g" % value)
        self.__peak_nstdv = value
        nmh.history("set peak_nstdv=%g" % self.__peak_nstdv, quiet=quiet)
        nmch.add_nm_command(
            "%s.peak_nstdv = %r" % (self._name, self.__peak_nstdv)
        )

    @property
    def peak_limit(self) -> float:
        """Max forward peak search distance (x-units)."""
        return self.__peak_limit

    @peak_limit.setter
    def peak_limit(self, value: float) -> None:
        self._peak_limit_set(value)

    def _peak_limit_set(self, value: float, quiet: bool = nmc.QUIET) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "peak_limit", "float"))
        value = float(value)
        if value <= 0:
            raise ValueError("peak_limit must be > 0, got %g" % value)
        self.__peak_limit = value
        nmh.history("set peak_limit=%g" % self.__peak_limit, quiet=quiet)
        nmch.add_nm_command(
            "%s.peak_limit = %r" % (self._name, self.__peak_limit)
        )

    @property
    def refractory(self) -> float:
        """Minimum inter-event interval (x-units)."""
        return self.__refractory

    @refractory.setter
    def refractory(self, value: float) -> None:
        self._refractory_set(value)

    def _refractory_set(self, value: float, quiet: bool = nmc.QUIET) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "refractory", "float"))
        value = float(value)
        if value < 0:
            raise ValueError("refractory must be >= 0, got %g" % value)
        self.__refractory = value
        nmh.history("set refractory=%g" % self.__refractory, quiet=quiet)
        nmch.add_nm_command(
            "%s.refractory = %r" % (self._name, self.__refractory)
        )

    @property
    def x0(self) -> float:
        """X-axis search start. Default ``-inf``."""
        return self.__x0

    @x0.setter
    def x0(self, value: float) -> None:
        self._x0_set(value)

    def _x0_set(self, value: float, quiet: bool = nmc.QUIET) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "x0", "float"))
        if math.isnan(value):
            raise ValueError("x0 cannot be NaN")
        self.__x0 = float(value)
        nmh.history("set x0=%g" % self.__x0, quiet=quiet)
        nmch.add_nm_command("%s.x0 = %r" % (self._name, self.__x0))

    @property
    def x1(self) -> float:
        """X-axis search end. Default ``+inf``."""
        return self.__x1

    @x1.setter
    def x1(self, value: float) -> None:
        self._x1_set(value)

    def _x1_set(self, value: float, quiet: bool = nmc.QUIET) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "x1", "float"))
        if math.isnan(value):
            raise ValueError("x1 cannot be NaN")
        self.__x1 = float(value)
        nmh.history("set x1=%g" % self.__x1, quiet=quiet)
        nmch.add_nm_command("%s.x1 = %r" % (self._name, self.__x1))

    @property
    def max_events(self) -> int:
        """Max accepted events per epoch (0 = no limit)."""
        return self.__max_events

    @max_events.setter
    def max_events(self, value: int) -> None:
        self._max_events_set(value)

    def _max_events_set(self, value: int, quiet: bool = nmc.QUIET) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(nmu.type_error_str(value, "max_events", "int"))
        if value < 0:
            raise ValueError("max_events must be >= 0, got %d" % value)
        self.__max_events = value
        nmh.history("set max_events=%d" % self.__max_events, quiet=quiet)
        nmch.add_nm_command("%s.max_events = %r" % (self._name, self.__max_events))

    # ------------------------------------------------------------------
    # Helpers

    def _get_match_criterion(self, data: NMData) -> np.ndarray | None:
        """Return the match template criterion array for *data*, using a cache.

        Normalizes ``self.template`` to [0, 1] and calls
        ``nm_math.match_template()`` only when the data array has changed
        since the last call (detected via ``id(data.nparray)``).  Returns
        ``None`` when ``algorithm != "template"`` or ``template`` is unset.
        """
        if self.__algorithm != "template" or self.__template is None:
            return None
        arr = data.nparray
        if arr is None:
            return None
        cache_key = (id(arr), self.__template_baseline)
        if cache_key != self._match_criterion_cache_id:
            tpl = self.__template.astype(float, copy=True)
            tpl_min, tpl_max = tpl.min(), tpl.max()
            if tpl_max != tpl_min:
                tpl = (tpl - tpl_min) / (tpl_max - tpl_min)
            if self.__template_baseline > 0:
                xdelta = data.xscale.delta if data.xscale.delta else 1.0
                n_base = max(1, round(self.__template_baseline / xdelta))
                tpl = np.concatenate([np.zeros(n_base), tpl])
            self._match_criterion_cache    = nm_math.match_template(arr, tpl)
            self._match_criterion_cache_id = cache_key
        return self._match_criterion_cache

    def _detection_value(self) -> float:
        """Return the active detection value for the current algorithm."""
        if self.__algorithm == "nstdv":
            return self.__nstdv
        return self.__threshold  # "threshold" or "template" (ignored by template path)

    def _note_str(self, epoch_name: str, n_accept: int, n_reject: int) -> str:
        alg = self.__algorithm
        pol = self.__polarity
        if alg == "threshold":
            params = "threshold=%g, baseline_avg=%g, baseline_dt=%g" % (
                self.__threshold, self.__baseline_avg, self.__baseline_dt,
            )
        elif alg == "nstdv":
            params = "nstdv=%g, baseline_avg=%g, baseline_dt=%g" % (
                self.__nstdv, self.__baseline_avg, self.__baseline_dt,
            )
        else:
            params = "criterion_threshold=%g, template_baseline=%g" % (
                self.__criterion_threshold, self.__template_baseline,
            )
        return (
            "NMEvent(source=%s, algorithm=%r, polarity=%r, %s, "
            "onset_search=%r, peak_search=%r, refractory=%g, "
            "x0=%s, x1=%s, n_accept=%d, n_reject=%d)"
            % (
                epoch_name, alg, pol, params,
                self.__onset_search, self.__peak_search, self.__refractory,
                self.__x0, self.__x1, n_accept, n_reject,
            )
        )

    def _add_note(self, data: NMData, text: str) -> None:
        notes = getattr(data, "notes", None)
        if notes is not None:
            notes.add(text)

    # ------------------------------------------------------------------
    # Lifecycle

    def run_init(self) -> bool:
        """Reset internal state and validate template if needed."""
        if self.__algorithm == "template" and self.__template is None:
            raise ValueError(
                "template must be set before running algorithm='template'"
            )
        self._detect_times    = []
        self._onset_times     = []
        self._peak_times      = []
        self._reject_times    = []
        self._match_criterion = []
        self._epoch_names     = []
        self._detected_xunits = None
        self._toolfolder      = None
        return True

    def run(self) -> bool:
        """Find events in the current NMData array.

        Calls :func:`find_events_nmdata` for the full x0→x1 range and
        appends per-epoch results to the internal accumulator lists.

        Returns:
            True on success.
        """
        data = self.data
        if not isinstance(data, NMData):
            raise RuntimeError("no data selected")
        if data.nparray is None:
            return True
        if self._detected_xunits is None:
            self._detected_xunits = data.xscale.units

        res = find_events_nmdata(
            data,
            algorithm=self.__algorithm,
            polarity=self.__polarity,
            value=self._detection_value(),
            baseline_avg=self.__baseline_avg,
            baseline_dt=self.__baseline_dt,
            template=self.__template,
            criterion_threshold=self.__criterion_threshold,
            template_baseline=self.__template_baseline,
            refractory=self.__refractory,
            x0=self.__x0,
            x1=self.__x1,
            onset_search=self.__onset_search,
            onset_avg=self.__onset_avg,
            onset_nstdv=self.__onset_nstdv,
            onset_limit=self.__onset_limit,
            peak_search=self.__peak_search,
            peak_avg=self.__peak_avg,
            peak_nstdv=self.__peak_nstdv,
            peak_limit=self.__peak_limit,
            max_events=self.__max_events,
            match_criterion=self._get_match_criterion(data),
        )

        self._detect_times.append(res["detect_times"])
        self._onset_times.append(res["onset_times"])
        self._peak_times.append(res["peak_times"])
        self._reject_times.append(res["reject_times"])
        self._match_criterion.append(res["match_criterion"])
        self._epoch_names.append(data.name)
        return True

    def run_finish(self) -> bool:
        """Persist results via the enabled output sinks.

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

    def _write_results_to_numpy(self) -> NMToolFolder | None:
        """Write EV_ NMData arrays to a new Event subfolder."""
        if not isinstance(self.folder, NMFolder):
            return None
        self._toolfolder = self._make_toolfolder("Event", overwrite=self._overwrite)
        f = self._toolfolder
        xunits = self._detected_xunits or ""

        # Store event-shape template (normalized to [0, 1], baseline not included)
        if self.__algorithm == "template" and self.__template is not None:
            tpl = self.__template.astype(float, copy=True)
            tmin, tmax = tpl.min(), tpl.max()
            if tmax != tmin:
                tpl = (tpl - tmin) / (tmax - tmin)
            f.data.new("EV_template", nparray=tpl,
                       yscale={"label": "Normalized template", "units": ""})

        counts        = np.zeros(len(self._epoch_names), dtype=float)
        reject_counts = np.zeros(len(self._epoch_names), dtype=float)

        for i, name in enumerate(self._epoch_names):
            dt = self._detect_times[i]
            rt = self._reject_times[i]
            ot = self._onset_times[i]
            pt = self._peak_times[i]
            mc = self._match_criterion[i]

            counts[i]        = len(dt)
            reject_counts[i] = len(rt)
            note = self._note_str(name, len(dt), len(rt))

            # Accepted detect times
            d = f.data.new(
                "EV_" + name,
                nparray=np.array(dt, dtype=float),
                yscale={"label": "Time", "units": xunits},
            )
            self._add_note(d, note)

            # Onset times (if onset_search enabled)
            if self.__onset_search:
                onset_arr = np.array(
                    [t if t is not None else math.nan for t in ot], dtype=float
                )
                d_on = f.data.new(
                    "EV_onset_" + name,
                    nparray=onset_arr,
                    yscale={"label": "Time", "units": xunits},
                )
                self._add_note(d_on, note)

            # Peak times (if peak_search enabled)
            if self.__peak_search:
                peak_arr = np.array(
                    [t if t is not None else math.nan for t in pt], dtype=float
                )
                d_pk = f.data.new(
                    "EV_peak_" + name,
                    nparray=peak_arr,
                    yscale={"label": "Time", "units": xunits},
                )
                self._add_note(d_pk, note)

            # Rejected detect times
            d_rej = f.data.new(
                "EV_reject_" + name,
                nparray=np.array(rt, dtype=float),
                yscale={"label": "Time", "units": xunits},
            )
            self._add_note(d_rej, note)

            # Template matching criterion wave (template algorithm only)
            if mc is not None:
                f.data.new(
                    "EV_Match_" + name,
                    nparray=mc,
                    yscale={"label": "Match criterion (scale/SE)", "units": ""},
                )

        d_cnt = f.data.new("EV_count", nparray=counts)
        self._add_note(d_cnt, "NMEvent: accepted event count per epoch")

        d_rcnt = f.data.new("EV_reject_count", nparray=reject_counts)
        self._add_note(d_rcnt, "NMEvent: rejected event count per epoch")

        f.data.new(
            "EV_epoch_names",
            nparray=np.array(self._epoch_names, dtype=object),
        )
        return f

    def _write_results_to_cache(self) -> None:
        """Save results dict to folder.toolresults."""
        if not isinstance(self.folder, NMFolder):
            return
        results = {
            name: {
                "detect": dt,
                "onset":  ot,
                "peak":   pt,
                "reject": rt,
            }
            for name, dt, ot, pt, rt in zip(
                self._epoch_names,
                self._detect_times,
                self._onset_times,
                self._peak_times,
                self._reject_times,
            )
        }
        self.folder.toolresults_save("event", results)

    def _write_results_to_history(self) -> None:
        """Print event counts to the history log."""
        for name, dt, rt in zip(
            self._epoch_names, self._detect_times, self._reject_times
        ):
            nmh.history(
                "event: %s: %d accepted, %d rejected"
                % (name, len(dt), len(rt))
            )

    # ------------------------------------------------------------------
    # Interactive (GUI) primitive

    def find_next_event(
        self,
        data: NMData,
        start_x: float,
    ) -> dict | None:
        """Find the next single event at or after *start_x*.

        Uses the same detection parameters as :meth:`run` but stops after the
        first candidate. Intended as a building block for interactive (GUI)
        event-by-event detection.

        Args:
            data: NMData array to search.
            start_x: X-time to start searching from (x-units).

        Returns:
            Dict with keys ``"t_event"``, ``"t_onset"`` (float or None),
            ``"t_peak"`` (float or None), ``"accepted"`` (bool), and
            ``"match_criterion"`` (np.ndarray or None for template).
            Returns None if no event is found.
        """
        if data.nparray is None:
            return None

        res = find_events_nmdata(
            data,
            algorithm=self.__algorithm,
            polarity=self.__polarity,
            value=self._detection_value(),
            baseline_avg=self.__baseline_avg,
            baseline_dt=self.__baseline_dt,
            template=self.__template,
            criterion_threshold=self.__criterion_threshold,
            template_baseline=self.__template_baseline,
            refractory=self.__refractory,
            x0=start_x,
            x1=self.__x1,
            onset_search=self.__onset_search,
            onset_avg=self.__onset_avg,
            onset_nstdv=self.__onset_nstdv,
            onset_limit=self.__onset_limit,
            peak_search=self.__peak_search,
            peak_avg=self.__peak_avg,
            peak_nstdv=self.__peak_nstdv,
            peak_limit=self.__peak_limit,
            max_events=1,
            match_criterion=self._get_match_criterion(data),
        )

        # Return first accepted event, or first rejected if none accepted
        if res["detect_times"]:
            return {
                "t_event":        res["detect_times"][0],
                "t_onset":        res["onset_times"][0],
                "t_peak":         res["peak_times"][0],
                "accepted":       True,
                "match_criterion": res["match_criterion"],
            }
        if res["reject_times"]:
            return {
                "t_event":        res["reject_times"][0],
                "t_onset":        None,
                "t_peak":         None,
                "accepted":       False,
                "match_criterion": res["match_criterion"],
            }
        return None
