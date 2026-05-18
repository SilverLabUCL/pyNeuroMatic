#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for nm_tool_event: NMToolEvent.

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from pyneuromatic.analysis.nm_tool_event import NMToolEvent, NMToolEventConfig
from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_folder import NMFolder
from pyneuromatic.core.nm_manager import NMManager, HIERARCHY_SELECT_KEYS

NM = NMManager(quiet=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SR    = 10000.0          # sample rate Hz
_DELTA = 1.0 / _SR        # seconds per sample (0.1 ms)
_MS    = 1e-3             # one millisecond in seconds


def _make_epsc_data(
    name: str = "recordA0",
    n: int = 10000,
    event_times_ms: list[float] | None = None,
    amplitude: float = -50.0,
    decay_ms: float = 5.0,
    baseline: float = 0.0,
    noise_std: float = 0.0,
    seed: int = 0,
) -> NMData:
    """NMData containing a flat baseline with EPSC-like decaying exponentials.

    Each event is a negative (or positive) exponential starting at event_time:
        y = amplitude * exp(-(t - t0) / decay)  for t >= t0

    Args:
        name: NMData name.
        n: Number of samples.
        event_times_ms: List of event onset times in milliseconds.
        amplitude: Event peak amplitude (negative for negative events).
        decay_ms: Exponential decay time constant (ms).
        baseline: Baseline y-value.
        noise_std: Standard deviation of Gaussian noise added to baseline.
        seed: Random seed for reproducible noise.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) * _DELTA
    y = np.full(n, baseline, dtype=float)
    if noise_std > 0:
        y += rng.normal(0.0, noise_std, n)
    decay_s = decay_ms * _MS
    for t0_ms in (event_times_ms or []):
        t0 = t0_ms * _MS
        mask = t >= t0
        y[mask] += amplitude * np.exp(-(t[mask] - t0) / decay_s)
    return NMData(
        NM,
        name=name,
        nparray=y,
        xscale={"start": 0.0, "delta": _DELTA, "label": "Time", "units": "s"},
        yscale={"label": "Current", "units": "pA"},
    )


def _make_targets(data_list: list, folder: NMFolder | None = None) -> list:
    if folder is None:
        folder = NMFolder(NM, name="TestFolder")
    return [
        {k: None for k in HIERARCHY_SELECT_KEYS} | {"folder": folder, "data": d}
        for d in data_list
    ]


def _run(tool: NMToolEvent, data_list: list, folder: NMFolder | None = None) -> NMFolder:
    if folder is None:
        folder = NMFolder(NM, name="TestFolder")
    tool.run_all(_make_targets(data_list, folder=folder))
    return folder


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestNMToolEventDefaults(unittest.TestCase):

    def setUp(self):
        self.tool = NMToolEvent()

    def test_algorithm_default(self):
        self.assertEqual(self.tool.algorithm, "threshold")

    def test_polarity_default(self):
        self.assertEqual(self.tool.polarity, "positive")

    def test_threshold_default(self):
        self.assertEqual(self.tool.threshold, 10.0)

    def test_nstdv_default(self):
        self.assertEqual(self.tool.nstdv, 4.0)

    def test_baseline_avg_default(self):
        self.assertEqual(self.tool.baseline_avg, 4.0)

    def test_baseline_dt_default(self):
        self.assertEqual(self.tool.baseline_dt, 2.0)

    def test_template_default(self):
        self.assertIsNone(self.tool.template)

    def test_criterion_threshold_default(self):
        self.assertEqual(self.tool.criterion_threshold, 4.0)

    def test_onset_search_default(self):
        self.assertTrue(self.tool.onset_search)

    def test_onset_avg_default(self):
        self.assertEqual(self.tool.onset_avg, 1.0)

    def test_onset_nstdv_default(self):
        self.assertEqual(self.tool.onset_nstdv, 1.0)

    def test_onset_limit_default(self):
        self.assertEqual(self.tool.onset_limit, 2.0)

    def test_peak_search_default(self):
        self.assertTrue(self.tool.peak_search)

    def test_peak_avg_default(self):
        self.assertEqual(self.tool.peak_avg, 1.0)

    def test_peak_nstdv_default(self):
        self.assertEqual(self.tool.peak_nstdv, 1.0)

    def test_peak_limit_default(self):
        self.assertEqual(self.tool.peak_limit, 4.0)

    def test_refractory_default(self):
        self.assertEqual(self.tool.refractory, 0.0)

    def test_max_events_default(self):
        self.assertEqual(self.tool.max_events, 0)

    def test_template_baseline_default(self):
        self.assertEqual(self.tool.template_baseline, 0.0)

    def test_results_to_history_default(self):
        self.assertFalse(self.tool.results_to_history)

    def test_results_to_cache_default(self):
        self.assertTrue(self.tool.results_to_cache)

    def test_results_to_numpy_default(self):
        self.assertTrue(self.tool.results_to_numpy)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestNMToolEventProperties(unittest.TestCase):

    def setUp(self):
        self.tool = NMToolEvent()

    # algorithm
    def test_algorithm_accepts_valid(self):
        for alg in ("threshold", "nstdv", "template"):
            self.tool.algorithm = alg
            self.assertEqual(self.tool.algorithm, alg)

    def test_algorithm_rejects_invalid(self):
        with self.assertRaises(ValueError):
            self.tool.algorithm = "peak"

    def test_algorithm_rejects_non_string(self):
        with self.assertRaises(TypeError):
            self.tool.algorithm = 1

    # polarity
    def test_polarity_accepts_valid(self):
        for p in ("negative", "positive"):
            self.tool.polarity = p
            self.assertEqual(self.tool.polarity, p)

    def test_polarity_rejects_invalid(self):
        with self.assertRaises(ValueError):
            self.tool.polarity = "up"

    def test_polarity_rejects_non_string(self):
        with self.assertRaises(TypeError):
            self.tool.polarity = True

    # threshold
    def test_threshold_accepts_float(self):
        self.tool.threshold = 30.5
        self.assertAlmostEqual(self.tool.threshold, 30.5)

    def test_threshold_accepts_zero(self):
        self.tool.threshold = 0.0
        self.assertEqual(self.tool.threshold, 0.0)

    def test_threshold_rejects_negative(self):
        with self.assertRaises(ValueError):
            self.tool.threshold = -1.0

    def test_threshold_rejects_bool(self):
        with self.assertRaises(TypeError):
            self.tool.threshold = True

    # nstdv
    def test_nstdv_rejects_negative(self):
        with self.assertRaises(ValueError):
            self.tool.nstdv = -0.5

    # baseline_avg
    def test_baseline_avg_accepts_zero(self):
        self.tool.baseline_avg = 0.0
        self.assertEqual(self.tool.baseline_avg, 0.0)

    def test_baseline_avg_rejects_negative(self):
        with self.assertRaises(ValueError):
            self.tool.baseline_avg = -1.0

    # baseline_dt
    def test_baseline_dt_rejects_zero(self):
        with self.assertRaises(ValueError):
            self.tool.baseline_dt = 0.0

    def test_baseline_dt_rejects_negative(self):
        with self.assertRaises(ValueError):
            self.tool.baseline_dt = -1.0

    # template
    def test_template_accepts_1d_array(self):
        arr = np.array([0.0, 0.5, 1.0, 0.5])
        self.tool.template = arr
        np.testing.assert_array_equal(self.tool.template, arr)

    def test_template_accepts_none(self):
        self.tool.template = None
        self.assertIsNone(self.tool.template)

    def test_template_rejects_2d(self):
        with self.assertRaises(ValueError):
            self.tool.template = np.zeros((3, 4))

    def test_template_rejects_non_array(self):
        with self.assertRaises(TypeError):
            self.tool.template = [1.0, 2.0, 3.0]

    # criterion_threshold
    def test_criterion_threshold_rejects_zero(self):
        with self.assertRaises(ValueError):
            self.tool.criterion_threshold = 0.0

    def test_criterion_threshold_rejects_negative(self):
        with self.assertRaises(ValueError):
            self.tool.criterion_threshold = -1.0

    # onset_search / peak_search
    def test_onset_search_accepts_bool(self):
        self.tool.onset_search = False
        self.assertFalse(self.tool.onset_search)

    def test_onset_search_rejects_int(self):
        with self.assertRaises(TypeError):
            self.tool.onset_search = 0

    def test_peak_search_accepts_bool(self):
        self.tool.peak_search = False
        self.assertFalse(self.tool.peak_search)

    def test_peak_search_rejects_int(self):
        with self.assertRaises(TypeError):
            self.tool.peak_search = 1

    # onset_limit / peak_limit — must be > 0
    def test_onset_limit_rejects_zero(self):
        with self.assertRaises(ValueError):
            self.tool.onset_limit = 0.0

    def test_peak_limit_rejects_zero(self):
        with self.assertRaises(ValueError):
            self.tool.peak_limit = 0.0

    # refractory
    def test_refractory_accepts_zero(self):
        self.tool.refractory = 0.0
        self.assertEqual(self.tool.refractory, 0.0)

    def test_refractory_rejects_negative(self):
        with self.assertRaises(ValueError):
            self.tool.refractory = -1.0

    # template_baseline
    def test_template_baseline_accepts_zero(self):
        self.tool.template_baseline = 0.0
        self.assertEqual(self.tool.template_baseline, 0.0)

    def test_template_baseline_accepts_positive(self):
        self.tool.template_baseline = 5e-3
        self.assertAlmostEqual(self.tool.template_baseline, 5e-3)

    def test_template_baseline_rejects_negative(self):
        with self.assertRaises(ValueError):
            self.tool.template_baseline = -1e-3

    def test_template_baseline_rejects_bool(self):
        with self.assertRaises(TypeError):
            self.tool.template_baseline = True

# ---------------------------------------------------------------------------
# Detection: threshold algorithm
# ---------------------------------------------------------------------------


class TestNMToolEventDetectionThreshold(unittest.TestCase):
    """Threshold < Baseline algorithm detects events at known times."""

    def setUp(self):
        self.tool = NMToolEvent()
        self.tool.algorithm = "threshold"
        self.tool.polarity = "negative"
        self.tool.threshold = 30.0          # detect deflections > 30 pA below baseline
        self.tool.baseline_avg = 2e-3       # 2 ms baseline window
        self.tool.baseline_dt = 2e-3        # look 2 ms ahead
        self.tool.onset_search = False
        self.tool.peak_search = False
        # place 3 events well separated in time
        self._event_times_ms = [20.0, 50.0, 80.0]
        self._data = _make_epsc_data(
            event_times_ms=self._event_times_ms,
            amplitude=-100.0,
            decay_ms=3.0,
        )

    def test_detects_correct_number_of_events(self):
        folder = _run(self.tool, [self._data])
        tf = folder.toolfolders.get("Event_0") or list(folder.toolfolders.values())[0]
        ev = tf.data.get("EV_recordA0")
        self.assertEqual(len(ev.nparray), 3)

    def test_detected_times_near_event_times(self):
        folder = _run(self.tool, [self._data])
        tf = list(folder.toolfolders.values())[0]
        ev = tf.data.get("EV_recordA0")
        times_ms = ev.nparray / _MS
        for expected_ms, detected_ms in zip(sorted(self._event_times_ms), sorted(times_ms)):
            # Detection point is at t0 + baseline_dt; allow ±5 ms tolerance
            self.assertAlmostEqual(detected_ms, expected_ms + 2.0, delta=5.0)

    def test_no_events_on_flat_baseline(self):
        flat = _make_epsc_data(name="flat", event_times_ms=[])
        folder = _run(self.tool, [flat])
        tf = list(folder.toolfolders.values())[0]
        ev = tf.data.get("EV_flat")
        self.assertEqual(len(ev.nparray), 0)

    def test_reject_count_zero_without_onset_peak_search(self):
        folder = _run(self.tool, [self._data])
        tf = list(folder.toolfolders.values())[0]
        rc = tf.data.get("EV_reject_count")
        self.assertEqual(rc.nparray[0], 0)


# ---------------------------------------------------------------------------
# Detection: nstdv algorithm
# ---------------------------------------------------------------------------


class TestNMToolEventDetectionNstdv(unittest.TestCase):
    """Nstdv < Baseline algorithm detects large events."""

    def setUp(self):
        self.tool = NMToolEvent()
        self.tool.algorithm = "nstdv"
        self.tool.polarity = "negative"
        self.tool.nstdv = 3.0
        self.tool.baseline_avg = 5e-3   # 5 ms
        self.tool.baseline_dt = 2e-3
        self.tool.onset_search = False
        self.tool.peak_search = False

    def test_detects_large_events(self):
        # baseline_avg=2ms < 2*baseline_dt=4ms keeps window clear of detection
        # point (no event contamination). nstdv=5 suppresses false positives
        # (P≈3e-7/sample). Refractory prevents re-detection on decaying tail.
        self.tool.baseline_avg = 2e-3
        self.tool.nstdv = 5.0
        self.tool.refractory = 15e-3   # 15 ms >> 3 ms decay τ
        data = _make_epsc_data(
            event_times_ms=[30.0, 70.0],
            amplitude=-200.0,
            decay_ms=3.0,
            noise_std=5.0,
        )
        folder = _run(self.tool, [data])
        tf = list(folder.toolfolders.values())[0]
        ev = tf.data.get("EV_recordA0")
        self.assertEqual(len(ev.nparray), 2)

    def test_nstdv_requires_baseline_avg_gt_zero(self):
        self.tool.baseline_avg = 0.0
        data = _make_epsc_data(event_times_ms=[30.0], amplitude=-200.0)
        with self.assertRaises(ValueError):
            _run(self.tool, [data])

    def test_ignores_small_fluctuations(self):
        # Pure noise well below 3 std threshold — expect few or no events
        rng = np.random.default_rng(42)
        y = rng.normal(0.0, 1.0, 10000)
        data = NMData(NM, name="noisy", nparray=y,
                      xscale={"start": 0.0, "delta": _DELTA})
        self.tool.nstdv = 10.0  # very high threshold
        folder = _run(self.tool, [data])
        tf = list(folder.toolfolders.values())[0]
        ev = tf.data.get("EV_noisy")
        self.assertEqual(len(ev.nparray), 0)


# ---------------------------------------------------------------------------
# Detection: template algorithm
# ---------------------------------------------------------------------------


class TestNMToolEventDetectionTemplate(unittest.TestCase):
    """Template matching stores EV_template and EV_Match_ arrays."""

    def setUp(self):
        self.tool = NMToolEvent()
        self.tool.algorithm = "template"
        self.tool.polarity = "negative"
        self.tool.criterion_threshold = 4.0   # Clements & Bekkers 1997 typical value
        self.tool.onset_search = False
        self.tool.peak_search = False
        # Template: positive unit EPSC shape (peak at t=0, exponential decay).
        # The OLS scale will be negative when matched against negative EPSC data,
        # giving a negative criterion that crosses -criterion_threshold at event onset.
        t_tpl = np.arange(200) * _DELTA
        tpl = np.exp(-t_tpl / (5e-3))   # positive: 1→0 over 20 ms
        self.tool.template = tpl.astype(float)

    def test_run_init_raises_without_template(self):
        self.tool.template = None
        folder = NMFolder(NM, name="TestFolder")
        data = _make_epsc_data(event_times_ms=[30.0], amplitude=-100.0)
        with self.assertRaises(ValueError):
            self.tool.run_all(_make_targets([data], folder))

    def test_ev_template_stored_normalized(self):
        data = _make_epsc_data(event_times_ms=[30.0, 70.0], amplitude=-150.0)
        folder = _run(self.tool, [data])
        tf = list(folder.toolfolders.values())[0]
        ev_tpl = tf.data.get("EV_template")
        self.assertIsNotNone(ev_tpl)
        arr = ev_tpl.nparray
        # Normalized: min=0, max=1
        self.assertAlmostEqual(float(arr.min()), 0.0, places=6)
        self.assertAlmostEqual(float(arr.max()), 1.0, places=6)

    def test_ev_matchtmplt_stored_per_epoch(self):
        data = _make_epsc_data(
            name="recordA0",
            event_times_ms=[30.0, 70.0],
            amplitude=-200.0,
        )
        folder = _run(self.tool, [data])
        tf = list(folder.toolfolders.values())[0]
        crit = tf.data.get("EV_Match_recordA0")
        self.assertIsNotNone(crit)
        self.assertEqual(len(crit.nparray), len(data.nparray))

    def test_template_algorithm_detects_events(self):
        data = _make_epsc_data(
            event_times_ms=[30.0, 70.0],
            amplitude=-500.0,
            decay_ms=5.0,
            noise_std=5.0,
        )
        folder = _run(self.tool, [data])
        tf = list(folder.toolfolders.values())[0]
        ev = tf.data.get("EV_recordA0")
        self.assertEqual(len(ev.nparray), 2)

    def test_template_baseline_shifts_detected_times(self):
        # Event at 50 ms. With template_baseline=10 ms, the criterion crossing
        # occurs ~10 ms early; the shift should recover the true event onset.
        # Without the shift, detected time ≈ 40 ms; with shift ≈ 50 ms.
        # Refractory prevents re-detections on the decaying tail.
        event_ms = 50.0
        baseline_ms = 10.0
        self.tool.template_baseline = baseline_ms * _MS
        self.tool.refractory = 20e-3
        data = _make_epsc_data(
            event_times_ms=[event_ms],
            amplitude=-500.0,
            decay_ms=5.0,
            noise_std=5.0,
        )
        folder = _run(self.tool, [data])
        tf = list(folder.toolfolders.values())[0]
        ev = tf.data.get("EV_recordA0")
        self.assertEqual(len(ev.nparray), 1)
        # Detected time should be within 2 ms of the true event onset
        detected_ms = ev.nparray[0] * 1000.0
        self.assertAlmostEqual(detected_ms, event_ms, delta=2.0)

    def test_template_baseline_zero_no_shift(self):
        # With template_baseline=0, detected time is also near the event onset
        # (criterion peaks at event start when no baseline is prepended).
        # Refractory prevents re-detections on the decaying tail.
        event_ms = 50.0
        self.tool.template_baseline = 0.0
        self.tool.refractory = 20e-3
        data = _make_epsc_data(
            event_times_ms=[event_ms],
            amplitude=-500.0,
            decay_ms=5.0,
            noise_std=5.0,
        )
        folder = _run(self.tool, [data])
        tf = list(folder.toolfolders.values())[0]
        ev = tf.data.get("EV_recordA0")
        self.assertEqual(len(ev.nparray), 1)
        detected_ms = ev.nparray[0] * 1000.0
        self.assertAlmostEqual(detected_ms, event_ms, delta=2.0)


# ---------------------------------------------------------------------------
# Onset and peak search
# ---------------------------------------------------------------------------


class TestNMToolEventOnsetSearch(unittest.TestCase):
    """Onset search: failed onset → event moved to EV_reject_."""

    def setUp(self):
        self.tool = NMToolEvent()
        self.tool.algorithm = "threshold"
        self.tool.polarity = "negative"
        self.tool.threshold = 30.0
        self.tool.baseline_avg = 2e-3
        self.tool.baseline_dt = 2e-3
        self.tool.onset_search = True
        self.tool.onset_avg = 1e-3
        self.tool.onset_nstdv = 1.0
        self.tool.onset_limit = 10e-3   # 10 ms backward
        self.tool.peak_search = False

    def test_onset_found_for_clear_events(self):
        # onset_nstdv algorithm needs baseline noise so Y_stdv > 0
        data = _make_epsc_data(
            event_times_ms=[30.0],
            amplitude=-100.0,
            decay_ms=5.0,
            noise_std=3.0,
        )
        folder = _run(self.tool, [data])
        tf = list(folder.toolfolders.values())[0]
        ev = tf.data.get("EV_recordA0")
        self.assertGreater(len(ev.nparray), 0)
        # onset array should exist
        on = tf.data.get("EV_onset_recordA0")
        self.assertIsNotNone(on)

    def test_onset_failure_leads_to_rejection(self):
        # Very tight onset limit that cannot reach the event onset
        self.tool.onset_limit = 1e-5   # ~0.01 ms — too short
        data = _make_epsc_data(
            event_times_ms=[30.0],
            amplitude=-100.0,
            decay_ms=5.0,
        )
        folder = _run(self.tool, [data])
        tf = list(folder.toolfolders.values())[0]
        rej = tf.data.get("EV_reject_recordA0")
        rc = tf.data.get("EV_reject_count")
        self.assertGreater(len(rej.nparray), 0)
        self.assertGreater(rc.nparray[0], 0)


class TestNMToolEventPeakSearch(unittest.TestCase):
    """Peak search: failed peak → event moved to EV_reject_."""

    def setUp(self):
        self.tool = NMToolEvent()
        self.tool.algorithm = "threshold"
        self.tool.polarity = "negative"
        self.tool.threshold = 30.0
        self.tool.baseline_avg = 2e-3
        self.tool.baseline_dt = 2e-3
        self.tool.onset_search = False
        self.tool.peak_search = True
        self.tool.peak_avg = 1e-3
        self.tool.peak_nstdv = 1.0
        self.tool.peak_limit = 20e-3   # 20 ms forward

    def test_peak_found_for_clear_events(self):
        # peak_nstdv algorithm needs baseline noise so Y_stdv > 0
        data = _make_epsc_data(
            event_times_ms=[30.0],
            amplitude=-100.0,
            decay_ms=5.0,
            noise_std=3.0,
        )
        folder = _run(self.tool, [data])
        tf = list(folder.toolfolders.values())[0]
        ev = tf.data.get("EV_recordA0")
        self.assertGreater(len(ev.nparray), 0)
        pk = tf.data.get("EV_peak_recordA0")
        self.assertIsNotNone(pk)

    def test_peak_failure_leads_to_rejection(self):
        # peak_avg=0 → 1-sample window → Y_std=0 → leftmost never < Y_avg=leftmost
        self.tool.peak_avg = 0.0
        data = _make_epsc_data(
            event_times_ms=[30.0],
            amplitude=-100.0,
            decay_ms=5.0,
        )
        folder = _run(self.tool, [data])
        tf = list(folder.toolfolders.values())[0]
        rej = tf.data.get("EV_reject_recordA0")
        self.assertGreater(len(rej.nparray), 0)


# ---------------------------------------------------------------------------
# Output arrays
# ---------------------------------------------------------------------------


class TestNMToolEventOutputArrays(unittest.TestCase):
    """Toolfolder naming and EV_ array structure."""

    def setUp(self):
        self.tool = NMToolEvent()
        self.tool.algorithm = "threshold"
        self.tool.polarity = "negative"
        self.tool.threshold = 30.0
        self.tool.baseline_avg = 2e-3
        self.tool.baseline_dt = 2e-3
        self.tool.onset_search = False
        self.tool.peak_search = False

    def test_toolfolder_created(self):
        data = _make_epsc_data(event_times_ms=[30.0], amplitude=-100.0)
        folder = _run(self.tool, [data])
        self.assertGreater(len(list(folder.toolfolders.values())), 0)

    def test_ev_epoch_names_array(self):
        d0 = _make_epsc_data(name="recordA0", event_times_ms=[30.0], amplitude=-100.0)
        d1 = _make_epsc_data(name="recordA1", event_times_ms=[50.0], amplitude=-100.0)
        folder = _run(self.tool, [d0, d1])
        tf = list(folder.toolfolders.values())[0]
        names = tf.data.get("EV_epoch_names")
        self.assertIsNotNone(names)
        self.assertIn("recordA0", list(names.nparray))
        self.assertIn("recordA1", list(names.nparray))

    def test_ev_count_array_length_matches_epochs(self):
        d0 = _make_epsc_data(name="recordA0", event_times_ms=[30.0], amplitude=-100.0)
        d1 = _make_epsc_data(name="recordA1", event_times_ms=[50.0], amplitude=-100.0)
        folder = _run(self.tool, [d0, d1])
        tf = list(folder.toolfolders.values())[0]
        cnt = tf.data.get("EV_count")
        self.assertEqual(len(cnt.nparray), 2)

    def test_ev_reject_array_exists_per_epoch(self):
        data = _make_epsc_data(name="recordA0", event_times_ms=[], amplitude=-100.0)
        folder = _run(self.tool, [data])
        tf = list(folder.toolfolders.values())[0]
        rej = tf.data.get("EV_reject_recordA0")
        self.assertIsNotNone(rej)

    def test_onset_arrays_absent_when_onset_search_false(self):
        data = _make_epsc_data(name="recordA0", event_times_ms=[30.0], amplitude=-100.0)
        folder = _run(self.tool, [data])
        tf = list(folder.toolfolders.values())[0]
        self.assertIsNone(tf.data.get("EV_onset_recordA0"))

    def test_onset_arrays_present_when_onset_search_true(self):
        self.tool.onset_search = True
        self.tool.onset_limit = 20e-3
        data = _make_epsc_data(name="recordA0", event_times_ms=[30.0], amplitude=-100.0)
        folder = _run(self.tool, [data])
        tf = list(folder.toolfolders.values())[0]
        # Onset array should exist regardless of whether onset was found
        on = tf.data.get("EV_onset_recordA0")
        self.assertIsNotNone(on)

    def test_peak_arrays_present_when_peak_search_true(self):
        self.tool.peak_search = True
        self.tool.peak_limit = 20e-3
        data = _make_epsc_data(name="recordA0", event_times_ms=[30.0], amplitude=-100.0)
        folder = _run(self.tool, [data])
        tf = list(folder.toolfolders.values())[0]
        pk = tf.data.get("EV_peak_recordA0")
        self.assertIsNotNone(pk)

    def test_ev_array_has_note(self):
        data = _make_epsc_data(name="recordA0", event_times_ms=[30.0], amplitude=-100.0)
        folder = _run(self.tool, [data])
        tf = list(folder.toolfolders.values())[0]
        ev = tf.data.get("EV_recordA0")
        self.assertGreater(len(ev.notes), 0)

    def test_ev_note_contains_algorithm_and_polarity(self):
        data = _make_epsc_data(name="recordA0", event_times_ms=[30.0], amplitude=-100.0)
        folder = _run(self.tool, [data])
        tf = list(folder.toolfolders.values())[0]
        ev = tf.data.get("EV_recordA0")
        note_text = " ".join(str(n) for n in ev.notes)
        self.assertIn("threshold", note_text)
        self.assertIn("negative", note_text)

    def test_ev_reject_array_has_note(self):
        data = _make_epsc_data(name="recordA0", event_times_ms=[], amplitude=-100.0)
        folder = _run(self.tool, [data])
        tf = list(folder.toolfolders.values())[0]
        rej = tf.data.get("EV_reject_recordA0")
        self.assertGreater(len(rej.notes), 0)

    def test_ev_count_array_has_note(self):
        data = _make_epsc_data(name="recordA0", event_times_ms=[30.0], amplitude=-100.0)
        folder = _run(self.tool, [data])
        tf = list(folder.toolfolders.values())[0]
        cnt = tf.data.get("EV_count")
        self.assertGreater(len(cnt.notes), 0)

    def test_onset_array_has_note(self):
        self.tool.onset_search = True
        self.tool.onset_limit = 20e-3
        data = _make_epsc_data(name="recordA0", event_times_ms=[30.0], amplitude=-100.0)
        folder = _run(self.tool, [data])
        tf = list(folder.toolfolders.values())[0]
        on = tf.data.get("EV_onset_recordA0")
        self.assertGreater(len(on.notes), 0)

    def test_peak_array_has_note(self):
        self.tool.peak_search = True
        self.tool.peak_limit = 20e-3
        data = _make_epsc_data(name="recordA0", event_times_ms=[30.0], amplitude=-100.0)
        folder = _run(self.tool, [data])
        tf = list(folder.toolfolders.values())[0]
        pk = tf.data.get("EV_peak_recordA0")
        self.assertGreater(len(pk.notes), 0)


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------


class TestNMToolEventCounts(unittest.TestCase):

    def setUp(self):
        self.tool = NMToolEvent()
        self.tool.algorithm = "threshold"
        self.tool.polarity = "negative"
        self.tool.threshold = 30.0
        self.tool.baseline_avg = 2e-3
        self.tool.baseline_dt = 2e-3
        self.tool.onset_search = False
        self.tool.peak_search = False

    def test_ev_count_matches_detected(self):
        data = _make_epsc_data(
            event_times_ms=[20.0, 50.0, 80.0],
            amplitude=-100.0,
            decay_ms=3.0,
        )
        folder = _run(self.tool, [data])
        tf = list(folder.toolfolders.values())[0]
        ev = tf.data.get("EV_recordA0")
        cnt = tf.data.get("EV_count")
        self.assertEqual(cnt.nparray[0], len(ev.nparray))

    def test_reject_count_matches_rejected(self):
        self.tool.onset_search = True
        self.tool.onset_limit = 1e-5   # force rejection
        data = _make_epsc_data(
            event_times_ms=[20.0, 50.0],
            amplitude=-100.0,
            decay_ms=3.0,
        )
        folder = _run(self.tool, [data])
        tf = list(folder.toolfolders.values())[0]
        rej = tf.data.get("EV_reject_recordA0")
        rc = tf.data.get("EV_reject_count")
        self.assertEqual(rc.nparray[0], len(rej.nparray))


# ---------------------------------------------------------------------------
# Polarity
# ---------------------------------------------------------------------------


class TestNMToolEventPolarity(unittest.TestCase):

    def test_positive_polarity_detects_upward_events(self):
        tool = NMToolEvent()
        tool.algorithm = "threshold"
        tool.polarity = "positive"
        tool.threshold = 30.0
        tool.baseline_avg = 2e-3
        tool.baseline_dt = 2e-3
        tool.onset_search = False
        tool.peak_search = False
        data = _make_epsc_data(
            event_times_ms=[30.0],
            amplitude=100.0,
            decay_ms=5.0,
        )
        folder = _run(tool, [data])
        tf = list(folder.toolfolders.values())[0]
        ev = tf.data.get("EV_recordA0")
        self.assertGreater(len(ev.nparray), 0)

    def test_positive_polarity_misses_negative_events(self):
        tool = NMToolEvent()
        tool.algorithm = "threshold"
        tool.polarity = "positive"
        tool.threshold = 30.0
        tool.baseline_avg = 2e-3
        tool.baseline_dt = 2e-3
        tool.onset_search = False
        tool.peak_search = False
        data = _make_epsc_data(
            event_times_ms=[30.0],
            amplitude=-100.0,   # negative event
            decay_ms=5.0,
        )
        folder = _run(tool, [data])
        tf = list(folder.toolfolders.values())[0]
        ev = tf.data.get("EV_recordA0")
        self.assertEqual(len(ev.nparray), 0)


# ---------------------------------------------------------------------------
# Search limits (xbgn / xend)
# ---------------------------------------------------------------------------


class TestNMToolEventSearchLimits(unittest.TestCase):

    def setUp(self):
        self.tool = NMToolEvent()
        self.tool.algorithm = "threshold"
        self.tool.polarity = "negative"
        self.tool.threshold = 30.0
        self.tool.baseline_avg = 2e-3
        self.tool.baseline_dt = 2e-3
        self.tool.onset_search = False
        self.tool.peak_search = False
        # Events at 20 ms and 80 ms
        self._data = _make_epsc_data(
            event_times_ms=[20.0, 80.0],
            amplitude=-100.0,
            decay_ms=3.0,
        )

    def test_xbgn_excludes_early_event(self):
        self.tool.xbgn = 50e-3   # start at 50 ms — misses 20 ms event
        folder = _run(self.tool, [self._data])
        tf = list(folder.toolfolders.values())[0]
        ev = tf.data.get("EV_recordA0")
        # Should only detect the 80 ms event
        self.assertEqual(len(ev.nparray), 1)
        self.assertGreater(ev.nparray[0], 50e-3)

    def test_xend_excludes_late_event(self):
        self.tool.xend = 50e-3   # end at 50 ms — misses 80 ms event
        folder = _run(self.tool, [self._data])
        tf = list(folder.toolfolders.values())[0]
        ev = tf.data.get("EV_recordA0")
        self.assertEqual(len(ev.nparray), 1)
        self.assertLess(ev.nparray[0], 50e-3)


# ---------------------------------------------------------------------------
# Refractory
# ---------------------------------------------------------------------------


class TestNMToolEventRefractory(unittest.TestCase):

    def setUp(self):
        self.tool = NMToolEvent()
        self.tool.algorithm = "threshold"
        self.tool.polarity = "negative"
        self.tool.threshold = 30.0
        self.tool.baseline_avg = 2e-3
        self.tool.baseline_dt = 2e-3
        self.tool.onset_search = False
        self.tool.peak_search = False

    def test_refractory_suppresses_nearby_events(self):
        # Two events only 10 ms apart — a 50 ms refractory should suppress the second
        data = _make_epsc_data(
            event_times_ms=[20.0, 30.0],
            amplitude=-100.0,
            decay_ms=2.0,
        )
        self.tool.refractory = 50e-3   # 50 ms
        folder = _run(self.tool, [data])
        tf = list(folder.toolfolders.values())[0]
        ev = tf.data.get("EV_recordA0")
        self.assertEqual(len(ev.nparray), 1)

    def test_no_refractory_allows_nearby_events(self):
        # decay_ms=2 ms: event tail is <2% of peak by 30 ms, so second event
        # is cleanly detected. No noise — expect exactly 2 detections.
        data = _make_epsc_data(
            event_times_ms=[20.0, 30.0],
            amplitude=-100.0,
            decay_ms=2.0,
        )
        self.tool.refractory = 0.0
        folder = _run(self.tool, [data])
        tf = list(folder.toolfolders.values())[0]
        ev = tf.data.get("EV_recordA0")
        self.assertEqual(len(ev.nparray), 2)


# ---------------------------------------------------------------------------
# max_events
# ---------------------------------------------------------------------------


class TestNMToolEventMaxEvents(unittest.TestCase):

    def setUp(self):
        self.tool = NMToolEvent()
        self.tool.algorithm = "threshold"
        self.tool.polarity = "negative"
        self.tool.threshold = 30.0
        self.tool.baseline_avg = 2e-3
        self.tool.baseline_dt = 2e-3
        self.tool.onset_search = False
        self.tool.peak_search = False

    def test_max_events_default(self):
        self.assertEqual(self.tool.max_events, 0)

    def test_max_events_set(self):
        self.tool.max_events = 2
        self.assertEqual(self.tool.max_events, 2)

    def test_max_events_zero_allowed(self):
        self.tool.max_events = 0
        self.assertEqual(self.tool.max_events, 0)

    def test_max_events_rejects_negative(self):
        with self.assertRaises(ValueError):
            self.tool.max_events = -1

    def test_max_events_rejects_bool(self):
        with self.assertRaises(TypeError):
            self.tool.max_events = True

    def test_max_events_rejects_float(self):
        with self.assertRaises(TypeError):
            self.tool.max_events = 1.0

    def test_max_events_limits_detection(self):
        data = _make_epsc_data(
            event_times_ms=[10.0, 30.0, 50.0, 70.0],
            amplitude=-100.0,
        )
        self.tool.max_events = 2
        folder = _run(self.tool, [data])
        tf = list(folder.toolfolders.values())[0]
        ev = tf.data.get("EV_recordA0")
        self.assertEqual(len(ev.nparray), 2)

    def test_max_events_zero_finds_all(self):
        data = _make_epsc_data(
            event_times_ms=[10.0, 30.0, 50.0, 70.0],
            amplitude=-100.0,
        )
        self.tool.max_events = 0
        folder = _run(self.tool, [data])
        tf = list(folder.toolfolders.values())[0]
        ev = tf.data.get("EV_recordA0")
        self.assertEqual(len(ev.nparray), 4)


# ---------------------------------------------------------------------------
# find_next_event
# ---------------------------------------------------------------------------


class TestNMToolEventFindNext(unittest.TestCase):

    def setUp(self):
        self.tool = NMToolEvent()
        self.tool.algorithm = "threshold"
        self.tool.polarity = "negative"
        self.tool.threshold = 30.0
        self.tool.baseline_avg = 2e-3
        self.tool.baseline_dt = 2e-3
        self.tool.onset_search = False
        self.tool.peak_search = False

    def test_find_next_returns_dict(self):
        data = _make_epsc_data(event_times_ms=[30.0], amplitude=-100.0)
        result = self.tool.find_next_event(data, start_x=0.0)
        self.assertIsNotNone(result)
        self.assertIn("t_event", result)
        self.assertIn("accepted", result)

    def test_find_next_returns_none_when_no_event(self):
        data = _make_epsc_data(event_times_ms=[], amplitude=-100.0)
        result = self.tool.find_next_event(data, start_x=0.0)
        self.assertIsNone(result)

    def test_find_next_returns_none_past_xend(self):
        data = _make_epsc_data(event_times_ms=[80.0], amplitude=-100.0)
        result = self.tool.find_next_event(data, start_x=0.0)
        self.assertIsNotNone(result)
        # But if we start past the event there should be nothing
        result2 = self.tool.find_next_event(data, start_x=0.09)  # 90 ms
        self.assertIsNone(result2)

    def test_find_next_accepted_true_when_no_onset_peak_required(self):
        data = _make_epsc_data(event_times_ms=[30.0], amplitude=-100.0)
        result = self.tool.find_next_event(data, start_x=0.0)
        self.assertIsNotNone(result)
        self.assertTrue(result["accepted"])


# ---------------------------------------------------------------------------
# Results to cache
# ---------------------------------------------------------------------------


class TestNMToolEventResultsToCache(unittest.TestCase):

    def test_results_saved_to_cache(self):
        tool = NMToolEvent()
        tool.algorithm = "threshold"
        tool.polarity = "negative"
        tool.threshold = 30.0
        tool.baseline_avg = 2e-3
        tool.baseline_dt = 2e-3
        tool.onset_search = False
        tool.peak_search = False
        tool.results_to_cache = True

        data = _make_epsc_data(event_times_ms=[30.0], amplitude=-100.0)
        folder = _run(tool, [data])
        self.assertIn("event", folder.toolresults)
        cached = folder.toolresults["event"][0]["results"]
        self.assertIn("recordA0", cached)


# ---------------------------------------------------------------------------
# Overwrite
# ---------------------------------------------------------------------------


class TestNMToolEventOverwrite(unittest.TestCase):

    def test_overwrite_true_reuses_toolfolder(self):
        tool = NMToolEvent()
        tool.algorithm = "threshold"
        tool.polarity = "negative"
        tool.threshold = 30.0
        tool.baseline_avg = 2e-3
        tool.baseline_dt = 2e-3
        tool.onset_search = False
        tool.peak_search = False
        tool.overwrite = True

        data = _make_epsc_data(event_times_ms=[30.0], amplitude=-100.0)
        folder = NMFolder(NM, name="TestFolder")
        _run(tool, [data], folder)
        n_before = len(list(folder.toolfolders.values()))
        _run(tool, [data], folder)
        n_after = len(list(folder.toolfolders.values()))
        self.assertEqual(n_before, n_after)

    def test_overwrite_false_creates_new_toolfolder(self):
        tool = NMToolEvent()
        tool.algorithm = "threshold"
        tool.polarity = "negative"
        tool.threshold = 30.0
        tool.baseline_avg = 2e-3
        tool.baseline_dt = 2e-3
        tool.onset_search = False
        tool.peak_search = False
        tool.overwrite = False

        data = _make_epsc_data(event_times_ms=[30.0], amplitude=-100.0)
        folder = NMFolder(NM, name="TestFolder")
        _run(tool, [data], folder)
        n_before = len(list(folder.toolfolders.values()))
        _run(tool, [data], folder)
        n_after = len(list(folder.toolfolders.values()))
        self.assertGreater(n_after, n_before)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestNMToolEventConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = NMToolEventConfig()
        self.assertEqual(cfg.algorithm, "threshold")
        self.assertEqual(cfg.polarity, "positive")
        self.assertEqual(cfg.threshold, 10.0)
        self.assertTrue(cfg.onset_search)
        self.assertTrue(cfg.peak_search)

    def test_schema_rejects_bad_algorithm(self):
        cfg = NMToolEventConfig()
        with self.assertRaises(ValueError):
            cfg.algorithm = "bad"

    def test_schema_rejects_bad_type(self):
        cfg = NMToolEventConfig()
        with self.assertRaises(TypeError):
            cfg.threshold = "big"

    def test_to_dict_round_trip(self):
        cfg = NMToolEventConfig()
        cfg.threshold = 50.0
        cfg.polarity = "positive"
        d = cfg.to_dict()
        cfg2 = NMToolEventConfig.from_dict(d)
        self.assertEqual(cfg2.threshold, 50.0)
        self.assertEqual(cfg2.polarity, "positive")

    def test_toml_type(self):
        self.assertEqual(NMToolEventConfig._TOML_TYPE, "event_config")


if __name__ == "__main__":
    unittest.main()
