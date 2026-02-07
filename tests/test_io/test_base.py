#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for io base utilities.

Part of pyNeuroMatic.
"""
import unittest

from pyneuromatic.io.base import (
    parse_units_from_label,
    make_data_name,
    ParsedUnits,
)


class TestParseUnitsFromLabel(unittest.TestCase):
    """Tests for parse_units_from_label function."""

    def test_simple_units(self):
        result = parse_units_from_label("Current (pA)")
        self.assertEqual(result.label, "Current")
        self.assertEqual(result.units, "pA")
        self.assertEqual(result.scale, 1.0)

    def test_voltage_conversion(self):
        result = parse_units_from_label("Voltage (V)")
        self.assertEqual(result.label, "Voltage")
        self.assertEqual(result.units, "mV")
        self.assertEqual(result.scale, 1e3)

    def test_current_conversion(self):
        result = parse_units_from_label("Current (A)")
        self.assertEqual(result.label, "Current")
        self.assertEqual(result.units, "pA")
        self.assertEqual(result.scale, 1e12)

    def test_time_conversion(self):
        result = parse_units_from_label("Time (s)")
        self.assertEqual(result.label, "Time")
        self.assertEqual(result.units, "ms")
        self.assertEqual(result.scale, 1e3)

    def test_no_conversion_needed(self):
        result = parse_units_from_label("Time (ms)")
        self.assertEqual(result.label, "Time")
        self.assertEqual(result.units, "ms")
        self.assertEqual(result.scale, 1.0)

    def test_no_units(self):
        result = parse_units_from_label("Data")
        self.assertEqual(result.label, "Data")
        self.assertEqual(result.units, "")
        self.assertEqual(result.scale, 1.0)

    def test_empty_string(self):
        result = parse_units_from_label("")
        self.assertEqual(result.label, "")
        self.assertEqual(result.units, "")
        self.assertEqual(result.scale, 1.0)

    def test_unknown_units(self):
        result = parse_units_from_label("Custom (xyz)")
        self.assertEqual(result.label, "Custom")
        self.assertEqual(result.units, "xyz")
        self.assertEqual(result.scale, 1.0)

    def test_whitespace_handling(self):
        result = parse_units_from_label("  Voltage  (  mV  )  ")
        self.assertEqual(result.label, "Voltage")
        self.assertEqual(result.units, "mV")

    def test_no_space_before_parens(self):
        result = parse_units_from_label("Current(pA)")
        self.assertEqual(result.label, "Current")
        self.assertEqual(result.units, "pA")


class TestMakeDataName(unittest.TestCase):
    """Tests for make_data_name function."""

    def test_channel_a(self):
        result = make_data_name("Record", 0, 0)
        self.assertEqual(result, "RecordA0")

    def test_channel_b(self):
        result = make_data_name("Record", 1, 0)
        self.assertEqual(result, "RecordB0")

    def test_different_epoch(self):
        result = make_data_name("Record", 0, 5)
        self.assertEqual(result, "RecordA5")

    def test_different_prefix(self):
        result = make_data_name("avg", 0, 0)
        self.assertEqual(result, "avgA0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
