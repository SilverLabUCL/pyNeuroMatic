#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Axograph file reader.

Part of pyNeuroMatic.
"""
import tempfile
import unittest
from pathlib import Path

from pyneuromatic.io.axograph import read_axograph
from pyneuromatic.core.nm_folder import NMFolder


FIXTURES_DIR = Path(__file__).parent / "fixtures"
AXGD_FILE = FIXTURES_DIR / "Vers6_060523 004.axgd"


class TestReadAxographErrors(unittest.TestCase):
    """Tests for error handling in read_axograph."""

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            read_axograph("nonexistent_file.axgx")

    def test_invalid_format(self):
        with tempfile.NamedTemporaryFile(suffix=".axgx", delete=False) as f:
            f.write(b"XXXX")  # Invalid header
            f.flush()
            filepath = Path(f.name)

        try:
            with self.assertRaises(ValueError) as cm:
                read_axograph(filepath)
            self.assertIn("Unrecognized", str(cm.exception))
        finally:
            filepath.unlink()


@unittest.skipUnless(AXGD_FILE.exists(), "Fixture file not available")
class TestReadAxographXFile(unittest.TestCase):
    """Tests with Axograph X fixture file (version 6)."""

    @classmethod
    def setUpClass(cls):
        cls.folder = read_axograph(AXGD_FILE)

    def test_returns_folder(self):
        self.assertIsInstance(self.folder, NMFolder)

    def test_folder_name(self):
        self.assertEqual(self.folder.name, "Vers6_060523_004")

    def test_data_count(self):
        self.assertEqual(len(self.folder.data), 6)

    def test_data_names(self):
        expected = [
            "RecordA0", "RecordB0", "RecordC0",
            "RecordA1", "RecordB1", "RecordC1",
        ]
        self.assertEqual(list(self.folder.data.keys()), expected)

    # channel A: Membrane Voltage

    def test_channel_a_label(self):
        self.assertEqual(self.folder.data["RecordA0"].yscale.label,
                         "Membrane Voltage-1")

    def test_channel_a_units(self):
        self.assertEqual(self.folder.data["RecordA0"].yscale.units, "mV")

    def test_channel_a_shape_epoch0(self):
        self.assertEqual(self.folder.data["RecordA0"].nparray.shape, (79840,))

    def test_channel_a_shape_epoch1(self):
        self.assertEqual(self.folder.data["RecordA1"].nparray.shape, (20479,))

    # channel B: Command Current

    def test_channel_b_label(self):
        self.assertEqual(self.folder.data["RecordB0"].yscale.label,
                         "Raw Command Current-1")

    def test_channel_b_units(self):
        self.assertEqual(self.folder.data["RecordB0"].yscale.units, "pA")

    # channel C: Analog Input

    def test_channel_c_label(self):
        self.assertEqual(self.folder.data["RecordC0"].yscale.label,
                         "Analog Input #2")

    def test_channel_c_units(self):
        self.assertEqual(self.folder.data["RecordC0"].yscale.units, "mV")

    # x-axis (time)

    def test_x_units(self):
        self.assertEqual(self.folder.data["RecordA0"].xscale.units, "ms")

    def test_x_delta(self):
        self.assertAlmostEqual(self.folder.data["RecordA0"].xscale.delta,
                               0.0501, places=4)

    # dataseries

    def test_dataseries_created(self):
        self.assertIn("Record", self.folder.dataseries)


@unittest.skipUnless(AXGD_FILE.exists(), "Fixture file not available")
class TestReadAxographOptions(unittest.TestCase):
    """Tests for read_axograph options."""

    def test_custom_prefix(self):
        folder = read_axograph(AXGD_FILE, prefix="Wave")
        self.assertIn("WaveA0", folder.data)

    def test_existing_folder(self):
        existing = NMFolder(name="MyFolder")
        folder = read_axograph(AXGD_FILE, folder=existing)
        self.assertIs(folder, existing)
        self.assertEqual(folder.name, "MyFolder")

    def test_make_dataseries_false(self):
        folder = read_axograph(AXGD_FILE, make_dataseries=False)
        self.assertEqual(len(folder.dataseries), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
