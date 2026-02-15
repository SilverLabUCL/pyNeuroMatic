#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for ABF file reader.

Part of pyNeuroMatic.
"""
import unittest
from pathlib import Path

from pyneuromatic.core.nm_folder import NMFolder

FIXTURES_DIR = Path(__file__).parent / "fixtures"
ABF_FILE = FIXTURES_DIR / "15804044.abf"


class TestReadAbfErrors(unittest.TestCase):
    """Tests for error handling in read_abf."""

    def test_file_not_found(self):
        from pyneuromatic.io.abf import read_abf

        with self.assertRaises(FileNotFoundError):
            read_abf("nonexistent_file.abf")


@unittest.skipUnless(ABF_FILE.exists(), "ABF fixture file not available")
class TestReadAbfFile(unittest.TestCase):
    """Tests with ABF fixture file."""

    @classmethod
    def setUpClass(cls):
        from pyneuromatic.io.abf import read_abf

        cls.folder = read_abf(ABF_FILE)

    def test_returns_folder(self):
        self.assertIsInstance(self.folder, NMFolder)

    def test_folder_name(self):
        # Starts with digit, so prefixed with "F"
        self.assertEqual(self.folder.name, "F15804044")

    def test_data_count(self):
        # 10 sweeps x 2 channels = 20
        self.assertEqual(len(self.folder.data), 20)

    def test_data_names_channel_a(self):
        for i in range(10):
            self.assertIn(f"RecordA{i}", self.folder.data)

    def test_data_names_channel_b(self):
        for i in range(10):
            self.assertIn(f"RecordB{i}", self.folder.data)

    # x-axis scaling

    def test_x_delta(self):
        # 5000 Hz = 0.2 ms sample interval
        self.assertAlmostEqual(
            self.folder.data["RecordA0"].xscale.delta, 0.2, places=4
        )

    def test_x_start(self):
        self.assertAlmostEqual(
            self.folder.data["RecordA0"].xscale.start, 0.0, places=4
        )

    def test_x_units(self):
        self.assertEqual(self.folder.data["RecordA0"].xscale.units, "ms")

    # y-axis

    def test_channel_a_y_label(self):
        self.assertEqual(self.folder.data["RecordA0"].yscale.label, "Im_1stCh2")

    def test_channel_a_y_units(self):
        self.assertEqual(self.folder.data["RecordA0"].yscale.units, "pA")

    def test_channel_b_y_label(self):
        self.assertEqual(self.folder.data["RecordB0"].yscale.label, "Light")

    def test_channel_b_y_units(self):
        self.assertEqual(self.folder.data["RecordB0"].yscale.units, "V")

    # data shape

    def test_data_shape(self):
        self.assertEqual(
            self.folder.data["RecordA0"].nparray.shape, (5000,)
        )

    # metadata

    def test_metadata_root(self):
        self.assertIn("root", self.folder.metadata)

    def test_metadata_numwaves(self):
        self.assertEqual(self.folder.metadata["root"]["NumWaves"], 10)

    def test_metadata_numchannels(self):
        self.assertEqual(self.folder.metadata["root"]["NumChannels"], 2)

    def test_metadata_samplerate(self):
        self.assertEqual(self.folder.metadata["root"]["SampleRate"], 5000)

    def test_metadata_creator(self):
        self.assertIn("Clampex", self.folder.metadata["root"]["Creator"])

    # dataseries

    def test_dataseries_created(self):
        self.assertIn("Record", self.folder.dataseries)


@unittest.skipUnless(ABF_FILE.exists(), "ABF fixture file not available")
class TestReadAbfOptions(unittest.TestCase):
    """Tests for read_abf options."""

    def test_custom_prefix(self):
        from pyneuromatic.io.abf import read_abf

        folder = read_abf(ABF_FILE, prefix="Wave")
        self.assertIn("WaveA0", folder.data)
        self.assertEqual(len(folder.data), 20)

    def test_existing_folder(self):
        from pyneuromatic.io.abf import read_abf

        existing = NMFolder(name="MyFolder")
        folder = read_abf(ABF_FILE, folder=existing)
        self.assertIs(folder, existing)
        self.assertEqual(folder.name, "MyFolder")

    def test_make_dataseries_false(self):
        from pyneuromatic.io.abf import read_abf

        folder = read_abf(ABF_FILE, make_dataseries=False)
        self.assertEqual(len(folder.dataseries), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
