#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for PXP file reader.

Part of pyNeuroMatic.
"""
import unittest
from pathlib import Path

from pyneuromatic.core.nm_folder import NMFolder

FIXTURES_DIR = Path(__file__).parent / "fixtures"
PXP_FILE = FIXTURES_DIR / "nm02Jul04c0_002.pxp"


class TestReadPxpErrors(unittest.TestCase):
    """Tests for error handling in read_pxp."""

    def test_file_not_found(self):
        from pyneuromatic.io.pxp import read_pxp

        with self.assertRaises(FileNotFoundError):
            read_pxp("nonexistent_file.pxp")


@unittest.skipUnless(PXP_FILE.exists(), "PXP fixture file not available")
class TestReadPxpFile(unittest.TestCase):
    """Tests with PXP fixture file."""

    @classmethod
    def setUpClass(cls):
        from pyneuromatic.io.pxp import read_pxp

        cls.folder = read_pxp(PXP_FILE)

    def test_returns_folder(self):
        self.assertIsInstance(self.folder, NMFolder)

    def test_folder_name(self):
        self.assertEqual(self.folder.name, "nm02Jul04c0_002")

    def test_data_count(self):
        # 19 epochs x 2 channels = 38 data waves
        self.assertEqual(len(self.folder.data), 38)

    def test_data_names_channel_a(self):
        for i in range(19):
            self.assertIn(f"RecordA{i}", self.folder.data)

    def test_data_names_channel_b(self):
        for i in range(19):
            self.assertIn(f"RecordB{i}", self.folder.data)

    # x-axis scaling

    def test_x_delta(self):
        self.assertAlmostEqual(
            self.folder.data["RecordA0"].xscale["delta"], 0.02, places=4
        )

    def test_x_start(self):
        self.assertAlmostEqual(
            self.folder.data["RecordA0"].xscale["start"], 0.0, places=4
        )

    def test_x_units(self):
        self.assertEqual(self.folder.data["RecordA0"].xscale["units"], "ms")

    # y-axis (from yLabel wave)

    def test_channel_a_y_label(self):
        self.assertEqual(self.folder.data["RecordA0"].yscale["label"], "Vmem")

    def test_channel_a_y_units(self):
        self.assertEqual(self.folder.data["RecordA0"].yscale["units"], "mV")

    def test_channel_b_y_label(self):
        self.assertEqual(self.folder.data["RecordB0"].yscale["label"], "Icmd")

    def test_channel_b_y_units(self):
        self.assertEqual(self.folder.data["RecordB0"].yscale["units"], "pA")

    # data shape

    def test_data_shape(self):
        self.assertEqual(
            self.folder.data["RecordA0"].nparray.shape, (60000,)
        )

    # metadata

    def test_metadata_root(self):
        self.assertIn("root", self.folder.metadata)

    def test_metadata_acqmode(self):
        self.assertEqual(
            self.folder.metadata["root"]["AcqMode"], "episodic"
        )

    def test_metadata_waveprefix(self):
        self.assertEqual(
            self.folder.metadata["root"]["WavePrefix"], "Record"
        )

    def test_metadata_numwaves(self):
        self.assertEqual(self.folder.metadata["root"]["NumWaves"], 19)

    def test_metadata_notes_folder(self):
        self.assertIn("Notes", self.folder.metadata)

    def test_metadata_notes_name(self):
        self.assertEqual(
            self.folder.metadata["Notes"]["H_Name"], "Jason Rothman"
        )

    def test_metadata_fistep_folder(self):
        self.assertIn("FIstep", self.folder.metadata)

    # dataseries

    def test_dataseries_created(self):
        self.assertIn("Record", self.folder.dataseries)


@unittest.skipUnless(PXP_FILE.exists(), "PXP fixture file not available")
class TestReadPxpOptions(unittest.TestCase):
    """Tests for read_pxp options."""

    def test_custom_prefix(self):
        from pyneuromatic.io.pxp import read_pxp

        folder = read_pxp(PXP_FILE, prefix="Wave")
        # Custom prefix doesn't match "Record" waves, so no data imported
        self.assertEqual(len(folder.data), 0)

    def test_matching_prefix(self):
        from pyneuromatic.io.pxp import read_pxp

        folder = read_pxp(PXP_FILE, prefix="Record")
        self.assertEqual(len(folder.data), 38)

    def test_existing_folder(self):
        from pyneuromatic.io.pxp import read_pxp

        existing = NMFolder(name="MyFolder")
        folder = read_pxp(PXP_FILE, folder=existing)
        self.assertIs(folder, existing)
        self.assertEqual(folder.name, "MyFolder")

    def test_make_dataseries_false(self):
        from pyneuromatic.io.pxp import read_pxp

        folder = read_pxp(PXP_FILE, make_dataseries=False)
        self.assertEqual(len(folder.dataseries), 0)

    def test_prefix_auto_detect(self):
        from pyneuromatic.io.pxp import read_pxp

        # prefix=None should auto-detect "Record" from WavePrefix
        folder = read_pxp(PXP_FILE, prefix=None)
        self.assertIn("RecordA0", folder.data)
        self.assertEqual(len(folder.data), 38)


if __name__ == "__main__":
    unittest.main(verbosity=2)
