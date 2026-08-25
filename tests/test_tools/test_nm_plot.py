"""Tests for pyneuromatic.tools.nm_plot."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from pyneuromatic.tools.nm_plot import plot_nmdata, plot_folder
from pyneuromatic.core.nm_folder import NMFolder


@pytest.fixture
def folder():
    f = NMFolder(name="Test")
    for chan in ("A", "B"):
        for epoch in range(3):
            arr = np.sin(np.linspace(0, 2 * np.pi, 100)) + epoch * 0.1
            f.data.new(
                "Record%s%d" % (chan, epoch),
                nparray=arr,
                xscale={"start": 0.0, "delta": 0.1, "label": "Time", "units": "ms"},
                yscale={"label": "Vm", "units": "mV"},
            )
    return f


# ---------------------------------------------------------------------------
# plot_nmdata
# ---------------------------------------------------------------------------

class TestPlotNMData:
    def test_returns_fig_and_ax(self, folder):
        nmdata = folder.data["RecordA0"]
        fig, ax = plot_nmdata(nmdata, show=False)
        assert fig is not None
        assert ax is not None

    def test_default_title_is_data_name(self, folder):
        nmdata = folder.data["RecordA0"]
        _, ax = plot_nmdata(nmdata, show=False)
        assert ax.get_title() == "RecordA0"

    def test_custom_title(self, folder):
        nmdata = folder.data["RecordA0"]
        _, ax = plot_nmdata(nmdata, title="My Title", show=False)
        assert ax.get_title() == "My Title"

    def test_xlabel_contains_label_and_units(self, folder):
        _, ax = plot_nmdata(folder.data["RecordA0"], show=False)
        assert "Time" in ax.get_xlabel()
        assert "ms" in ax.get_xlabel()

    def test_ylabel_contains_label_and_units(self, folder):
        _, ax = plot_nmdata(folder.data["RecordA0"], show=False)
        assert "Vm" in ax.get_ylabel()
        assert "mV" in ax.get_ylabel()

    def test_uses_supplied_ax(self, folder):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        nmdata = folder.data["RecordA0"]
        _, ax2 = plot_nmdata(nmdata, ax=ax, show=False)
        assert ax2 is ax

    def test_line_drawn(self, folder):
        _, ax = plot_nmdata(folder.data["RecordA0"], show=False)
        assert len(ax.lines) == 1

    def test_x_axis_built_from_xscale(self, folder):
        nmdata = folder.data["RecordA0"]
        _, ax = plot_nmdata(nmdata, show=False)
        xdata = ax.lines[0].get_xdata()
        assert xdata[0] == pytest.approx(nmdata.xscale.start)
        expected_end = nmdata.xscale.start + (len(nmdata.nparray) - 1) * nmdata.xscale.delta
        assert xdata[-1] == pytest.approx(expected_end)

    def test_empty_array_plots_nothing(self, folder):
        import matplotlib.pyplot as plt
        from pyneuromatic.core.nm_data import NMData
        nmdata = NMData(name="EmptyA0")
        fig, ax = plt.subplots()
        plot_nmdata(nmdata, ax=ax, show=False)
        assert len(ax.lines) == 0


# ---------------------------------------------------------------------------
# plot_folder
# ---------------------------------------------------------------------------

class TestPlotFolder:
    def test_returns_one_ax_per_channel(self, folder):
        _, axes = plot_folder(folder, prefix="Record", show=False)
        assert len(axes) == 2

    def test_epoch_count_in_subplot_title(self, folder):
        _, axes = plot_folder(folder, prefix="Record", show=False)
        for ax in axes:
            assert "3 epochs" in ax.get_title()

    def test_channel_filter(self, folder):
        _, axes = plot_folder(folder, prefix="Record", channels=["A"], show=False)
        assert len(axes) == 1
        assert "Channel A" in axes[0].get_title()

    def test_epoch_filter(self, folder):
        _, axes = plot_folder(folder, prefix="Record", epochs=[0, 1], show=False)
        for ax in axes:
            assert "2 epochs" in ax.get_title()

    def test_custom_suptitle(self, folder):
        fig, _ = plot_folder(folder, prefix="Record", title="My Sweep", show=False)
        assert fig._suptitle.get_text() == "My Sweep"

    def test_default_suptitle_contains_folder_and_prefix(self, folder):
        fig, _ = plot_folder(folder, prefix="Record", show=False)
        assert "Test" in fig._suptitle.get_text()
        assert "Record" in fig._suptitle.get_text()

    def test_missing_prefix_raises_value_error(self, folder):
        with pytest.raises(ValueError, match="no data found"):
            plot_folder(folder, prefix="Bad", show=False)

    def test_lines_per_channel_match_epoch_count(self, folder):
        _, axes = plot_folder(folder, prefix="Record", show=False)
        for ax in axes:
            assert len(ax.lines) == 3

    def test_xlabel_on_bottom_ax_only(self, folder):
        _, axes = plot_folder(folder, prefix="Record", show=False)
        assert axes[-1].get_xlabel() != ""
        assert axes[0].get_xlabel() == ""

    def test_single_epoch_title_singular(self, folder):
        _, axes = plot_folder(folder, prefix="Record", epochs=[0], show=False)
        for ax in axes:
            assert "1 epoch" in ax.get_title()
            assert "1 epochs" not in ax.get_title()
