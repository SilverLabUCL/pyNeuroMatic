import sys

import numpy as np
from PyQt6 import QtWidgets

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow, set_app_name


manager = NMManager(quiet=True)
manager.settings.load_user()  # ~/.pyneuromatic/settings.toml, saved as it changes


def add_dataseries(folder, prefix, channels, n_epochs, phase):
    x = np.linspace(0.0, 2 * np.pi, 200)
    for channel_index, channel in enumerate(channels):
        for epoch in range(n_epochs):
            signal = np.sin(x + phase + channel_index * 0.35)
            arr = signal + epoch * 0.2 + channel_index * 0.1
            folder.data.new(
                f"{prefix}{channel}{epoch}",
                nparray=arr,
                xscale={"start": 0.0, "delta": 0.1, "label": "Time", "units": "ms"},
                yscale={"label": "Vm", "units": "mV"},
            )
    folder.sync_dataseries(prefix)


demo1 = manager.folders.new("Demo1")
add_dataseries(demo1, "Record", ("A", "B"), n_epochs=3, phase=0.0)
add_dataseries(demo1, "Stim", ("A", "B", "C"), n_epochs=2, phase=0.5)
demo1.data.sets.add("RecordSet", ["RecordA0", "RecordB1"])
# An x array for Record (200 points): time stamps 0.05-0.15 ms apart, so the
# trace is visibly stretched and squeezed when used (Prefix ⋯ › Edit Scales…,
# X › From: Times)
intervals = 0.1 + 0.05 * np.sin(np.arange(199) / 10.0)
demo1.data.new(
    "Times",
    nparray=np.concatenate(([0.0], np.cumsum(intervals))),
    yscale={"label": "Time", "units": "ms"},
)

demo2 = manager.folders.new("Demo2")
add_dataseries(demo2, "Sweep", ("A", "B", "C"), n_epochs=2, phase=1.0)
add_dataseries(demo2, "Event", ("A", "C"), n_epochs=3, phase=1.5)
demo2.data.sets.add("SweepSet", ["SweepA0", "SweepB1"])

set_app_name()  # "pyNeuroMatic" in the macOS menu bar, not "Python"
app = QtWidgets.QApplication(sys.argv)
window = NMAppWindow(manager)
window.resize(1400, 900)
window.show()

sys.exit(app.exec())
