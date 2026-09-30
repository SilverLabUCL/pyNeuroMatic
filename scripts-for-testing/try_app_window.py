import sys

import numpy as np
from PyQt6 import QtWidgets

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow


manager = NMManager(quiet=True)


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

demo2 = manager.folders.new("Demo2")
add_dataseries(demo2, "Sweep", ("A", "B", "C"), n_epochs=2, phase=1.0)
add_dataseries(demo2, "Event", ("A", "C"), n_epochs=3, phase=1.5)
demo2.data.sets.add("SweepSet", ["SweepA0", "SweepB1"])

app = QtWidgets.QApplication(sys.argv)
window = NMAppWindow(manager)
window.setWindowTitle("pyNeuroMatic - App Window")
window.resize(1400, 900)
window.show()

sys.exit(app.exec())
