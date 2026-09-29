import sys

import numpy as np
from PyQt6 import QtWidgets

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow


manager = NMManager(quiet=True)
folder = manager.folders.new("Demo")
for chan in ("A", "B"):
    for epoch in range(3):
        arr = np.sin(np.linspace(0, 2 * np.pi, 200)) + epoch * 0.2
        folder.data.new(
            f"Record{chan}{epoch}",
            nparray=arr,
            xscale={"start": 0.0, "delta": 0.1, "label": "Time", "units": "ms"},
            yscale={"label": "Vm", "units": "mV"},
        )
folder.sync_dataseries("Record")

app = QtWidgets.QApplication(sys.argv)
window = NMAppWindow(manager)
window.setWindowTitle("pyNeuroMatic - App Window")
window.resize(1400, 900)
window.show()

sys.exit(app.exec())
