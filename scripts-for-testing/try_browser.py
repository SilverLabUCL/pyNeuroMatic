import sys

import numpy as np
from PyQt6 import QtWidgets

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui import FolderBrowserWidget

nm = NMManager(quiet=True)

folder = nm.folders.new("Demo")
for chan in ("A", "B"):
    for epoch in range(3):
        arr = np.sin(np.linspace(0, 2 * np.pi, 200)) + epoch * 0.2
        folder.data.new(
            "Record%s%d" % (chan, epoch),
            nparray=arr,
            xscale={"start": 0.0, "delta": 0.1, "label": "Time", "units": "ms"},
            yscale={"label": "Vm", "units": "mV"},
        )
folder.sync_dataseries("Record")
folder.data.new("Standalone0", nparray=np.array([9.0]))  # flat data, no dataseries
folder.toolfolders.get_or_create("Spike_Record_A_0")      # exercise Tool Folders group

app = QtWidgets.QApplication(sys.argv)

widget = FolderBrowserWidget(nm)
widget.setWindowTitle("pyNeuroMatic - Folder Browser")
widget.resize(320, 480)
widget.tree.expandAll()
widget.show()


def _on_current_changed(current, previous):
    # v1 is a pure navigator: clicking doesn't touch nm.select_keys.
    # This is just an example of observing clicks from outside the widget.
    node = current.internalPointer() if current.isValid() else None
    print("clicked:", node)


widget.tree.selectionModel().currentChanged.connect(_on_current_changed)

sys.exit(app.exec())
