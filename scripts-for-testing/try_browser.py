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
folder.data.sets.add("SetA", ["RecordA0", "RecordA1"])    # exercise Sets
nm.folders.new("EmptyFolder")  # shows Data/Data Series even with nothing in them yet

app = QtWidgets.QApplication(sys.argv)

widget = FolderBrowserWidget(nm)
widget.setWindowTitle("pyNeuroMatic - Folder Browser")
widget.resize(640, 480)
widget.tree.expandAll()
widget.show()

# Click a group row (Data, Data Series, Channels, Epochs, Tool Folders,
# or the top-level Folders row) to see its children listed in the right
# pane. Right-click a group row *in the tree* for "New..." (works even
# on an empty folder, or the top-level "Folders" row with zero folders -
# try it on "EmptyFolder" or delete everything to see it bootstrap from
# nothing). Right-click row(s) *in the right pane* for Rename/Delete/Add
# to Set.
#
# Every group also has a "Sets" child (e.g. Data > Sets > SetA). Right-
# click "Sets" in the tree for New/Rename/Delete Set. Click a specific
# set to see its members in the right pane; right-click there to Remove
# from Set (the object itself isn't deleted).


def _on_current_changed(current, previous):
    # v1 is a pure navigator: clicking doesn't touch nm.select_keys.
    # This is just an example of observing clicks from outside the widget.
    node = current.internalPointer() if current.isValid() else None
    print("clicked:", node)


widget.tree.selectionModel().currentChanged.connect(_on_current_changed)

sys.exit(app.exec())
