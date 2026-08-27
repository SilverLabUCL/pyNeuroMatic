import numpy as np
from pyneuromatic.core.nm_folder import NMFolder
from pyneuromatic.tools.nm_plot import plot_folder

folder = NMFolder(name="Demo")
for chan in ("A", "B"):
    for epoch in range(3):
        arr = np.sin(np.linspace(0, 2 * np.pi, 200)) + epoch * 0.2
        folder.data.new(
            "Record%s%d" % (chan, epoch),
            nparray=arr,
            xscale={"start": 0.0, "delta": 0.1, "label": "Time", "units": "ms"},
            yscale={"label": "Vm", "units": "mV"},
        )

plot_folder(folder, prefix="Record")   # show=True by default
