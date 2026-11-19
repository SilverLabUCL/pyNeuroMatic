"""Start pyNeuroMatic's window: ``python -m pyneuromatic.gui``.

The window starts with an empty manager, its settings from the user's file
(~/.pyneuromatic/settings.toml), so the command history is complete from
"nm = NMManager()".

On macOS it restarts as pyNeuroMatic.app, so the Dock names it pyNeuroMatic,
with its icon, not "Python" (see mac_bundle, #545).
"""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    if argv is None:  # started as python -m pyneuromatic.gui
        from pyneuromatic.gui.mac_bundle import relaunch_in_bundle

        relaunch_in_bundle(["-m", "pyneuromatic.gui", *sys.argv[1:]])  # macOS: no return
    from pyneuromatic.gui import check_gui_available

    check_gui_available()
    from PyQt6 import QtWidgets

    from pyneuromatic.core.nm_manager import NMManager
    from pyneuromatic.gui.app_window import NMAppWindow, set_app_name

    manager = NMManager(quiet=True)
    manager.settings.load_user()  # saved as they change
    set_app_name()  # "pyNeuroMatic" in the macOS menu bar
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
        sys.argv if argv is None else argv)
    window = NMAppWindow(manager)
    window.resize(1400, 900)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
