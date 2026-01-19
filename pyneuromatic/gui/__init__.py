# -*- coding: utf-8 -*-
"""
pyNeuroMatic GUI - Optional graphical interface.

This module requires PyQt6. Install with:
    pip install pyneuromatic[gui]
"""

# Check if GUI dependencies are available
try:
    from PyQt6 import QtWidgets, QtCore
    GUI_AVAILABLE = True
    _GUI_IMPORT_ERROR = None
except ImportError as e:
    GUI_AVAILABLE = False
    _GUI_IMPORT_ERROR = str(e)


def check_gui_available():
    """
    Check if GUI dependencies are installed.
    
    Raises
    ------
    ImportError
        If PyQt6 is not installed
    """
    if not GUI_AVAILABLE:
        raise ImportError(
            f"GUI dependencies not available: {_GUI_IMPORT_ERROR}\n"
            "Install with: pip install pyneuromatic[gui]\n"
            "Or: pip install PyQt6"
        )


# Only define exports if GUI is available
if GUI_AVAILABLE:
    __all__ = [
        'GUI_AVAILABLE',
        'check_gui_available',
    ]
else:
    __all__ = [
        'GUI_AVAILABLE',
        'check_gui_available',
    ]
    
    # Provide helpful error for common imports
    def launch_gui(*args, **kwargs):
        """Placeholder that raises informative error."""
        check_gui_available()