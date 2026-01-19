# -*- coding: utf-8 -*-
"""
pyNeuroMatic - Python tools for electrophysiology analysis.

A Python implementation of NeuroMatic for acquiring, analyzing,
and simulating electrophysiological data.

Website: https://github.com/SilverLabUCL/pyNeuroMatic
"""

__version__ = "0.0.1"
__author__ = "Jason Rothman"
__license__ = "MIT"

# Import core components for convenient access
try:
    from pyneuromatic.core.nm_manager import NMManager
    from pyneuromatic.core.nm_project import NMProject
    from pyneuromatic.core.nm_folder import NMFolder
    from pyneuromatic.core.nm_data import NMData
    
    # Make submodules accessible
    from . import core
    from . import analysis
    
    __all__ = [
        '__version__',
        '__author__',
        '__license__',
        # Core classes
        'NMManager',
        'NMProject',
        'NMFolder',
        'NMData',
        # Submodules
        'core',
        'analysis',
    ]

except ImportError as e:
    # If imports fail during setup/build, don't crash
    import warnings
    warnings.warn(f"Some imports failed during initialization: {e}")
    __all__ = ['__version__', '__author__', '__license__']

# GUI is imported separately (optional dependency)
# Users must explicitly do: from pyneuromatic.gui import ...