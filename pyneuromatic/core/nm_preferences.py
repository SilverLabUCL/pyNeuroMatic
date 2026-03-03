# -*- coding: utf-8 -*-
"""
Compatibility shim — nm_preferences.py renamed to nm_configurations.py.

All existing ``import pyneuromatic.core.nm_preferences as nmp`` call sites
continue to work unchanged.  New code should import nm_configurations directly.
"""
from pyneuromatic.core.nm_configurations import (
    DATASERIES_SET_LIST,
    QUIET,
    GUI,
    NAN_EQ_NAN,
)

__all__ = ["DATASERIES_SET_LIST", "QUIET", "GUI", "NAN_EQ_NAN"]
