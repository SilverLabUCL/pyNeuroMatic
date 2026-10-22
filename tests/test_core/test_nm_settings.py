"""Tests for NMSettings (general pyNeuroMatic settings)."""
from pathlib import Path

import pytest

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.core.nm_settings import NMSettings


def test_defaults_and_choices():
    settings = NMSettings()
    assert settings.table_nan == "NaN"
    settings.table_nan = "NaN (grey)"
    settings.table_nan = "blank"
    with pytest.raises(ValueError):
        settings.table_nan = "hidden"
    with pytest.raises(AttributeError):
        settings.colour = "red"


def test_round_trip(tmp_path):
    settings = NMSettings()
    settings.table_nan = "blank"
    path = settings.save(tmp_path / "settings.toml")
    assert NMSettings.load(path).table_nan == "blank"


def test_load_user_and_save_user(tmp_path):
    path = tmp_path / "nm" / "settings.toml"
    settings = NMSettings()
    assert settings.save_user() is None  # never loaded: the file is left alone
    assert settings.load_user(path) is False  # no file yet
    assert settings.path == path
    settings.table_nan = "NaN (grey)"
    assert settings.save_user() == path and path.exists()  # makes the folder

    again = NMSettings()
    assert again.load_user(path) is True
    assert again.table_nan == "NaN (grey)"


def test_user_path_is_in_the_home_folder():
    assert NMSettings.user_path() == Path.home() / ".pyneuromatic" / "settings.toml"


def test_manager_has_default_settings():
    manager = NMManager(quiet=True)
    assert isinstance(manager.settings, NMSettings)
    assert manager.settings.table_nan == "NaN" and manager.settings.path is None
