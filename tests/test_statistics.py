"""Stream statistics are saved in the user's cache directory instead of the working directory."""

import os

from pddlstream.core.utils import ensure_dir, write_pickle
from pddlstream.language.statistics import LEGACY_DATA_DIR, get_data_path, load_data


def test_cache_location(monkeypatch, tmp_path):
    monkeypatch.delenv('PDDLSTREAM_CACHE_DIR')
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'xdg'))
    assert get_data_path('tamp') == str(tmp_path / 'xdg' / 'pddlstream' / 'statistics' / 'tamp.pkl')


def test_legacy_statistics():
    # Statistics saved by earlier versions in ./statistics/ are read until new ones are saved
    legacy_path = os.path.join(LEGACY_DATA_DIR, 'tamp.pkl')
    ensure_dir(legacy_path)
    write_pickle(legacy_path, {'legacy': {}})
    assert load_data('tamp') == {'legacy': {}}
    ensure_dir(get_data_path('tamp'))
    write_pickle(get_data_path('tamp'), {'cached': {}})
    assert load_data('tamp') == {'cached': {}}
