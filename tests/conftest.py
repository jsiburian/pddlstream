import pytest


@pytest.fixture(autouse=True)
def in_tmp_dir(tmp_path, monkeypatch):
    # Solvers write temp/ and statistics/ into the working directory
    monkeypatch.chdir(tmp_path)
