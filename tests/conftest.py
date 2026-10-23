import pytest


def pytest_addoption(parser):
    parser.addoption(
        '--record',
        action='store_true',
        help='Rewrites the recorded outputs (tests/recorded_outputs.json) of the cases that run',
    )


@pytest.fixture
def record(request):
    return request.config.getoption('--record')


@pytest.fixture(autouse=True)
def in_tmp_dir(tmp_path, monkeypatch):
    # Solvers write temp/ and statistics/ into the working directory
    monkeypatch.chdir(tmp_path)
