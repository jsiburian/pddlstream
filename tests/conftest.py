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


@pytest.fixture(scope='session', autouse=True)
def private_cache(tmp_path_factory):
    # Module-scoped fixtures solve before in_tmp_dir runs; their stream statistics stay out of the user's cache too
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv('PDDLSTREAM_CACHE_DIR', str(tmp_path_factory.mktemp('cache')))
        yield


@pytest.fixture(autouse=True)
def in_tmp_dir(tmp_path, monkeypatch):
    # Keeps the files that solvers write (stream statistics, debugging output) out of the user's directories
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('PDDLSTREAM_CACHE_DIR', str(tmp_path / 'cache'))
