from tests.helpers.cli import run_paranmr


def test_paranmr_help():
    result = run_paranmr(["--help"])
    assert result.returncode == 0
