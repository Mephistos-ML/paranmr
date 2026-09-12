from tests.helpers.cli import run_paranmr


def test_paranmr_version():
    result = run_paranmr(["--version"])
    assert result.returncode == 0
    assert result.stdout.strip() != ""
