from tests.helpers.cli import run_simpnmr_x


def test_simpnmr_x_version():
    result = run_simpnmr_x(["--version"])
    assert result.returncode == 0
    assert result.stdout.strip() != ""
