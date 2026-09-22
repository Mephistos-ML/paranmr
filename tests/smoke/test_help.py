from tests.helpers.cli import run_simpnmr_x


def test_simpnmr_x_help():
    result = run_simpnmr_x(["--help"])
    assert result.returncode == 0
