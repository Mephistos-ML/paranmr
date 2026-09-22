from tests.helpers.cli import run_simpnmr_x


def test_simpnmr_x_help_with_diagnostics():
    result = run_simpnmr_x(["--help"])
    assert result.returncode == 0, (
        f"simpnmr-x --help failed with return code {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
