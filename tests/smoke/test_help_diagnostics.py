from tests.helpers.cli import run_paranmr


def test_paranmr_help_with_diagnostics():
    result = run_paranmr(["--help"])
    assert result.returncode == 0, (
        f"paranmr --help failed with return code {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
