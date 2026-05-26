import subprocess


def test_paranmr_help_with_diagnostics():
    result = subprocess.run(["paranmr", "--help"], capture_output=True, text=True)
    assert result.returncode == 0, (
        f"paranmr --help failed with return code {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
