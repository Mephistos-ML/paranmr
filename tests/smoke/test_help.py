import subprocess


def test_simpnmr_help():
    result = subprocess.run(["paranmr", "--help"], capture_output=True)
    assert result.returncode == 0
