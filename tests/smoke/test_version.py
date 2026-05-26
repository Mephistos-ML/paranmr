import subprocess


def test_simpnmr_version():
    result = subprocess.run(["paranmr", "--version"], capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout.strip() != ""
