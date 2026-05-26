import subprocess


def test_paranmr_unknown_subcommand():
    result = subprocess.run(
        ["paranmr", "definitely_not_a_command"], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert result.stderr.strip() != ""
