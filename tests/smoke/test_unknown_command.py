from tests.helpers.cli import run_simpnmr_x


def test_simpnmr_x_unknown_subcommand():
    result = run_simpnmr_x(["definitely_not_a_command"])
    assert result.returncode != 0
    assert result.stderr.strip() != ""
