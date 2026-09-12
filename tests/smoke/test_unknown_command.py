from tests.helpers.cli import run_paranmr


def test_paranmr_unknown_subcommand():
    result = run_paranmr(["definitely_not_a_command"])
    assert result.returncode != 0
    assert result.stderr.strip() != ""
