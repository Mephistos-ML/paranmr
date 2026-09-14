import pytest

from tests.helpers.cli import run_paranmr


@pytest.mark.parametrize(
    "subcommand",
    [
        "fit_susc",
        "predict",
        "fit_corr_time",
        "calc_pcs_iso",
    ],
)
def test_paranmr_subcommand_help(subcommand):
    result = run_paranmr([subcommand, "--help"])
    assert result.returncode == 0
