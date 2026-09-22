import pytest

from tests.helpers.cli import run_simpnmr_x


@pytest.mark.parametrize(
    "subcommand",
    [
        "fit_susc",
        "predict",
        "fit_corr_time",
        "calc_pcs_iso",
    ],
)
def test_simpnmr_x_subcommand_help(subcommand):
    result = run_simpnmr_x([subcommand, "--help"])
    assert result.returncode == 0
