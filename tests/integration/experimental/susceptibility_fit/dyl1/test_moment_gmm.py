import os
from pathlib import Path

import pandas as pd
import pytest

from tests.helpers.fixtures import materialize_example_fixture
from tests.helpers.cli import run_paranmr


@pytest.mark.integration
def test_dyl1_gmm_moment_fit_writes_diagnostics(tmp_path: Path) -> None:
    root = materialize_example_fixture(tmp_path=tmp_path, system="DyL1")
    cwd = root / "SIMULATIONS" / "Fitting" / "Moments" / "GMM"
    env = {**os.environ, "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(tmp_path / "mpl")}
    result = run_paranmr(["--hide", "fit_susc", "DyL1_1H_GMM_Fitting_moments_iso_ax_rho.yml"], cwd=cwd, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    output = cwd / "DyL1_1H_Fitting_Moments_iso_ax_rho"
    assert (output / "moment_covariance_302.15_K.csv").is_file()
    assert (output / "moment_weighting_matrix_302.15_K.csv").is_file()
    assert pd.read_csv(output / "susceptibility_tensor.csv", comment="#", encoding="utf-8-sig").shape[0] == 1
