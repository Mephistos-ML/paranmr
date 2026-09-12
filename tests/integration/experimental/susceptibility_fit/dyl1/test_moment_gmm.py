import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.helpers.fixtures import materialize_canonical_fixture
from tests.helpers.cli import run_paranmr


@pytest.mark.integration
def test_dyl1_gmm_moment_fit_writes_diagnostics(tmp_path: Path) -> None:
    root = materialize_canonical_fixture(tmp_path=tmp_path, system="DyL1")
    cwd = root / "SIMULATIONS" / "Fitting" / "Moments" / "GMM"
    env = {**os.environ, "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(tmp_path / "mpl")}
    result = run_paranmr(["--hide", "fit_susc", "DyL1_1H_GMM_Fitting_moments_iso_ax_rho.yml"], cwd=cwd, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    output = cwd / "DyL1_1H_Fitting_Moments_iso_ax_rho"
    assert (output / "moment_covariance_302.15_K.csv").is_file()
    assert (output / "moment_weighting_matrix_302.15_K.csv").is_file()
    tensor = pd.read_csv(output / "susceptibility_tensor.csv", comment="#", encoding="utf-8-sig")
    assert tensor.shape[0] == 1
    assert np.isfinite(tensor[["MAE (ppm)", "RMSE (ppm)"]].to_numpy()).all()
    diagnostics = pd.read_csv(output / "moment_fit_diagnostics_302.15_K.csv", comment="#", encoding="utf-8-sig")
    assert set(diagnostics["quantity"]) == {"observed", "calculated"}
    assert np.isfinite(diagnostics.select_dtypes("number").to_numpy()).all()
