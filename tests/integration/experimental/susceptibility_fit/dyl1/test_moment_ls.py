import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.helpers.cli import run_paranmr
from tests.helpers.fixtures import materialize_canonical_fixture


@pytest.mark.integration
def test_dyl1_weighted_ls_moment_fit(tmp_path: Path) -> None:
    root = materialize_canonical_fixture(tmp_path=tmp_path, system="DyL1")
    cwd = root / "SIMULATIONS" / "Fitting" / "Moments" / "Weighted_LS_Obj"
    env = {**os.environ, "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(tmp_path / "mpl")}
    result = run_paranmr(
        ["--hide", "fit_susc", "DyL1_1H_Fitting_moments_iso_ax_rho.yml"],
        cwd=cwd,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    output = cwd / "DyL1_1H_Fitting_Moments_iso_ax_rho"
    diagnostics = pd.read_csv(
        output / "moment_fit_diagnostics_302.15_K.csv",
        comment="#",
        encoding="utf-8-sig",
    )
    assert set(diagnostics["quantity"]) == {"observed", "calculated"}
    assert np.isfinite(diagnostics.select_dtypes("number").to_numpy()).all()
    tensor = pd.read_csv(
        output / "susceptibility_tensor.csv", comment="#", encoding="utf-8-sig"
    )
    assert tensor.shape[0] == 1
    assert np.isfinite(tensor[["MAE (ppm)", "RMSE (ppm)"]].to_numpy()).all()
