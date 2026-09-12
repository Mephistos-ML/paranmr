import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.helpers.fixtures import materialize_example_fixture


@pytest.mark.integration
def test_dyl1_weighted_ls_moment_fit(tmp_path: Path) -> None:
    root = materialize_example_fixture(tmp_path=tmp_path, system="DyL1")
    cwd = root / "SIMULATIONS" / "Fitting" / "Moments" / "Weighted_LS_Obj"
    env = {**os.environ, "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(tmp_path / "mpl")}
    result = subprocess.run(["paranmr", "--hide", "fit_susc", "DyL1_1H_Fitting_moments_iso_ax_rho.yml"], cwd=cwd, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    output = cwd / "DyL1_1H_Fitting_Moments_iso_ax_rho"
    diagnostics = pd.read_csv(output / "moment_fit_diagnostics_302.15_K.csv", comment="#", encoding="utf-8-sig")
    assert set(diagnostics["quantity"]) == {"observed", "calculated"}
    assert np.isfinite(diagnostics.select_dtypes("number").to_numpy()).all()
