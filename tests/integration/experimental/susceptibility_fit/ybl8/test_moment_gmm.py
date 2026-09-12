import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.helpers.fixtures import materialize_example_fixture
from tests.integration.experimental.susceptibility_fit.assertions import range_based_ppm_tolerance


@pytest.mark.integration
def test_ybl8_gmm_moment_fit_produces_finite_shifts(tmp_path: Path) -> None:
    root = materialize_example_fixture(tmp_path=tmp_path, system="YbL8")
    cwd = root / "SIMULATIONS" / "Fitting" / "Moments" / "GMM"
    env = {**os.environ, "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(tmp_path / "mpl")}
    result = subprocess.run(["paranmr", "--hide", "fit_susc", "YbL8_PD_GMM_fit_momens.yml"], cwd=cwd, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    output = cwd / "YbL8_1H_PD_fit_Moments"
    peak_data = pd.read_csv(output / "peak_data_302.15_K.csv", comment="#", encoding="utf-8-sig")
    shifts = peak_data["δ_total_avg (ppm)"].to_numpy(float)
    assert np.isfinite(shifts).all()
    assert range_based_ppm_tolerance({frozenset({"min"}): float(shifts.min()), frozenset({"max"}): float(shifts.max())}) > 0.0
