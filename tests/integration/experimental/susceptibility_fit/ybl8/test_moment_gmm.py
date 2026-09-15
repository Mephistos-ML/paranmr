import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.helpers.cli import run_paranmr
from tests.helpers.fixtures import materialize_canonical_fixture


@pytest.mark.integration
def test_ybl8_gmm_moment_fit_produces_finite_shifts(tmp_path: Path) -> None:
    root = materialize_canonical_fixture(tmp_path=tmp_path, system="YbL8")
    cwd = root / "SIMULATIONS" / "Fitting" / "Moments" / "GMM"
    env = {**os.environ, "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(tmp_path / "mpl")}
    result = run_paranmr(
        ["--hide", "fit_susc", "YbL8_PD_GMM_fit_momens.yml"],
        cwd=cwd,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    output = cwd / "YbL8_1H_PD_fit_Moments"
    peak_data = pd.read_csv(
        output / "peak_data_302.15_K.csv", comment="#", encoding="utf-8-sig"
    )
    shifts = peak_data["δ_total_avg (ppm)"].to_numpy(float)
    experimental = pd.read_csv(
        root / "DATA" / "PARA" / "exp.csv", comment="#", encoding="utf-8-sig"
    )
    experimental_shifts = np.sort(experimental["shift (ppm)"].dropna().to_numpy(float))
    assert len(peak_data) == 31
    assert {
        "δ_total_avg (ppm)",
        "δ_dia_avg (ppm)",
        "δ_pc_avg (ppm)",
        "linewidth_r6_fit (ppm)",
    }.issubset(peak_data.columns)
    assert np.isfinite(shifts).all()
    assert np.isfinite(peak_data.select_dtypes("number").to_numpy()).all()
    assert np.sort(shifts) == pytest.approx(experimental_shifts, abs=5.0)
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
    assert float(tensor["RMSE (ppm)"].iloc[0]) < 5.0
