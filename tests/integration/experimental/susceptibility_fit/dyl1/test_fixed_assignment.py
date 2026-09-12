import os
from pathlib import Path

import pandas as pd
import pytest

from tests.helpers.fixtures import materialize_example_fixture
from tests.helpers.cli import run_paranmr


@pytest.mark.integration
def test_dyl1_fixed_assignment_fit(tmp_path: Path) -> None:
    root = materialize_example_fixture(tmp_path=tmp_path, system="DyL1")
    cwd = root / "SIMULATIONS" / "Fitting" / "Standart_Fit"
    env = {**os.environ, "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(tmp_path / "mpl")}
    result = run_paranmr(["--hide", "fit_susc", "DyL1_1H_Fitting.yml"], cwd=cwd, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    output = cwd / "DyL1_1H_Fitting"
    assert (output / "susceptibility_tensor.csv").is_file()
    peak_data = pd.read_csv(output / "peak_data_302.15_K.csv", comment="#", encoding="utf-8-sig")
    assert peak_data["linewidth_exp (ppm)"].notna().all()
