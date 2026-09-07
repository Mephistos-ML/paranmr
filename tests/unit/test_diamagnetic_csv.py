from pathlib import Path

import pytest

from paranmr.app.loaders.dia_load import load_diamagnetic_shifts
from paranmr.cfg.config import FitSuscConfig
from paranmr.core.domain.exp import Experiment, Signal
from paranmr.io.csv.exp import load_experiments_from_csv, write_experiment_to_csv


def test_load_diamagnetic_csv_accepts_atom_labels(tmp_path: Path):
    input_file = tmp_path / "diamagnetic.csv"
    input_file.write_text("atom_label,shift\nH1,1.25\nH2,2.5\n", encoding="utf-8")

    shifts, key_kind, reference = load_diamagnetic_shifts(str(input_file))

    assert shifts == {"H1": 1.25, "H2": 2.5}
    assert key_kind == "atom_label"
    assert reference is None


def test_load_diamagnetic_csv_rejects_ambiguous_label_columns(tmp_path: Path):
    input_file = tmp_path / "diamagnetic.csv"
    input_file.write_text(
        "atom_label,signal_label,shift\nH1,signal-1,1.25\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="exactly one"):
        load_diamagnetic_shifts(str(input_file))


def test_moments_fit_config_accepts_csv_diamagnetic_shifts(tmp_path: Path):
    config_file = tmp_path / "moments.yml"
    config_file.write_text(
        """
project:
  name: paranmr_fitted_output
hyperfine:
  method: pdip
  file: geometry.xyz
  paramagnetic_centre: [0.0, 0.0, 0.0]
  spin: 0.5
  orbit: 3
  total_momentum_J: 3.5
nuclei:
  include: H
diamagnetic:
  method: csv
  file: diamagnetic.csv
experiment:
  files: generated_shifts.csv
assignment:
  method: moments
  moment_objective:
    type: gmm
    number_of_moments: 6
    covariance:
      method: monte_carlo
      n_samples: 500
      random_seed: 42
      perturbation:
        shift_sigma_abs: 0.02
        width_sigma_rel: 0.05
linewidth:
  method: r6
  variables:
    p1: [fit, 1000.0, [500.0, 2000.0]]
    p2: [fit, 0.1, [0.0, 1.0]]
susc_fit:
  type: isoaxrho_euler
  variables:
    iso: [fit, 0.0]
    ax: [fit, 0.01]
    rho_over_ax: [fit, 0.1]
    alpha: [fit, 0.0]
    beta: [fit, 0.0]
    gamma: [fit, 0.0]
  average_shifts: methyls
""".strip(),
        encoding="utf-8",
    )

    config = FitSuscConfig.from_file(config_file)

    assert config.assignment_method == "moments"
    assert config.diamagnetic_method == "csv"


def test_experiment_csv_writer_round_trips_through_reader(tmp_path: Path):
    input_file = tmp_path / "experiment.csv"
    experiment = Experiment(
        temperature=302.15,
        magnetic_field=4.7,
        isotope="1H",
        signals=[Signal(shift=4.2, width=100.0, area=1.0, signal_label="H1")],
    )
    write_experiment_to_csv(
        experiment,
        str(input_file),
        comment=["temperature 302.15", "magnetic_field 4.7", "isotope 1H"],
        verbose=False,
    )

    [loaded] = load_experiments_from_csv(str(input_file))

    assert loaded.keys() == ["H1"]
    assert loaded.signals[0].shift == pytest.approx(4.2)
