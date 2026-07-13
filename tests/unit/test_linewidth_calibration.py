# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

import pytest

from paranmr.cfg.config import FitSuscConfig
from paranmr.core.fitting.linewidth import estimate_r6_linewidth_parameters


@pytest.mark.unit
def test_estimate_r6_linewidth_parameters_fits_nonnegative_p1_and_p2():
    result = estimate_r6_linewidth_parameters(
        mean_inv_r6_by_label={
            "a": 1.0,
            "b": 2.0,
            "c": 3.0,
        },
        observed_widths_by_label={
            "a": 2.5,
            "b": 4.5,
            "c": 6.5,
        },
        fit_offset=True,
    )

    assert result.linewidth_method == "r6"
    assert result.estimate_mode == "p1_p2"
    assert result.p1 == pytest.approx(2.0)
    assert result.p2 == pytest.approx(0.5)
    assert result.rmse == pytest.approx(0.0)


@pytest.mark.unit
def test_fit_susc_config_accepts_fixed_assignment_linewidth_estimate(tmp_path):
    config_file = tmp_path / "fit.yml"
    config_file.write_text(
        "\n".join(
            [
                "project:",
                "  name: test",
                "hyperfine:",
                "  method: pdip",
                "  file: hf.xyz",
                "  paramagnetic_centre: [0.0, 0.0, 0.0]",
                "experiment:",
                "  files: exp.csv",
                "nuclei:",
                "  include: H",
                "susc_fit:",
                "  type: isoaxrho",
                "  variables:",
                "    iso: [fit, 0.0]",
                "assignment:",
                "  method: fixed",
                "linewidth:",
                "  method: experimental",
                "  estimate: p1_p2",
            ]
        ),
        encoding="utf-8",
    )

    config = FitSuscConfig.from_file(config_file)

    assert config.linewidth_estimate == "p1_p2"


@pytest.mark.unit
def test_fit_susc_config_rejects_linewidth_estimate_for_moments(tmp_path):
    config_file = tmp_path / "fit.yml"
    config_file.write_text(
        "\n".join(
            [
                "project:",
                "  name: test",
                "hyperfine:",
                "  method: pdip",
                "  file: hf.xyz",
                "  paramagnetic_centre: [0.0, 0.0, 0.0]",
                "experiment:",
                "  files: exp.csv",
                "nuclei:",
                "  include: H",
                "susc_fit:",
                "  type: isoaxrho",
                "  variables:",
                "    iso: [fit, 0.0]",
                "assignment:",
                "  method: moments",
                "  moment_objective:",
                "    type: ls",
                "linewidth:",
                "  method: experimental",
                "  estimate: p1_p2",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="linewidth:estimate"):
        FitSuscConfig.from_file(config_file)


@pytest.mark.unit
def test_fit_susc_config_rejects_unknown_moment_weight_name(tmp_path):
    config_file = tmp_path / "fit.yml"
    config_file.write_text(
        "\n".join(
            [
                "project:",
                "  name: test",
                "hyperfine:",
                "  method: pdip",
                "  file: hf.xyz",
                "  paramagnetic_centre: [0.0, 0.0, 0.0]",
                "experiment:",
                "  files: exp.csv",
                "nuclei:",
                "  include: H",
                "diamagnetic:",
                "  method: dft",
                "  file: dia.out",
                "diamagnetic_ref:",
                "  method: dft",
                "  file: ref.out",
                "susc_fit:",
                "  type: isoaxrho",
                "  variables:",
                "    iso: [fit, 0.0]",
                "assignment:",
                "  method: moments",
                "  moment_objective:",
                "    type: ls",
                "    weights:",
                "      m7: 1.0",
                "linewidth:",
                "  method: experimental",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown moment"):
        FitSuscConfig.from_file(config_file)


@pytest.mark.unit
def test_fit_susc_config_accepts_gmm_moment_objective_placeholder(tmp_path):
    config_file = tmp_path / "fit.yml"
    config_file.write_text(
        "\n".join(
            [
                "project:",
                "  name: test",
                "hyperfine:",
                "  method: pdip",
                "  file: hf.xyz",
                "  paramagnetic_centre: [0.0, 0.0, 0.0]",
                "experiment:",
                "  files: exp.csv",
                "nuclei:",
                "  include: H",
                "diamagnetic:",
                "  method: dft",
                "  file: dia.out",
                "diamagnetic_ref:",
                "  method: dft",
                "  file: ref.out",
                "susc_fit:",
                "  type: isoaxrho",
                "  variables:",
                "    iso: [fit, 0.0]",
                "assignment:",
                "  method: moments",
                "  moment_objective:",
                "    type: gmm",
                "linewidth:",
                "  method: experimental",
            ]
        ),
        encoding="utf-8",
    )

    config = FitSuscConfig.from_file(config_file)

    assert config.assignment_moment_objective["type"] == "gmm"


@pytest.mark.unit
def test_fit_susc_config_rejects_unknown_moment_objective_type(tmp_path):
    config_file = tmp_path / "fit.yml"
    config_file.write_text(
        "\n".join(
            [
                "project:",
                "  name: test",
                "hyperfine:",
                "  method: pdip",
                "  file: hf.xyz",
                "  paramagnetic_centre: [0.0, 0.0, 0.0]",
                "experiment:",
                "  files: exp.csv",
                "nuclei:",
                "  include: H",
                "susc_fit:",
                "  type: isoaxrho",
                "  variables:",
                "    iso: [fit, 0.0]",
                "assignment:",
                "  method: moments",
                "  moment_objective:",
                "    type: made_up",
                "linewidth:",
                "  method: experimental",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ls"):
        FitSuscConfig.from_file(config_file)
