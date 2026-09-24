from pathlib import Path

import pytest

from simpnmr_x.cfg.plot_chit import PlotChiTConfig


def test_plot_chit_config_resolves_sources_and_output(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / "plot.yml"
    config_file.write_text(
        """
plot_chit:
  xrd:
    file: xrd.out
    format: orca
    section: NEVPT2
  temperature:
    min: 2.0
    max: 300.0
  output:
    file: figure.pdf
"""
    )

    config = PlotChiTConfig.from_file(str(config_file))

    assert config.xrd is not None
    assert config.xrd.file == str(tmp_path / "xrd.out")
    assert config.xrd.section == "nevpt2"
    assert config.output_file == str(tmp_path / "figure.pdf")
    assert config.opt is None
    assert config.tip is None
    assert config.temperature is not None
    assert config.temperature.minimum == 2.0
    assert config.temperature.maximum == 300.0


def test_plot_chit_config_requires_source(tmp_path: Path):
    with pytest.raises(ValueError, match="at least one of xrd or opt"):
        PlotChiTConfig(output={"file": str(tmp_path / "figure.pdf")})


def test_plot_chit_config_rejects_tip_without_opt(tmp_path: Path):
    with pytest.raises(ValueError, match="requires an opt source"):
        PlotChiTConfig(
            xrd={"file": "xrd.out", "format": "orca", "section": "nevpt2"},
            tip={"mode": "analytic", "reference_temperature": "max"},
            output={"file": str(tmp_path / "figure.pdf")},
        )


def test_plot_chit_config_rejects_invalid_temperature_limits(tmp_path: Path):
    with pytest.raises(ValueError, match="greater than min"):
        PlotChiTConfig(
            xrd={"file": "xrd.out", "format": "orca", "section": "nevpt2"},
            temperature={"min": 300.0, "max": 2.0},
            output={"file": str(tmp_path / "figure.pdf")},
        )
