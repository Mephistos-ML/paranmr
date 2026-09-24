from pathlib import Path

from simpnmr_x.app.params.options import PlotChiTRunOptions, RuntimeSettings
from simpnmr_x.app.pipelines.plot.chit_plot import run_plot_chit
from simpnmr_x.cfg.plot_chit import PlotChiTConfig


def test_plot_chit_with_analytic_tip(tmp_path: Path):
    source = Path("tests/data/P3FeCl/DATA/CHI/P3FeCl_Susceptibility_NEVPT2.out")
    output = tmp_path / "XTvsT_double_plot.pdf"
    config = PlotChiTConfig(
        xrd={"file": str(source), "format": "orca", "section": "nevpt2"},
        opt={"file": str(source), "format": "orca", "section": "nevpt2"},
        tip={"mode": "analytic", "reference_temperature": "max"},
        output={"file": str(output)},
    )

    result = run_plot_chit(
        config,
        PlotChiTRunOptions(RuntimeSettings(show_plots=False)),
    )

    assert result == 0
    assert output.is_file()
