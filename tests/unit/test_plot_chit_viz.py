import matplotlib.pyplot as plt
import numpy as np
import pytest

from simpnmr_x.viz.plots import susc
from simpnmr_x.viz.style.theme import build_spec


def test_plot_chit_comparison_uses_temperature_axis(monkeypatch):
    captured: dict[str, object] = {}

    def capture_figure(fig, **kwargs):
        captured["figure"] = fig

    monkeypatch.setattr(susc, "render_figure", capture_figure)

    spec = build_spec("paper")
    with spec.context():
        susc.plot_chit_comparison(
            {
                "XRD Geometry": (
                    np.asarray([2.0, 300.0]),
                    np.asarray([1.0, 2.0]),
                ),
                "Opt. Geometry": (
                    np.asarray([2.0, 300.0]),
                    np.asarray([1.1, 2.1]),
                ),
                "Opt. Geom. - TIP": (
                    np.asarray([2.0, 300.0]),
                    np.asarray([1.2, 2.2]),
                ),
            },
            spec,
            show=False,
            save=False,
            temperature_limits=(2.0, 300.0),
        )

    figure = captured["figure"]
    axis = figure.axes[0]
    np.testing.assert_allclose(axis.lines[0].get_xdata(), [2.0, 300.0])
    assert axis.get_xlim() == pytest.approx((-12.9, 314.9))
    assert axis.get_ylim() == pytest.approx((0.94, 2.32))
    assert axis.get_xlabel() == r"$T$ (K)"
    assert [line.get_color() for line in axis.lines] == [
        spec.palette.secondary,
        spec.palette.primary,
        spec.palette.highlight,
    ]
    plt.close(figure)
