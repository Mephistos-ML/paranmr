# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

import numpy as np
import pytest

from paranmr.core.domain.mol import Nucleus
from paranmr.core.domain.tensor import Hyperfine
from paranmr.core.fitting.susceptibility.jacobian.assembly import (
    build_moment_jacobian,
)
from paranmr.core.fitting.susceptibility.jacobian.types import (
    MOMENT_JACOBIAN_PARAMETER_NAMES,
)
from paranmr.core.fitting.susceptibility.linewidths import (
    SusceptibilityLinewidthInputs,
)
from paranmr.core.fitting.susceptibility.moments.descriptors import MOMENT_NAMES


def _test_nuclei() -> list[Nucleus]:
    return [
        Nucleus(
            label="H1",
            coord=[0.0, 0.0, 0.0],
            A=Hyperfine(
                tensor_full=np.array(
                    [[1.2, 0.1, 0.0], [0.1, -0.5, 0.2], [0.0, 0.2, 0.7]],
                    dtype=float,
                )
            ),
        ),
        Nucleus(
            label="H2",
            coord=[1.0, 0.0, 0.0],
            A=Hyperfine(
                tensor_full=np.array(
                    [[-0.3, 0.0, 0.1], [0.0, 0.8, -0.2], [0.1, -0.2, 0.4]],
                    dtype=float,
                )
            ),
        ),
        Nucleus(
            label="H3",
            coord=[2.0, 0.0, 0.0],
            A=Hyperfine(
                tensor_full=np.array(
                    [[0.5, -0.1, 0.0], [-0.1, 0.2, 0.0], [0.0, 0.0, -0.9]],
                    dtype=float,
                )
            ),
        ),
    ]


@pytest.mark.unit
def test_build_moment_jacobian_returns_canonical_6x8_contract():
    result = build_moment_jacobian(
        temperature=302.15,
        parameters={
            "iso": 0.0,
            "ax": 0.12,
            "rho_over_ax": 0.08,
            "alpha": 25.0,
            "beta": 40.0,
            "gamma": 75.0,
        },
        nuclei=_test_nuclei(),
        linewidth_inputs=SusceptibilityLinewidthInputs(
            mean_inv_r6_by_atom_label={"H1": 2.0, "H2": 5.0, "H3": 3.0}
        ),
        linewidth_vars_by_name={"p1": 1000.0, "p2": 0.5},
        average_labels=(),
    )

    assert result.temperature == pytest.approx(302.15)
    assert result.moment_names == MOMENT_NAMES
    assert result.parameter_names == MOMENT_JACOBIAN_PARAMETER_NAMES
    assert result.values.shape == (6, 8)
