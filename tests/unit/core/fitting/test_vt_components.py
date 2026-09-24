"""Tests for variable-temperature susceptibility component helpers."""

import numpy as np
import pytest

from simpnmr_x.core.fitting.variable_temperatures.components import (
    compute_curie_prefactor,
    compute_g_components,
    compute_g_sq_components,
    validate_common_principal_axes,
)


def test_g_components_preserve_canonical_gx_gy_order():
    """The rhombic sign follows the supplied x/y axes, without sorting."""
    g_tensor = np.diag([2.03, 1.97, 2.60])

    components = compute_g_components(g_tensor)
    squared = compute_g_sq_components(g_tensor)

    assert components["g_rho"] == pytest.approx((2.03 - 1.97) / 2.0)
    assert squared["g_sq_rh"] == pytest.approx((2.03**2 - 1.97**2) / 2.0)


def test_common_principal_axes_reject_misaligned_g_tensor():
    """Reject a g-tensor whose principal axes are not the ZFS/χ axes."""
    eff_h = np.diag([1.0, -1.0, 2.0])
    g_tensor = np.array(
        [
            [2.03, 0.0, 0.05],
            [0.0, 1.97, 0.0],
            [0.05, 0.0, 2.60],
        ]
    )

    with pytest.raises(ValueError, match="not aligned"):
        validate_common_principal_axes(eff_h, g_tensor, tolerance=1.0e-2)


def test_common_principal_axes_accept_alignment_at_tolerance():
    """Accept a tensor whose relative off-diagonal component is exactly 1%."""
    eff_h = np.diag([1.0, -1.0, 2.0])
    g_tensor = np.array(
        [
            [2.03, 0.0, 0.026],
            [0.0, 1.97, 0.0],
            [0.026, 0.0, 2.60],
        ]
    )

    validate_common_principal_axes(eff_h, g_tensor, tolerance=1.0e-2)


def test_curie_prefactor_supports_effective_total_angular_momentum():
    """Use J for TIP normalisation when the electronic state supplies it."""
    assert compute_curie_prefactor(2.0, total_J=1.0) == pytest.approx(
        compute_curie_prefactor(1.0)
    )
