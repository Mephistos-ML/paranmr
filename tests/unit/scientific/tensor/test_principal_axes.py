"""Tests for canonical tensor principal-axis conventions."""

import numpy as np

from simpnmr_x.core.domain.tensor import canonical_principal_axes


def test_canonical_principal_axes_use_deviation_from_iso_order():
    """Order axes by deviation from the isotropic value, not raw magnitude."""
    tensor = np.diag([1.0, 2.0, 100.0])

    eigenvalues, rotation = canonical_principal_axes(tensor)

    np.testing.assert_allclose(eigenvalues, [2.0, 1.0, 100.0])
    np.testing.assert_allclose(np.abs(rotation), np.eye(3)[:, [1, 0, 2]])
