import numpy as np
import pytest

from paranmr.core.build.hfc import _assemble_hfc_from_full_tensor


def test_full_hyperfine_tensor_decomposes_into_isotropic_and_traceless_parts():
    full = np.array([[4.0, 1.0, -2.0], [1.0, 1.0, 0.5], [-2.0, 0.5, 7.0]])
    hfc = _assemble_hfc_from_full_tensor(tensor_full=full, label="H1")

    assert hfc.fc == pytest.approx(np.eye(3) * np.trace(full) / 3.0)
    assert np.trace(hfc.sd) == pytest.approx(0.0)
    assert hfc.fc + hfc.sd == pytest.approx(full)


def test_hyperfine_decomposition_preserves_off_diagonal_components():
    full = np.array([[2.0, 0.7, -0.4], [0.7, 3.0, 0.2], [-0.4, 0.2, 5.0]])
    hfc = _assemble_hfc_from_full_tensor(tensor_full=full, label="H1")

    assert hfc.sd[0, 1] == pytest.approx(0.7)
    assert hfc.sd[0, 2] == pytest.approx(-0.4)
    assert hfc.sd[1, 2] == pytest.approx(0.2)
