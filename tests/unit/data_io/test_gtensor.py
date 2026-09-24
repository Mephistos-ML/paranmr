"""Tests for ORCA g-tensor readers."""

from __future__ import annotations

import numpy as np
import pytest

from simpnmr_x.io.qc.backends.orca.gtensor import read_g_tensor_ab_initio
from simpnmr_x.io.qc.errors import ParseError


def test_read_g_tensor_ab_initio_reconstructs_matrix_from_qdpt_block(tmp_path):
    """Reconstruct an ab-initio g-tensor from factors and orientation."""
    file_name = tmp_path / "orca.out"
    file_name.write_text(
        "\n".join(
            [
                "QDPT WITH NEVPT2",
                "ELECTRONIC G-MATRIX FROM EFFECTIVE HAMILTONIAN",
                "g-factors:",
                "1.0 2.0 3.0 iso = 2.0",
                "Orientation:",
                "X 0.6 0.8 0.0",
                "Y -0.8 0.6 0.0",
                "Z 0.0 0.0 1.0",
            ]
        ),
        encoding="utf-8",
    )

    g_tensor = read_g_tensor_ab_initio(str(file_name), section="nevpt2")

    expected = np.array(
        [
            [1.64, 0.48, 0.0],
            [0.48, 1.36, 0.0],
            [0.0, 0.0, 3.0],
        ]
    )
    np.testing.assert_allclose(g_tensor, expected)

def test_read_g_tensor_ab_initio_returns_none_when_block_is_absent(tmp_path):
    """Return None when the requested QDPT block is absent."""
    file_name = tmp_path / "orca.out"
    file_name.write_text("QDPT WITH CASSCF\n", encoding="utf-8")

    assert read_g_tensor_ab_initio(str(file_name), section="nevpt2") is None


def test_read_g_tensor_ab_initio_rejects_incomplete_block(tmp_path):
    """Reject a QDPT block missing the orientation matrix."""
    file_name = tmp_path / "orca.out"
    file_name.write_text(
        "\n".join(
            [
                "QDPT WITH NEVPT2",
                "ELECTRONIC G-MATRIX FROM EFFECTIVE HAMILTONIAN",
                "g-factors:",
                "1.0 2.0 3.0 iso = 2.0",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ParseError):
        read_g_tensor_ab_initio(str(file_name), section="nevpt2")


def test_read_g_tensor_ab_initio_rejects_nonorthogonal_orientation(tmp_path):
    """Reject an ORCA orientation matrix that is not orthogonal."""
    file_name = tmp_path / "orca.out"
    file_name.write_text(
        "\n".join(
            [
                "QDPT WITH NEVPT2",
                "ELECTRONIC G-MATRIX FROM EFFECTIVE HAMILTONIAN",
                "g-factors:",
                "1.0 2.0 3.0 iso = 2.0",
                "Orientation:",
                "X 1.0 0.2 0.0",
                "Y 0.0 1.0 0.0",
                "Z 0.0 0.0 1.0",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ParseError):
        read_g_tensor_ab_initio(str(file_name), section="nevpt2")
