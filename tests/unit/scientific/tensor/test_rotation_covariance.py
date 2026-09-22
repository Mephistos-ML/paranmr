import numpy as np
import pytest

from simpnmr_x.core.util.transform import rotate_coords, rotate_tensor
from tests.unit.scientific.oracles.rotations import (
    rotate_tensor as reference_rotate_tensor,
)
from tests.unit.scientific.oracles.rotations import (
    z_rotation_degrees,
)


def test_explicit_quarter_turn_rotates_vector_with_right_handed_convention():
    rotation = z_rotation_degrees(90.0)
    coordinates = np.array([[1.0, 0.0, 0.0]])

    assert rotate_coords(coordinates, rotation) == pytest.approx(
        np.array([[0.0, 1.0, 0.0]])
    )


def test_tensor_rotation_matches_independent_rank_two_oracle():
    rotation = z_rotation_degrees(90.0)
    tensor = np.diag([1.0, 2.0, 4.0])

    assert rotate_tensor(tensor, rotation) == pytest.approx(
        reference_rotate_tensor(tensor, rotation)
    )


def test_tensor_rotation_preserves_spectral_invariants():
    rotation = z_rotation_degrees(37.0)
    tensor = np.array([[2.0, 0.3, -0.2], [0.3, 1.0, 0.4], [-0.2, 0.4, 3.0]])
    rotated = rotate_tensor(tensor, rotation)

    assert np.trace(rotated) == pytest.approx(np.trace(tensor))
    assert np.linalg.det(rotated) == pytest.approx(np.linalg.det(tensor))
    assert np.linalg.eigvalsh(rotated) == pytest.approx(np.linalg.eigvalsh(tensor))
