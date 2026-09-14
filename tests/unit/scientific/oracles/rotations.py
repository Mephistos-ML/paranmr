"""Explicit rotation oracles used to avoid testing rotations through themselves."""

from __future__ import annotations

import numpy as np


def z_rotation_degrees(angle: float) -> np.ndarray:
    """Return an explicit right-handed rotation about z."""
    theta = np.deg2rad(angle)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotate_tensor(tensor: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Reference rank-two tensor transform, written independently in tests."""
    return rotation @ np.asarray(tensor, dtype=float) @ rotation.T
