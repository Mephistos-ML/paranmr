"""Generalized-method-of-moments objective components."""

from .covariance import (
    JacobianMomentCovarianceConfig,
    MomentCovarianceEstimate,
    estimate_moment_covariance_from_jacobian,
)
from .objective import GMMMomentObjective
from .weighting import build_gmm_weighting_matrix

__all__ = [
    "GMMMomentObjective",
    "JacobianMomentCovarianceConfig",
    "MomentCovarianceEstimate",
    "estimate_moment_covariance_from_jacobian",
    "build_gmm_weighting_matrix",
]
