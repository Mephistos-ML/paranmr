"""Least-squares moment objective components."""

from .objective import WeightedLSMomentObjective
from .weighting import build_ls_weight_vector, build_ls_weights_by_name

__all__ = [
    "WeightedLSMomentObjective",
    "build_ls_weights_by_name",
    "build_ls_weight_vector",
]
