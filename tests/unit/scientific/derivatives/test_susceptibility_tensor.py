import numpy as np
import pytest

from paranmr.core.fitting.susceptibility.jacobian.susceptibility_tensor import (
    differentiate_tensor_by_susc_ax,
    tensor_paf_from_parameters,
)


def test_axial_tensor_derivative_matches_independent_central_difference():
    parameters = {
        "iso": 0.2,
        "ax": 0.7,
        "rho_over_ax": 0.15,
        "alpha": 0.0,
        "beta": 0.0,
        "gamma": 0.0,
    }
    step = 1e-6
    plus = {**parameters, "ax": parameters["ax"] + step}
    minus = {**parameters, "ax": parameters["ax"] - step}
    numerical = (tensor_paf_from_parameters(plus) - tensor_paf_from_parameters(minus)) / (
        2.0 * step
    )

    assert differentiate_tensor_by_susc_ax(parameters) == pytest.approx(numerical)


def test_axial_tensor_derivative_preserves_zero_trace():
    parameters = {"iso": 0.2, "ax": 0.7, "rho_over_ax": 0.15}

    derivative = differentiate_tensor_by_susc_ax(parameters)

    assert np.trace(derivative) == pytest.approx(0.0)
