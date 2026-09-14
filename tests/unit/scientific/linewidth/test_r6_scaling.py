import pytest

from paranmr.core.fitting.linewidth.r6 import predict_r6_linewidths


def test_r6_linewidth_is_affine_in_mean_inverse_sixth_distance():
    values = {"near": 1.0 / 2.0**6, "far": 1.0 / 4.0**6}
    result = predict_r6_linewidths(values, p1=12.0, p2=0.5)

    assert result["near"] == pytest.approx(12.0 / 2.0**6 + 0.5)
    assert result["far"] == pytest.approx(12.0 / 4.0**6 + 0.5)


def test_r6_distance_scaling_is_sixth_power():
    values = {"r": 1.0}
    doubled_distance = {"r": 1.0 / 2.0**6}
    first = predict_r6_linewidths(values, p1=1.0, p2=0.0)["r"]
    second = predict_r6_linewidths(doubled_distance, p1=1.0, p2=0.0)["r"]

    assert second == pytest.approx(first / 64.0)


@pytest.mark.parametrize("name,value", [("p1", -1.0), ("p2", -1.0)])
def test_r6_rejects_negative_model_parameters(name: str, value: float):
    parameters = {"p1": 1.0, "p2": 0.0}
    parameters[name] = value

    with pytest.raises(ValueError, match=name):
        predict_r6_linewidths({"H1": 1.0}, **parameters)
