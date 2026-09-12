import numpy as np
import pytest

from paranmr.core.build.eff_factors import calc_g_eff, choose_S_eff
from paranmr.core.const.physics import KB, MUB, MU0
from paranmr.core.phys.susc import get_spin_only_susc


def test_spin_only_susceptibility_matches_curie_reference_formula():
    spin, orbit, temperature = 2.5, 5.0, 302.15
    g_eff = calc_g_eff(spin, orbit, None)
    s_eff = choose_S_eff(spin, None)
    expected_si = MU0 * MUB**2 * g_eff**2 * s_eff * (s_eff + 1.0) / (
        3.0 * KB * temperature
    )

    assert get_spin_only_susc(spin, orbit, None, temperature) == pytest.approx(
        expected_si * 1e30
    )


def test_curie_susceptibility_scales_inversely_with_temperature():
    chi_low = get_spin_only_susc(2.5, 5.0, None, 200.0)
    chi_high = get_spin_only_susc(2.5, 5.0, None, 400.0)

    assert chi_low / chi_high == pytest.approx(2.0)


def test_lande_factor_matches_explicit_reference_formula():
    spin, orbit, total_momentum = 0.5, 3.0, 3.5
    expected = 1.5 + (
        spin * (spin + 1.0) - orbit * (orbit + 1.0)
    ) / (2.0 * total_momentum * (total_momentum + 1.0))

    assert calc_g_eff(spin, orbit, total_momentum) == pytest.approx(expected)


def test_spin_only_uses_free_electron_g_factor_when_j_is_absent():
    assert calc_g_eff(2.5, 5.0, None) == pytest.approx(calc_g_eff(2.5, 5.0, 0.0))
