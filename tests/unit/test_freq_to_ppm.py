# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

import numpy as np
import pytest

from paranmr.core.const.gammas import NUCLEAR_GAMMAS
from paranmr.core.conv.freq_to_ppm import signal_widths_hz_to_ppm


@pytest.mark.unit
def test_signal_widths_hz_to_ppm_converts_widths_from_hz_to_ppm():
    widths_hz = [10.0, 20.0, 40.0]
    isotope = "1H"
    magnetic_field = 14.1

    widths_ppm = signal_widths_hz_to_ppm(widths_hz, isotope, magnetic_field)

    expected = np.asarray(widths_hz, dtype=float) / (
        NUCLEAR_GAMMAS["H"] * magnetic_field
    )
    assert widths_ppm == pytest.approx(expected)


@pytest.mark.unit
def test_signal_widths_hz_to_ppm_rejects_unknown_isotope():
    with pytest.raises(ValueError, match="nonzero gyromagnetic ratio"):
        signal_widths_hz_to_ppm([10.0], "Xx", 14.1)
