import numpy as np

from simpnmr_x.core.conv.a3_to_cm3mol import a3_to_cm3mol
from simpnmr_x.core.conv.cm3mol_to_a3 import cm3mol_to_a3


def test_susceptibility_unit_converters_are_inverse():
    values = np.asarray([0.0, 1.0, -2.5])

    converted = a3_to_cm3mol(values)

    np.testing.assert_allclose(cm3mol_to_a3(converted), values)
