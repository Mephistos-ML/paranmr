from pathlib import Path

from simpnmr_x.io.qc.backends.orca.susc import read_orca_chi_t


def test_read_orca_chi_t_is_section_specific():
    file_name = Path(
        "tests/data/P3FeCl/DATA/CHI/P3FeCl_Susceptibility_NEVPT2.out"
    )

    casscf = read_orca_chi_t(str(file_name), "casscf")
    nevpt2 = read_orca_chi_t(str(file_name), "nevpt2")

    assert len(casscf) == 300
    assert len(nevpt2) == 300
    assert casscf[300.0] == 3.932567
    assert nevpt2[300.0] == 3.817605
