"""
Module for computing rotation and transformation matrices used in PCS-related
coordinate mapping between NEVPT2 and DFT geometries. The module loads input
settings from a YAML configuration, reads susceptibility and hyperfine data,
and provides the function `get_rotation_and_transformation` as its public API.
"""
import os
import glob
import numpy as np
import numpy.linalg as la
import xyz_py as xyzp
from . import readers as rdrs
from . import utils as ut
from .inputs import PredictConfig

def get_rotation_and_transformation():
    """
    Compute and return both the rotation matrix (R) aligning NEVPT2 and DFT
    coordinate sets, and the final transformation matrix (trans_mat) using
    the susceptibility tensor.

    Returns
    -------
    tuple of numpy.ndarray
        R : (3, 3)
            Rotation matrix that aligns NEVPT2 geometry to DFT geometry.
        trans_mat : (3, 3)
            Transformation matrix used for PCS-related coordinate mapping.
    """

    default_ymls = glob.glob(os.path.join(os.getcwd(), "*.yml"))
    if not default_ymls:
        raise FileNotFoundError("No .yml file found in current directory")

    INPUT_YML = os.environ.get("SIMPNMR_INPUT", default_ymls[0])

    cfg = PredictConfig.from_file(INPUT_YML)
    susc_path = cfg.susceptibility_file
    hfc_path = cfg.hyperfine_file

    # Temperatures come from YAML; we treat it as a single-element list for now
    temperature = cfg.susceptibility_temperatures

    # NEVPT2 coordinates
    _, nevpt2_coords = rdrs.read_orca5_output_xyz(susc_path)

    # DFT coordinates
    qca = rdrs.QCA.guess_from_file(hfc_path)
    _ = qca.labels
    dft_coords = qca.coords

    if np.allclose(nevpt2_coords, dft_coords, rtol=1e-6, atol=1e-8):
        return np.eye(3), np.eye(3)
    else:
        pass

    if len(nevpt2_coords) != len(dft_coords):
        raise ValueError
    
    # Susceptibility tensor at selected temperature
    chi_dict = rdrs.read_orca_susceptibility(susc_path, section="nevpt2")
    chiT = chi_dict[temperature[0]]

    # Compute rotation aligning NEVPT2 → DFT
    rot_mat, rmsd = xyzp.find_rotation(nevpt2_coords, dft_coords)

    # Temperature-normalised tensor
    chi = chiT / temperature[0]

    # Eigen-decomposition
    evals, evecs = la.eigh(chi)

    # Transformation matrix
    trans_mat = evecs.T @ rot_mat

    ut.cprint(
        f' Distinct Susceptibility and DFT geometries detected; applied rotational alignment (RMSD = {rmsd:.6f}). \n',
          'cyan'
          )
    
    return rot_mat, trans_mat

# Exported API
__all__ = ["get_rotation_and_transformation"]