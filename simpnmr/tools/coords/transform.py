# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Transform coordinates for PCS geometry mapping.

Provides helpers to align Susceptibility and HFC geometries and to rotate coordinates
into the susceptibility principal-axis (chi) frame.
"""

import logging
import os
import re

import numpy as np
import numpy.linalg as la

from simpnmr.io.xyz import xyz_write
from simpnmr.tools.coords import xyz_fmt

from ...cfg import config as inps
from ...io.qc import gateway as rdrs

logger = logging.getLogger(__name__)


def access_input_data(cfg: inps.PredictConfig):
    """
    Load and extract all PCS-related input data using an already parsed
    PredictConfig instance.

    Args:
        cfg (PredictConfig): Parsed YAML configuration object containing file paths
            and susceptibility settings.

    Returns:
        tuple[np.ndarray, list[float], list[str], np.ndarray, np.ndarray]:
            - chiT (np.ndarray): Susceptibility tensor for the target temperature
              with shape (3, 3).
            - temperature (list[float]): Temperature values provided in the YAML
              configuration.
            - nevpt2_labels (list[str]): Atomic labels extracted from the NEVPT2
              coordinate file.
            - nevpt2_coords (np.ndarray): NEVPT2 atomic Cartesian coordinates with
              shape (N, 3).
            - dft_coords (np.ndarray): DFT atomic Cartesian coordinates extracted
              from the hyperfine file with shape (N, 3).
    """

    # Temperatures come from YAML; we treat it as a single-element list for now
    temperature = cfg.susceptibility_temperatures

    # NEVPT2 coordinates
    nevpt2_labels, nevpt2_coords = rdrs.read_orca5_output_xyz(cfg.susceptibility_file)

    # DFT coordinates
    qca = rdrs.QCA.guess_from_file(cfg.hyperfine_file)
    dft_coords = qca.coords

    # Susceptibility tensor
    chi_dict = rdrs.read_orca_susceptibility(cfg.susceptibility_file, section="nevpt2")
    chiT = chi_dict[temperature[0]]

    return chiT, temperature, nevpt2_labels, nevpt2_coords, dft_coords


def get_rotation_and_transformation(cfg: inps.PredictConfig):
    """
    Compute and return both the rotation matrix aligning NEVPT2 and DFT
    geometries and the final transformation matrix used for PCS mapping.

    Args:
        cfg (PredictConfig): Parsed YAML configuration containing susceptibility
            and geometry inputs.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - rot_mat (np.ndarray): Rotation matrix with shape (3, 3) that maps
              DFT-frame coordinates/components into the NEVPT2 (susceptibility) frame.
            - trans_mat (np.ndarray): Transformation matrix with shape (3, 3) that maps
              DFT-frame coordinates/components into the susceptibility principal-axis
              (chi) frame.
    """

    chiT, temperature, _, nevpt2_coords, dft_coords = access_input_data(cfg)

    if len(nevpt2_coords) != len(dft_coords):
        raise ValueError(
            "NEVPT2 and DFT coordinate sets have different lengths; cannot determine "
            "a meaningful rotational alignment."
        )

    # Compute rotation aligning DFT → NEVPT2 (susceptibility frame).
    # `find_rotation(coords_1, coords_2)` returns R
    # such that coords_2 rotated onto coords_1.
    if np.allclose(nevpt2_coords, dft_coords, rtol=1e-6, atol=1e-8):
        rot_mat = np.eye(3)
        rmsd = 0.0
    else:
        rot_mat, rmsd = xyz_fmt.find_rotation(nevpt2_coords, dft_coords)

    # Temperature-normalised tensor (scaling does not change eigenvectors)
    chi = chiT / temperature[0]

    # Use a single chi-frame convention everywhere:
    # 1) remove isotropic (trace) component
    # 2) diagonalise
    # 3) sort axes by |eigenvalue| (same convention as rotate_coords_to_chi_frame)
    chi_traceless = chi - np.eye(3) * (np.trace(chi) / 3.0)
    evals, evecs = la.eigh(chi_traceless)
    idx = np.argsort(np.abs(evals))
    evecs = evecs[:, idx]

    # Transformation matrix mapping DFT → chi frame
    trans_mat = evecs.T @ rot_mat

    if rmsd > 0.0:
        logger.warning(
            "Distinct Susceptibility and DFT geometries detected; "
            "applied rotational alignment (RMSD = %.2f).",
            rmsd,
        )

    # TODO Need to add an additional functional to check if HFC coords are in chi frame
    # because it leads to the wrong prediction

    return rot_mat, trans_mat


def rotate_coords_to_chi_frame(file_path, cfg: inps.PredictConfig):
    """
    Rotate NEVPT2 coordinates into the principal-axis frame of the
    susceptibility (chi) tensor and write the resulting structure to an
    XYZ file.

    Args:
        file_path (str): Directory in which the output chi-frame XYZ file
            should be saved.
        cfg (PredictConfig): Parsed YAML configuration containing susceptibility
            and geometry inputs.

    Returns:
        list[tuple[str, np.ndarray]]: A list of (label, coordinate) pairs
        representing the rotated structure, suitable for downstream
        processing.
    """

    chiT, _, nevpt2_labels, nevpt2_coords, _ = access_input_data(cfg)

    # Subtract isotropic component (trace)
    chiT_traceless = chiT - np.eye(3) * (np.trace(chiT) / 3.0)

    # Diagonalize matrix
    eigvals_traceless, eigvecs_traceless = la.eigh(chiT_traceless)

    idx = np.argsort(np.abs(eigvals_traceless))

    # Single chi-frame convention (must match get_rotation_and_transformation):
    # sort eigenvectors by |eigenvalue|, no additional arbitrary global-axis alignment.
    eigenvecs_sort_traceless = eigvecs_traceless[:, idx]

    # Center NEVPT2 coordinates
    nevpt2_coords_center = nevpt2_coords.mean(axis=0, keepdims=True)
    nevpt2_coords_centerless = nevpt2_coords - nevpt2_coords_center

    # Convert NEVPT2 coordinates to chi frame
    nevpt2_coords_chi_frame = (
        nevpt2_coords_centerless @ eigenvecs_sort_traceless + nevpt2_coords_center
    )

    # Clean labels (remove numeric indices, if any)
    clean_labels = [re.sub(r"\d+", "", str(label)) for label in nevpt2_labels]

    # Prepare output directory and filename
    os.makedirs(file_path, exist_ok=True)
    xyz_filename = os.path.join(file_path, "chi_frame_structure.xyz")

    # Build a descriptive comment line
    _comment = "NEVPT2 coordinates rotated into the susceptibility (chi) frame."

    # Save XYZ
    xyz_write.save_xyz(
        file_name=xyz_filename,
        labels=clean_labels,
        coords=nevpt2_coords_chi_frame,
        verbose=False,
        comment=_comment,
    )

    logger.info("Chi-frame coordinates saved to %s", xyz_filename)

    # Return list of (label, coord) tuples for possible downstream use
    coords_chi_frame_out = list(zip(clean_labels, nevpt2_coords_chi_frame))

    return coords_chi_frame_out
