from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from simpnmr.app.policies.susc import resolve_susceptibility_source
from simpnmr.core.build.gtensor import (
    build_g_tensor_ab_initio,
    build_g_tensor_from_dft_components,
)
from simpnmr.core.domain.mol import Molecule
from simpnmr.io.qc import gateway as rdrs

logger = logging.getLogger(__name__)


def load_g_tensor_ab_initio(
    molecule: Molecule,
    susceptibility_file: str | None,
    susceptibility_format: str | None = None,
) -> Molecule:
    """Load the ab-initio spin-Hamiltonian g-tensor into the molecule domain.

    This loader resolves and reads the ab-initio g-tensor from the configured
    susceptibility source, then delegates domain population to the builder layer.
    """
    if susceptibility_file is None:
        logger.info(
            "No susceptibility file provided; skipping ab initio g-tensor load."
        )
        return build_g_tensor_ab_initio(molecule, g_tensor=None)

    backend, section = resolve_susceptibility_source(
        susceptibility_file,
        susceptibility_format,
    )

    if backend != "orca":
        logger.info("g-tensor not loaded: susceptibility backend is %s.", backend)
        return build_g_tensor_ab_initio(molecule, g_tensor=None)

    g_tensor = rdrs.read_g_tensor_ab_initio(
        susceptibility_file,
        section=section,
    )

    if g_tensor is None:
        logger.info("No ab-initio g-tensor found in ORCA susceptibility output.")
        return build_g_tensor_ab_initio(molecule, g_tensor=None)

    molecule = build_g_tensor_ab_initio(molecule, g_tensor=g_tensor)

    logger.info("Ab-initio g-tensor loaded from ORCA susceptibility output.")
    return molecule


def load_g_tensor_dft(config: object) -> NDArray[np.floating] | None:
    """Load the DFT-derived g-tensor.

    This loader reads decomposed DFT g-tensor contribution tensors from the
    configured hyperfine QC source and delegates assembly of the full physical
    g-tensor to the builder layer.

    Args:
        config: Runtime config with at least:
            - hyperfine_file

    Returns:
        The assembled DFT-derived g-tensor as a (3, 3) ndarray, or None if
        no DFT g-tensor data is available in the input file.
    """
    g_components = rdrs.read_g_tensor_dft(config.hyperfine_file)
    if g_components is None:
        return None

    g_rmc, g_dso, g_pso = g_components

    g_tensor = build_g_tensor_from_dft_components(
        g_rmc=g_rmc,
        g_dso=g_dso,
        g_pso=g_pso,
    )

    g_tensor = np.asarray(g_tensor, dtype=float)
    if g_tensor.shape != (3, 3):
        raise ValueError("Invalid DFT g-tensor: expected a (3, 3) matrix.")

    logger.info("DFT g-tensor loaded from hyperfine QC output.")
    return g_tensor
