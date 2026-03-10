from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from simpnmr.app.policies.susc import resolve_susceptibility_source
from simpnmr.core.build.gtensor import build_g_tensor_from_dft_components
from simpnmr.io.qc import gateway as rdrs

logger = logging.getLogger(__name__)


def load_g_tensor_ab_initio(config: Any) -> NDArray[np.floating] | None:
    """Load the ab initio spin-Hamiltonian g-tensor according to susceptibility policy.

    This loader is intentionally narrow: it only resolves and reads the
    ab initio g-tensor from the configured susceptibility source. The calling
    pipeline is responsible for attaching it to the domain, e.g.
    `molecule.sh.g_tensor_ab_initio = g_tensor`.

    The `ab initio` qualifier is important here: this tensor is the
    spin-Hamiltonian g-tensor obtained from an electronic-structure
    calculation and should remain distinct from any DFT-derived g-tensor
    that may also be loaded elsewhere in the workflow. These tensors may
    have different physical meanings and must not be conflated.

    Args:
        config: Runtime config with at least:
            - susceptibility_file
            - susceptibility_format

    Returns:
        Ab initio g-tensor as a (3, 3) ndarray, or None if not available
        for the backend.
    """
    backend, section = resolve_susceptibility_source(
        config.susceptibility_file,
        config.susceptibility_format,
    )

    if backend != "orca":
        logger.info("g-tensor not loaded: susceptibility backend is %s.", backend)
        return None

    g_tensor = rdrs.read_g_tensor_ab_initio(
        config.susceptibility_file,
        section=section,
    )

    g_tensor = np.asarray(g_tensor, dtype=float)
    if g_tensor.shape != (3, 3):
        raise ValueError("Invalid g-tensor: expected a (3, 3) matrix.")

    logger.info("Ab-initio g-tensor loaded from ORCA susceptibility output.")
    return g_tensor


def load_g_tensor_dft(config: Any) -> NDArray[np.floating] | None:
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

    logger.info("DFT g-tensor loaded from ORCA susceptibility output.")

    return build_g_tensor_from_dft_components(
        g_rmc=g_rmc,
        g_dso=g_dso,
        g_pso=g_pso,
    )
