from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from simpnmr.app.policies.susc import resolve_susceptibility_source
from simpnmr.io.qc import gateway as rdrs

logger = logging.getLogger(__name__)


def load_g_tensor(config: Any) -> NDArray[np.floating] | None:
    """Load the spin-Hamiltonian g-tensor according to susceptibility policy.

    This loader is intentionally narrow: it only resolves and reads the g-tensor.
    The calling pipeline is responsible for attaching it to the domain, e.g.
    `molecule.sh.g_tensor = g_tensor`.

    Args:
        config: Runtime config with at least:
            - susceptibility_file
            - susceptibility_format

    Returns:
        g-tensor as a (3, 3) ndarray, or None if not available for the backend.
    """
    backend, section = resolve_susceptibility_source(
        config.susceptibility_file,
        config.susceptibility_format,
    )

    if backend != "orca":
        logger.info("g-tensor not loaded: susceptibility backend is %s.", backend)
        return None

    g_tensor = rdrs.read_orca_g_tensor(
        config.susceptibility_file,
        section=section,
    )

    g_tensor = np.asarray(g_tensor, dtype=float)
    if g_tensor.shape != (3, 3):
        raise ValueError("Invalid g-tensor: expected a (3, 3) matrix.")

    logger.info("g-tensor loaded from ORCA susceptibility output.")
    return g_tensor
