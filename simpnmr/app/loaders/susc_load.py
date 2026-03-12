# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Load magnetic susceptibility tensors from CSV or QC output.

Reads external data and returns Susceptibility domain objects.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from simpnmr.app.policies.susc import resolve_susceptibility_source
from simpnmr.core.build.susc import susc_from_orca_xt, susc_from_spin_only_iso
from simpnmr.core.domain.tensor import Susceptibility
from simpnmr.io.csv.susc import read_susceptibilities_csv
from simpnmr.io.qc import gateway as rdrs

logger = logging.getLogger(__name__)


def resolve_susceptibilities(
    susceptibility_file: str | None,
    susceptibility_format: str | None = None,
    *,
    temperatures: list[float],
    electronic: Any | None = None,
    g_tensor: np.ndarray | None = None,
) -> list[Susceptibility]:
    """Load susceptibility objects from an explicit source or a fallback path.

    If a susceptibility source file is provided, this function delegates to
    :func:`load_susceptibilities`. Otherwise, it builds isotropic-only fallback
    susceptibility objects with zero anisotropy for the requested temperatures.

    Args:
        susceptibility_file: Optional path to the susceptibility source.
        susceptibility_format: Optional format identifier used when a source file
            is provided.
        temperatures: Temperatures in Kelvin for which susceptibility objects
            must be returned.
        electronic: Optional electronic-state context used for fallback
            susceptibility construction.
        g_tensor: Optional g-tensor used for g-corrected isotropic susceptibility
            when loading file-backed sources.

    Returns:
        Susceptibility domain objects for the requested temperatures.

    Raises:
        ValueError: If no susceptibility file is provided and the electronic-state
            data required for spin-only fallback susceptibility is unavailable.
    """
    if susceptibility_file is not None:
        return load_susceptibilities(
            susceptibility_file,
            susceptibility_format,
            electronic=electronic,
            g_tensor=g_tensor,
        )

    if electronic is None:
        raise ValueError(
            "Electronic-state data is required to build spin-only susceptibility "
            "objects when no susceptibility:file is provided"
        )

    logger.warning(
        "Susceptibility file not provided; anisotropic susceptibility will be "
        "set to zero, so only the spin-only Fermi contribution will be used "
        "in the prediction."
    )

    return [
        susc_from_spin_only_iso(
            spin=electronic.spin_S,
            orbit=electronic.orbit_L,
            total_momentum_J=electronic.total_J,
            temperature=float(temperature),
        )
        for temperature in temperatures
    ]


def load_susceptibilities(
    susceptibility_file: str,
    susceptibility_format: str | None = None,
    *,
    electronic: Any | None = None,
    g_tensor: np.ndarray | None = None,
) -> list[Susceptibility]:
    """Load susceptibility objects from an explicit source file.

    Args:
        susceptibility_file: Path to the susceptibility source file.
        susceptibility_format: Optional format identifier. If not provided, the
            format is detected from ``susceptibility_file``.
        electronic: Optional electronic-state context passed through to ORCA-based
            susceptibility builders.
        g_tensor: Optional g-tensor used for g-corrected isotropic susceptibility
            in ORCA-based loading paths.

    Returns:
        Loaded susceptibility domain objects.

    Raises:
        ValueError: If the source format is unsupported or no data is found.
    """

    backend, section = resolve_susceptibility_source(
        susceptibility_file,
        susceptibility_format,
    )

    if backend == "csv":
        rows = read_susceptibilities_csv(susceptibility_file)
        return [Susceptibility(tensor, temperature=t) for tensor, t in rows]

    if backend == "orca":
        logger.info(
            "Susceptibility source: %s (%s)",
            backend.upper(),
            section.upper(),
        )
        return _load_orca_susceptibilities(
            susceptibility_file,
            section=section,
            electronic=electronic,
            g_tensor=g_tensor,
        )

    raise ValueError(f"Unsupported susceptibility backend: {backend!r}")


def _load_orca_susceptibilities(
    susceptibility_file: str,
    *,
    section: str | None,
    electronic: Any | None = None,
    g_tensor: np.ndarray | None = None,
) -> list[Susceptibility]:
    """Load susceptibility objects from an ORCA output file.

    Args:
        susceptibility_file: Path to the ORCA output file.
        section: Resolved ORCA QDPT section label to read.
        electronic: Optional electronic-state context passed through to the ORCA
            susceptibility builder.
        g_tensor: Optional g-tensor used for g-corrected isotropic susceptibility.

    Returns:
        Loaded ORCA susceptibility domain objects.

    Raises:
        ValueError: If the ORCA section is missing or no susceptibility data is
            parsed.
    """

    if section is None:
        raise ValueError("ORCA susceptibility section is required but was None")

    # ORCA reader returns temperature -> tensor (XT), typically in cm^3 mol^-1 K.
    tensors = rdrs.read_orca_susceptibility(susceptibility_file, section)

    if not tensors:
        raise ValueError("No susceptibility data found in ORCA output")

    if g_tensor is not None:
        logger.info(
            "Using Ab-initio g-tensor–corrected isotropic magnetic susceptibility"
        )
    else:
        logger.info(
            "Using isotropic magnetic susceptibility defined as 1/3 "
            "of the trace of the susceptibility tensor"
        )

    suscs: list[Susceptibility] = []
    for temperature, tensor_xt in tensors.items():
        suscs.append(
            susc_from_orca_xt(
                temperature=float(temperature),
                tensor_xt=tensor_xt,
                electronic=electronic,
                g_tensor=g_tensor,
            )
        )

    return suscs
