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
from simpnmr.core.build.susc import (
    build_chi_d_tensor_from_csv,
    build_chi_d_tensor_from_orca,
    build_chi_iso_from_csv,
    build_chi_iso_g_corr,
    build_chi_iso_spin_only,
)
from simpnmr.core.domain.tensor import Susceptibility
from simpnmr.io.csv.susc import read_susceptibilities_csv
from simpnmr.io.qc import gateway as rdrs

logger = logging.getLogger(__name__)


def load_susceptibilities(
    susceptibility_file: str,
    susceptibility_format: str | None = None,
    *,
    electronic: Any | None = None,
    g_tensor: np.ndarray | None = None,
) -> list[Susceptibility]:
    """Load susceptibility objects from an explicit source file.

    This function acts as the orchestrator for susceptibility loading. It
    resolves the source backend and delegates to the corresponding CSV or ORCA
    loader helper.

    Args:
        susceptibility_file: Path to the susceptibility source file.
        susceptibility_format: Optional format identifier. If not provided, the
            format is detected from ``susceptibility_file``.
        electronic: Optional electronic-state context used for isotropic
            susceptibility enrichment.
        g_tensor: Optional g-tensor used for g-corrected isotropic
            susceptibility enrichment.

    Returns:
        Loaded susceptibility domain objects.

    Raises:
        ValueError: If the source format is unsupported.
    """
    backend, section = resolve_susceptibility_source(
        susceptibility_file,
        susceptibility_format,
    )

    if backend == "csv":
        return load_susceptibility_csv(
            susceptibility_file,
            electronic=electronic,
            g_tensor=g_tensor,
        )

    if backend == "orca":
        return load_susceptibility_orca(
            susceptibility_file,
            section=section,
            electronic=electronic,
            g_tensor=g_tensor,
        )

    raise ValueError(f"Unsupported susceptibility backend: {backend!r}")


def load_susceptibility_csv(
    susceptibility_file: str,
    *,
    electronic: Any | None = None,
    g_tensor: np.ndarray | None = None,
) -> list[Susceptibility]:
    """Load susceptibility objects from a CSV source file.

    The CSV loader always constructs the tensor-backed susceptibility object
    first. If a CSV isotropic susceptibility value is present, it is attached
    directly. Otherwise the loader attempts g-corrected isotropic enrichment,
    then spin-only isotropic enrichment, and finally logs that isotropic
    susceptibility was skipped when insufficient data is available.

    Args:
        susceptibility_file: Path to the CSV susceptibility source file.
        electronic: Optional electronic-state context used for isotropic
            susceptibility enrichment.
        g_tensor: Optional g-tensor used for g-corrected isotropic
            susceptibility enrichment.

    Returns:
        Loaded susceptibility domain objects.
    """
    rows = read_susceptibilities_csv(susceptibility_file)
    suscs: list[Susceptibility] = []

    for tensor, temperature, chi_iso in rows:
        susc = build_chi_d_tensor_from_csv(
            temperature=float(temperature),
            tensor=tensor,
        )

        if chi_iso is not None:
            logger.info("Isotropic susceptibility is loaded directly from CSV")
            susc = build_chi_iso_from_csv(susc, chi_iso=float(chi_iso))
        elif electronic is not None and g_tensor is not None:
            logger.info(
                "Using Ab-initio g-tensor-corrected isotropic magnetic susceptibility"
            )
            susc = build_chi_iso_g_corr(
                susc,
                spin=electronic.spin_S,
                orbit=electronic.orbit_L,
                total_momentum_J=electronic.total_J,
                g_tensor=g_tensor,
            )
        elif electronic is not None:
            logger.info(
                "CSV source does not provide chi_iso; using spin-only "
                "isotropic magnetic susceptibility"
            )
            susc = build_chi_iso_spin_only(
                susc,
                spin=electronic.spin_S,
                orbit=electronic.orbit_L,
                total_momentum_J=electronic.total_J,
            )
        else:
            logger.info(
                "CSV source does not provide chi_iso and insufficient "
                "electronic-state data is available; isotropic magnetic "
                "susceptibility was skipped"
            )

        suscs.append(susc)

    return suscs


def load_susceptibility_orca(
    susceptibility_file: str,
    *,
    section: str | None,
    electronic: Any | None = None,
    g_tensor: np.ndarray | None = None,
) -> list[Susceptibility]:
    """Load susceptibility objects from an ORCA output file.

    The ORCA loader always constructs the tensor-backed susceptibility object
    first. It then attempts g-corrected isotropic enrichment, then spin-only
    isotropic enrichment, and finally logs that isotropic susceptibility was
    skipped when insufficient data is available.

    Args:
        susceptibility_file: Path to the ORCA output file.
        section: Resolved ORCA QDPT section label to read.
        electronic: Optional electronic-state context used for isotropic
            susceptibility enrichment.
        g_tensor: Optional g-tensor used for g-corrected isotropic
            susceptibility enrichment.

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

    suscs: list[Susceptibility] = []
    for temperature, tensor_xt in tensors.items():
        susc = build_chi_d_tensor_from_orca(
            temperature=float(temperature),
            tensor_xt=tensor_xt,
        )

        if electronic is not None and g_tensor is not None:
            logger.info(
                "Using Ab-initio g-tensor-corrected isotropic magnetic susceptibility"
            )
            susc = build_chi_iso_g_corr(
                susc,
                spin=electronic.spin_S,
                orbit=electronic.orbit_L,
                total_momentum_J=electronic.total_J,
                g_tensor=g_tensor,
            )
        elif electronic is not None:
            logger.info("Using spin-only isotropic magnetic susceptibility")
            susc = build_chi_iso_spin_only(
                susc,
                spin=electronic.spin_S,
                orbit=electronic.orbit_L,
                total_momentum_J=electronic.total_J,
            )
        else:
            logger.info(
                "Insufficient electronic-state data is available; isotropic "
                "magnetic susceptibility was skipped"
            )

        suscs.append(susc)

    return suscs
