'''
            simpNMR

        Copyright (C) 2025

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

This script reads temperature-dependent molar magnetic susceptibility
tensors from an ORCA output file and plots isotropic, axial, and rhombic
components versus inverse temperature for the specified section
(casscf or nevpt2).
'''

"""
Module for plotting temperature-dependent magnetic susceptibility tensors.

This script reads susceptibility tensors for CASSCF or NEVPT2 sections from an ORCA output file,
calculates isotropic, axial, and rhombic components, normalizes them, and plots these components
against inverse temperature with an optional secondary axis showing temperature in K.
It can also perform linear regression on provided experimental standard deviations (where present) and display predicted uncertainty bands.
"""

# Standard library imports
import sys
import math
import argparse
import csv
import logging

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import mu_0, physical_constants, k, Avogadro, h, c
from simple_pnmr import utils as ut

# Physical constants from scipy.constants:
#   mu_0: Vacuum permeability [H/m]
#   k: Boltzmann constant [J/K]
#   Avogadro: Avogadro constant [1/mol]
#   h: Planck constant [J·s]
#   c: Speed of light [m/s]

def get_chiT_tensors(f, start_line: str) -> tuple[list[float], dict[float, np.ndarray], str]:
    """
    Extract temperature-dependent magnetic susceptibility tensors from ORCA output.

    Args:
        f (iterator): File iterator over lines of ORCA output.
        start_line (str): Line marking the start of the susceptibility tensor section.

    Returns:
        temps (list of float): Sorted temperatures in K.
        tensors (dict of float->ndarray): Mapping from temperature to 3×3 susceptibility tensor (SI units).
        last_line (str): The last line read after parsing.
    """
    # Initialize container for temperature
    tensors = {}
    line = start_line

    # Find susceptibility tensor header
    while 'TEMPERATURE DEPENDENT MOLAR MAGNETIC SUSCEPTIBILITY TENSOR' not in line:
        line = next(f)

    # Skip header lines
    for _ in range(6):
        line = next(f)

    # Read each temperature block
    while 'TEMPERATURE/K' in line:
        temp = float(line.split('TEMPERATURE/K:')[1])

        # Advance to tensor rows
        line = next(f); line = next(f)
        row1 = [float(val) for val in line.split()]
        line = next(f)
        row2 = [float(val) for val in line.split()]
        line = next(f)
        row3 = [float(val) for val in line.split()]
        tensors[temp] = np.array([row1, row2, row3])

        # Advance past blank lines
        line = next(f); line = next(f)

    temps = sorted(tensors.keys())
    return temps, tensors, line

def get_g_matrix(f, start_line: str) -> tuple[np.ndarray, float, str]:
    """
    Extract the electronic g-matrix and spin quantum number from ORCA output.

    Args:
        f (iterator): File iterator over lines of ORCA output.
        start_line (str): Line marking the start of the g-matrix section.

    Returns:
        g_matrix (ndarray): 3×3 electronic g-matrix.
        S (float): Spin quantum number.
        last_line (str): The last line read after parsing.
    """
    line = start_line

    # Seek the header
    while 'ELECTRONIC G-MATRIX FROM EFFECTIVE HAMILTONIAN' not in line:
        line = next(f)

    # Find spin multiplicity line reliably
    while True:
        line = next(f)
        if 'Spin multiplicity =' in line:
            S = (float(line.split('Spin multiplicity =')[1].strip()) - 1) / 2
            break

    # Advance to the first numeric row of the 3x3 matrix
    def _is_numeric_row(s: str) -> bool:
        parts = s.split()
        if len(parts) < 3:
            return False
        try:
            float(parts[0]); float(parts[1]); float(parts[2])
            return True
        except ValueError:
            return False

    while True:
        line = next(f)
        if _is_numeric_row(line):
            row1 = [float(val) for val in line.split()[:3]]
            break

    line = next(f)
    if not _is_numeric_row(line):
        raise ValueError('Failed to parse g-matrix row 2')
    row2 = [float(val) for val in line.split()[:3]]

    line = next(f)
    if not _is_numeric_row(line):
        raise ValueError('Failed to parse g-matrix row 3')
    row3 = [float(val) for val in line.split()[:3]]

    g_matrix = np.array([row1, row2, row3])

    # Advance one line to return context
    line = next(f)

    return g_matrix, S, line

def get_eff_hamiltonian_matrix(f, start_line: str) -> np.ndarray: 
    """
    Extract raw Effective Hamiltonian matrix from the effective Hamiltonian section in ORCA output.

    Args:
        f (iterator): File iterator over lines of ORCA output.
        start_line (str): Line marking the start of the effective Hamiltonian section.

    Returns:
        eff_H_raw (ndarray): 3×3 electronic Effective Hamiltonian matrix.
    """

    line = start_line

    while 'Effective Hamiltonian from projected relativistic states and relativistic energies' not in line:
        line = next(f)

    # Skip lines until we find the "Raw matrix (cm-1):" header
    while 'Raw matrix (cm-1):' not in line:
        line = next(f)
    # Move to the first numeric row after that header
    line = next(f)
    while True:
        parts = line.split()
        try:
            [float(x) for x in parts]
            break
        except ValueError:
            line = next(f)

    # Read D-matrix rows
    row1 = [float(val) for val in line.split()]
    line = next(f)
    row2 = [float(val) for val in line.split()]
    line = next(f)
    row3 = [float(val) for val in line.split()]
    eff_H_raw = np.array([row1, row2, row3])

    return eff_H_raw

def calculate_chi_components_nevpt2(tensors: dict[float, np.ndarray], g_matrix: np.ndarray) -> tuple[list[float], list[float], list[float], list[np.ndarray]]:
    """
    Calculate inverse temperature list and isotropic, axial, rhombic chi components.

    Args:
        tensors (dict of float->ndarray): Mapping from temperature to 3×3 susceptibility tensor.

    Returns:
        chi_iso (list of float): Isotropic susceptibility components.
        chi_ax (list of float): Axial susceptibility components.
        chi_rho (list of float): Rhombic susceptibility components.
    """

    # Initialize containers for susceptibility components
    chi_iso, chi_ax, chi_rho = [], [], []
    chi_eigenvectors_list = []

    # Convert
    conv = 1e-6 * 4 * math.pi

    for T in tensors.keys():
        chiT = tensors[T]

        # Use trace/3 only to get the traceless part for eigen-decomposition (axial/rhombic)
        iso_trace = np.trace(chiT) / 3.0
        traceless = chiT - np.eye(3) * iso_trace

        # Principal values/vectors of the traceless tensor
        eigs, eigvls = np.linalg.eig(traceless)
        idx = np.argsort(np.abs(eigs))
        chi_diag_traceless = np.diag(eigs.real[idx])
        eigvls_sorted = eigvls[:, idx]
        chi_eigenvectors_list.append(eigvls_sorted)

        # Axial and rhombic components from the traceless eigenvalues
        chi_ax.append(1.5 * chi_diag_traceless[2, 2] / Avogadro * conv)
        chi_rho.append((chi_diag_traceless[0, 0] - chi_diag_traceless[1, 1]) / 2.0 / Avogadro * conv)

        # Liza's χ_iso definition: (χ_x/g_x + χ_y/g_y + χ_z/g_z)/3
        # Work in the χ eigenframe, but use the FULL χ tensor principal values
        V = eigvls_sorted
        chi_diag_full = np.diag(V.T @ chiT @ V)
        g_diag = np.diag(V.T @ g_matrix @ V)

        # Scale χ principal components to per-particle SI and then form the weighted average
        chi_diag_scaled = (chi_diag_full / Avogadro) * conv
        term = np.divide(chi_diag_scaled, g_diag, out=np.zeros_like(chi_diag_scaled, dtype=float), where=g_diag != 0)
        chi_iso.append((2.0023/3) * float(np.sum(term)))

    return chi_iso, chi_ax, chi_rho, chi_eigenvectors_list

def calculate_g_components(rotated_g_tensors: list[np.ndarray]) -> tuple[list[float], list[float], list[float]]:
    """
    Calculate isotropic, axial, and rhombic g-matrix components
    for each rotated g-tensor in χ bases.

    Args:
        rotated_g_tensors (list of ndarray): List of 3×3 g-matrices
        already rotated into each χ eigenframe.

    Returns:
        g_sq_iso (list of float): Isotropic components squared at each temperature.
        g_sq_ax (list of float): Axial components squared at each temperature.
        g_sq_rh (list of float): Rhombic components squared at each temperature.
    """

    # Calculate g-components directly from the diagonal of each rotated tensor
    g_sq_iso = []
    g_sq_ax  = []
    g_sq_rh  = []

    for g_mat in rotated_g_tensors:
        iso_sq = (g_mat[0, 0]**2 + g_mat[1, 1]**2 + g_mat[2, 2]**2) / 3.0
        ax_sq  = 1.5 * (g_mat[2, 2]**2 - iso_sq)
        rh_sq  = (g_mat[0, 0]**2 - g_mat[1, 1]**2) / 2.0

        g_sq_iso.append(iso_sq)
        g_sq_ax.append(ax_sq)
        g_sq_rh.append(rh_sq)

    return g_sq_iso, g_sq_ax, g_sq_rh

def calculate_E_D_components(rotated_eff_H_tensors: list[np.ndarray]) -> tuple[list[float], list[float]]:
    """
    Calculate E and D Effective Hamiltonian matrix components
    for each rotated Effective Hamiltonian tensor in χ bases.

    Args:
        rotated_eff_H_tensors (list of ndarray): List of 3×3 Effective Hamiltonian matrices
        already rotated into each χ eigenframe.

    Returns:
        D (list of float): Axial components at each temperature.
        E (list of float): Rhombic components at each temperature.
    """

    # Calculate Effective Hamiltonian directly from the diagonal of each rotated tensor
    D = []
    E = []

    for eff_H_mat in rotated_eff_H_tensors:
        iso = np.trace(eff_H_mat) / 3.0
        D.append(1.5 * (eff_H_mat[2, 2] - iso))
        # D.append(1.5 * (eff_H_mat[2, 2]))
        E.append((eff_H_mat[0, 0] - eff_H_mat[1, 1]) / 2.0)

    return D, E


# --- Helper to write all computed series to a CSV file ---
def write_results_csv(
    output_csv_path: str,
    temps: list[float],
    inv_t: list[float],
    chi_iso_nevpt2_si: list[float],
    chi_ax_nevpt2_si: list[float],
    chi_rho_nevpt2_si: list[float],
    chi_iso_analytic: list[float],
    chi_ax_analytic: list[float],
    chi_rho_analytic: list[float],
    g_sq_iso: list[float],
    g_sq_ax: list[float],
    g_sq_rh: list[float],
    D_cm_inv: list[float],
    E_cm_inv: list[float],
    D_J: list[float],
    E_J: list[float],
    chi_iso_fit_pred: list[float] | None,
    chi_ax_fit_pred: list[float] | None,
    chi_rho_fit_pred: list[float] | None,
    chi_iso_sdev_pred: list[float] | None,
    chi_ax_sdev_pred: list[float] | None,
    temps_csv: list[float] | None,
    inv_t_csv: list[float] | None,
    chi_iso_fit_csv: list[float] | None,
    chi_ax_fit_csv: list[float] | None,
    chi_rho_fit_csv: list[float] | None,
    chi_iso_sdev_csv: list[float] | None,
    chi_ax_sdev_csv: list[float] | None,
    a_iso: float | None,
    b_iso: float | None,
    a_ax: float | None,
    b_ax: float | None,
    a_rho: float | None,
    b_rho: float | None,
    a_iso_sd: float | None,
    b_iso_sd: float | None,
    a_ax_sd: float | None,
    b_ax_sd: float | None,
) -> None:
    """
    Write all calculated NEVPT2/analytic quantities and (if available) linear-regression
    fitted series to a CSV that shares the same folder as the plot image.

    Also writes a second block with the original CSV fitted points (on their own temperature grid), if provided.
    A third block stores the linear regression coefficients (slope a, intercept b) for each fitted target.

    The CSV is written on the NEVPT2 temperature grid `temps`.
    """
    # Guard against length mismatches by truncating everything to the common min length
    series = [
        temps, inv_t,
        chi_iso_nevpt2_si, chi_ax_nevpt2_si, chi_rho_nevpt2_si,
        chi_iso_analytic, chi_ax_analytic, chi_rho_analytic,
        g_sq_iso, g_sq_ax, g_sq_rh,
        D_cm_inv, E_cm_inv, D_J, E_J,
    ]

    # Optional series
    opt_series = [
        chi_iso_fit_pred, chi_ax_fit_pred, chi_rho_fit_pred,
        chi_iso_sdev_pred, chi_ax_sdev_pred,
    ]

    # Compute minimum valid length across mandatory series
    min_len = min(len(s) for s in series)

    # Helper to safely fetch a value or return '' when optional is missing
    def _get_opt(seq, i):
        if seq is None:
            return ''
        if i < len(seq):
            return seq[i]
        return ''

    headers = [
        'Temperature (K)', '1/T (1/K)',
        'chi_iso_NEPT2_norm', 'chi_ax_NEPT2_norm', 'chi_rho_NEPT2_norm',
        'chi_iso_analytic', 'chi_ax_analytic', 'chi_rho_analytic',
        '(g^2)_iso', '(g^2)_ax', '(g^2)_rho',
        'D (cm^-1)', 'E (cm^-1)', 'D (J)', 'E (J)',
        'chi_iso_fit_LR', 'chi_ax_fit_LR', 'chi_rho_fit_LR',
        'chi_iso_sdev_LR', 'chi_ax_sdev_LR',
    ]

    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i in range(min_len):
            row = [
                temps[i], inv_t[i],
                chi_iso_nevpt2_si[i], chi_ax_nevpt2_si[i], chi_rho_nevpt2_si[i],
                chi_iso_analytic[i],  chi_ax_analytic[i],  chi_rho_analytic[i],
                g_sq_iso[i], g_sq_ax[i], g_sq_rh[i],
                D_cm_inv[i], E_cm_inv[i], D_J[i], E_J[i],
                _get_opt(chi_iso_fit_pred, i), _get_opt(chi_ax_fit_pred, i), _get_opt(chi_rho_fit_pred, i),
                _get_opt(chi_iso_sdev_pred, i), _get_opt(chi_ax_sdev_pred, i),
            ]
            writer.writerow(row)

        # If we have raw CSV-fitted points, append them as a second block
        if temps_csv is not None and inv_t_csv is not None and chi_iso_fit_csv is not None and chi_ax_fit_csv is not None and chi_rho_fit_csv is not None:
            # Blank separator row
            writer.writerow([])
            # Header for raw CSV data
            raw_headers = [
                'CSV Temperature (K)', 'CSV 1/T (1/K)',
                'chi_iso_fitted_raw', 'chi_ax_fitted_raw', 'chi_rho_fitted_raw',
                'chi_iso_sdev_raw', 'chi_ax_sdev_raw',
            ]
            writer.writerow(raw_headers)
            # Determine length safely
            n_raw = min(
                len(temps_csv), len(inv_t_csv),
                len(chi_iso_fit_csv), len(chi_ax_fit_csv), len(chi_rho_fit_csv)
            )
            # sdev arrays may be None or contain None entries
            for j in range(n_raw):
                iso_sd = '' if (chi_iso_sdev_csv is None or j >= len(chi_iso_sdev_csv) or chi_iso_sdev_csv[j] is None) else chi_iso_sdev_csv[j]
                ax_sd  = '' if (chi_ax_sdev_csv  is None or j >= len(chi_ax_sdev_csv)  or chi_ax_sdev_csv[j]  is None) else chi_ax_sdev_csv[j]
                writer.writerow([
                    temps_csv[j], inv_t_csv[j],
                    chi_iso_fit_csv[j], chi_ax_fit_csv[j], chi_rho_fit_csv[j],
                    iso_sd, ax_sd,
                ])

        # Append LR coefficients as a summary block
        writer.writerow([])
        lr_headers = ['target', 'slope_a', 'intercept_b']
        writer.writerow(lr_headers)
        def _w(name, a, b):
            writer.writerow([name, '' if a is None else a, '' if b is None else b])
        _w('chi_iso (fit)', a_iso, b_iso)
        _w('chi_ax (fit)',  a_ax,  b_ax)
        _w('chi_rho (fit)', a_rho, b_rho)
        _w('chi_iso_sdev (fit)', a_iso_sd, b_iso_sd)
        _w('chi_ax_sdev (fit)',  a_ax_sd,  b_ax_sd)


# Top-level linear regression helper for chi(T) CSV fitting
def linreg_predict(x_train: list[float], y_train: list[float], x_pred: list[float]) -> tuple[list[float] | None, float | None, float | None]:
    """
    Simple linear regression y = a*x + b using numpy.polyfit.

    Args:
        x_train (list[float]): Training x values (e.g., 1/T for CSV data).
        y_train (list[float]): Training y values (e.g., chi*T for CSV data).
        x_pred (list[float]): X values where predictions are desired.

    Returns:
        (y_pred, a, b):
            y_pred (list[float] | None): Predicted y values for x_pred, or None if fitting is not possible.
            a (float | None): slope of the linear regression.
            b (float | None): intercept of the linear regression.
    """
    if x_train is None or y_train is None:
        return None, None, None
    if len(x_train) < 2 or len(y_train) < 2:
        return None, None, None
    try:
        coeffs = np.polyfit(np.array(x_train, dtype=float), np.array(y_train, dtype=float), 1)
        a, b = float(coeffs[0]), float(coeffs[1])
        y_pred = (a * np.array(x_pred, dtype=float) + b).tolist()
        return y_pred, a, b
    except Exception as e:
        logging.warning(f"Linear regression failed: {e}")
        return None, None, None

def rotate_tensor_to_chi_basis(
    tensor: np.ndarray, chi_eigenvectors_list: list[np.ndarray], temps: list[float]
) -> list[np.ndarray]:
    """
    Rotate a single 3×3 tensor into each χ eigenframe.

    Args:
        tensor (ndarray): 3×3 tensor to rotate (e.g., g_matrix or D_tensor).
        chi_eigenvectors_list (list of ndarray): List of 3×3 eigenvector matrices for each temperature.
        temps (list of float): List of temperatures corresponding to each eigenvector matrix.

    Returns:
        list[ndarray]: Rotated tensor for each χ basis, in the order of temps.
    """
    rotated = []
    for T, V in zip(temps, chi_eigenvectors_list):
        # transform tensor into the χ basis at temperature T
        rotated.append(V.T @ tensor @ V)
    return rotated

# Helper function to read susceptibility data from a CSV/TSV/whitespace file
def read_susceptibility_csv(csv_path: str) -> tuple[list[dict[str, float | None]], dict[str, str]]:
    """
    Read susceptibility data from a CSV/TSV/whitespace file and return:
      - rows: list of dicts with keys 'Temperature (K)', 'chi_iso', 'chi_ax', 'chi_rho'
      - units: dict with units for the chi columns, taken from headers
    The parser tolerates:
      * comma/semicolon/tab separated text
      * columns separated by variable runs of spaces (fixed-width-like tables)
      * optional preamble or comment lines beginning with '#'
    """
    import re
    with open(csv_path, 'r', newline='') as f:
        raw_lines = f.readlines()

    # Drop comment/blank lines for the header search
    candidate_lines = []
    for raw in raw_lines:
        s = raw.strip()
        if not s or s.startswith('#'):
            continue
        candidate_lines.append(s)

    # Find the header line including Temperature and chi_* bases
    header_idx_in_candidates = None
    for i, line in enumerate(candidate_lines):
        lc = line.lower()
        if 'temperature' in lc and 'chi_iso' in lc and 'chi_ax' in lc and 'chi_rho' in lc:
            header_idx_in_candidates = i
            break
    if header_idx_in_candidates is None:
        raise KeyError("Column 'Temperature (K)' not found")

    # Normalize header+data from that point forward into CSV by collapsing:
    #   - tabs or runs of 2+ spaces -> ','
    # Keep single spaces (e.g., inside 'Temperature (K)').
    def normalize(s: str) -> str:
        return re.sub(r'(?:\t+|\s{2,}|;)', ',', s.strip())

    normalized = []
    # Rebuild a slice starting at the detected header using the ORIGINAL raw lines,
    # but skip comments/blank lines.
    header_found = False
    for raw in raw_lines:
        s = raw.strip()
        if not s or s.startswith('#'):
            continue
        if not header_found:
            # Seek the exact header string we saw in candidate_lines
            if s != candidate_lines[header_idx_in_candidates]:
                continue
            header_found = True
        normalized.append(normalize(s))

    reader = csv.DictReader(normalized, delimiter=',')
    headers = [h.strip() if h else h for h in (reader.fieldnames or [])]

    # Debug: keep a human-readable snapshot of headers we detected
    detected_header = ', '.join(headers)

    def find_col(base_name):
        # Look for headers like "chi_iso (cm^3 mol^-1)" or "chi_iso (Å^3)"
        for h in headers:
            if h is None:
                continue
            hs = h.strip()
            if hs.lower().startswith(base_name) and '(' in hs and hs.endswith(')'):
                unit = hs[hs.rfind('(')+1:-1]
                return hs, unit
        # Try exact base name (unitless)
        for h in headers:
            if h and h.strip().lower() == base_name:
                return h.strip(), ''
        raise KeyError(f"Column for '{base_name}' not found. Headers: {headers}")

    def find_optional_col(base_name):
        for h in headers:
            if h is None:
                continue
            hs = h.strip()
            if hs.lower().startswith(base_name) and '(' in hs and hs.endswith(')'):
                unit = hs[hs.rfind('(')+1:-1]
                return hs, unit
        for h in headers:
            if h and h.strip().lower() == base_name:
                return h.strip(), ''
        return None, ''

    # Temperature header tolerance:
    # accept "Temperature (K)", "Temperature(K)", or any header starting with "Temperature"
    temp_col = None
    for h in headers:
        if not h:
            continue
        hs = h.strip()
        hsl = hs.lower().replace(' ', '')  # remove spaces to allow Temperature(K)
        if hsl == 'temperature(k)' or hs.lower().startswith('temperature'):
            temp_col = hs
            break
    if temp_col is None:
        raise KeyError(f"Column 'Temperature (K)' not found. Detected headers: {detected_header}")

    iso_col, iso_unit = find_col('chi_iso')
    ax_col,  ax_unit  = find_col('chi_ax')
    rho_col, rho_unit = find_col('chi_rho')
    iso_sdev_col, iso_sdev_unit = find_optional_col('chi_iso-s-dev')
    ax_sdev_col, ax_sdev_unit = find_optional_col('chi_ax-s-dev')

    rows = []
    for row in reader:
        if not row:
            continue
        # Skip rows where temperature cell is empty or non-numeric
        t_val = str(row.get(temp_col, '')).strip()
        try:
            T = float(t_val)
        except ValueError:
            continue
        try:
            chi_iso_val = float(str(row.get(iso_col, '')).strip())
            chi_ax_val  = float(str(row.get(ax_col,  '')).strip())
            chi_rho_val = float(str(row.get(rho_col, '')).strip())
        except ValueError:
            continue

        # Scale susceptibilities to desired units and multiply by temperature T from CSV:
        # Convert per-mole to per-particle (divide by Avogadro), multiply by T (to get chi*T),
        # then apply SI scaling factor: 1e-6 * 4 * pi
        conv = 1e-6 * 4 * math.pi
        chi_iso_fit_csv = ((chi_iso_val * T / Avogadro) * conv)
        chi_ax_fit_csv  = (chi_ax_val  * T / Avogadro) * conv
        chi_rho_fit_csv = (chi_rho_val * T / Avogadro) * conv

        chi_iso_sdev_val = None
        if iso_sdev_col is not None:
            try:
                sdev_raw = float(str(row.get(iso_sdev_col, '')).strip())
                # Convert the std dev of chi to std dev of chi*T with the same scaling
                chi_iso_sdev_val = (sdev_raw * T / Avogadro) * conv
            except ValueError:
                chi_iso_sdev_val = None

        chi_ax_sdev_val = None
        if ax_sdev_col is not None:
            try:
                sdev_raw_ax = float(str(row.get(ax_sdev_col, '')).strip())
                # Convert the std dev of chi to std dev of chi*T with the same scaling
                chi_ax_sdev_val = (sdev_raw_ax * T / Avogadro) * conv
            except ValueError:
                chi_ax_sdev_val = None

        rows.append({
            'Temperature (K)': T,
            'chi_iso': chi_iso_fit_csv,
            'chi_ax':  chi_ax_fit_csv,
            'chi_rho': chi_rho_fit_csv,
            'chi_iso_sdev': chi_iso_sdev_val,
            'chi_ax_sdev': chi_ax_sdev_val,
        })

    units = {
        'chi_iso': iso_unit,
        'chi_ax':  ax_unit,
        'chi_rho': rho_unit,
    }
    if iso_sdev_col is not None:
        units['chi_iso_sdev'] = iso_sdev_unit

    if ax_sdev_col is not None:
        units['chi_ax_sdev'] = ax_sdev_unit

    return rows, units

# Script to parse ORCA output and plot magnetic susceptibility components vs inverse temperature

def plot_chi_temperature_dependence(
    file_name: str, section: str, csv_path: str | None = None
) -> tuple[plt.Figure, plt.Axes]:
    """
    Reads susceptibility tensors from an ORCA output file and plots
    isotropic, axial and rhombic components versus inverse temperature.

    Args:
        file_name (str): Path to the ORCA output file.
        section (str): Section to parse ('casscf' or 'nevpt2').

    Returns:
        fig, ax: Matplotlib figure and axes objects of the plot.
    """

    # Parse ORCA output file for the given section to extract susceptibility tensors

    # Initialize container for temperature
    tensors = {}

    # Set minimal temperature considered
    t_limit = 160

    try:
        with open(file_name, 'r') as f:
            for line in f:
                # Find the start of the QDPT section corresponding to the chosen method
                if f'QDPT WITH {section.upper()}' in line:

                    temps, tensors, line = get_chiT_tensors(f, line)
                    g_matrix, S, line = get_g_matrix(f, line)
                    
                    # Extract the raw Effective Hamiltonian matrix
                    eff_H_raw = get_eff_hamiltonian_matrix(f, line)

                    temps = [T for T in temps if T >= t_limit]
                    tensors = {T: tensors[T] for T in temps}

    except (IOError, StopIteration) as e:
        logging.error(f"Error reading '{file_name}': {e}")

    if not tensors:
        logging.error(f"No data found for section '{section}' in file '{file_name}'")

    chi_iso_nevpt2_si, chi_ax_nevpt2_si, chi_rho_nevpt2_si, chi_eigenvectors = calculate_chi_components_nevpt2(tensors, g_matrix)
    # Optionally read additional CSV with experimental chi(T) data
    chi_iso_fit_csv, chi_ax_fit_csv, chi_rho_fit_csv = [], [], []
    chi_iso_sdev_csv = []
    chi_ax_sdev_csv = []
    temps_csv = []
    if csv_path is not None:
        try:
            csv_rows, _ = read_susceptibility_csv(csv_path)
            # keep rows that satisfy T >= t_limit to match tensors filtering
            for r in csv_rows:
                if r['Temperature (K)'] >= t_limit:
                    temps_csv.append(r['Temperature (K)'])
                    chi_iso_fit_csv.append(r['chi_iso'])
                    chi_ax_fit_csv.append(r['chi_ax'])
                    chi_rho_fit_csv.append(r['chi_rho'])
                    if r.get('chi_iso_sdev') is not None:
                        chi_iso_sdev_csv.append(r['chi_iso_sdev'])
                    else:
                        chi_iso_sdev_csv.append(None)
                    if r.get('chi_ax_sdev') is not None:
                        chi_ax_sdev_csv.append(r['chi_ax_sdev'])
                    else:
                        chi_ax_sdev_csv.append(None)
                        
        except Exception as e:
            logging.error(f"There is an error in processing CSV '{csv_path}' in plot_chi_temperature_dependence: {e}")

    # Uppload constant
    mu_B = physical_constants['Bohr magneton'][0]
    f_S = (2*S-1)*(2*S+3)

    # Compute normalization factor: mu0 * mu_B**2 * S(S+1) / (3k)
    norm_factor = (mu_0 * mu_B**2 * S * (S + 1)) / (3 * k)

    for i in range(len(temps)):
        chi_iso_nevpt2_si[i] /= norm_factor
        chi_ax_nevpt2_si[i]  /= norm_factor
        chi_rho_nevpt2_si[i] /= norm_factor

    # Normalize CSV-derived chi*T values if provided
    for i in range(len(chi_iso_fit_csv)):
        chi_iso_fit_csv[i] /= norm_factor
        chi_ax_fit_csv[i]  /= norm_factor
        chi_rho_fit_csv[i] /= norm_factor
    for i in range(len(chi_iso_sdev_csv)):
        if chi_iso_sdev_csv[i] is not None:
            chi_iso_sdev_csv[i] /= norm_factor
    for i in range(len(chi_ax_sdev_csv)):
        if chi_ax_sdev_csv[i] is not None:
            chi_ax_sdev_csv[i] /= norm_factor

    # Rotate g-tensors into each χ eigenframe
    rotated_g_tensors = rotate_tensor_to_chi_basis(g_matrix, chi_eigenvectors, temps)

    # Now compute g-components for each rotated tensor
    g_sq_iso, g_sq_ax, g_sq_rh = calculate_g_components(rotated_g_tensors)

    # Rotate D-tensors into each χ eigenframe
    rotated_eff_H_tensors = rotate_tensor_to_chi_basis(eff_H_raw, chi_eigenvectors, temps)

    # Now compute D and E components for each rotated tensor
    D_list, E_list = calculate_E_D_components(rotated_eff_H_tensors)
    
    # Convert to joules for each temperature // need to change to Kelvins
    D_J = [d * h * c * 100 for d in D_list]
    E_J = [e * h * c * 100 for e in E_list]

    inv_t = [1.0 / T for T in temps]
    # Build a separate inverse-temperature axis for CSV data (lengths may differ from theory)
    inv_t_csv = [1.0 / T for T in temps_csv] if len(temps_csv) > 0 else []

    # Linear regression of CSV data onto the NEVPT2 inverse-temperature grid
    # We fit y = a * x + b with x = 1/T
    chi_iso_fit_pred = None
    chi_ax_fit_pred = None
    chi_rho_fit_pred = None
    # Standard-deviation LR predictions (optional)
    chi_iso_sdev_pred = None
    chi_ax_sdev_pred = None
    a_iso = b_iso = a_ax = b_ax = a_rho = b_rho = None
    a_iso_sd = b_iso_sd = a_ax_sd = b_ax_sd = None

    if csv_path is not None:
        if len(inv_t_csv) >= 2:
            # LR for chi values
            chi_iso_fit_pred, a_iso, b_iso = linreg_predict(inv_t_csv, chi_iso_fit_csv, inv_t)
            chi_ax_fit_pred,  a_ax,  b_ax  = linreg_predict(inv_t_csv, chi_ax_fit_csv,  inv_t)
            chi_rho_fit_pred, a_rho, b_rho = linreg_predict(inv_t_csv, chi_rho_fit_csv, inv_t)

            # LR for standard deviations (where available)
            # Filter out rows without sdev values
            try:
                x_iso_sdev = [x for x, y in zip(inv_t_csv, chi_iso_sdev_csv) if y is not None]
                y_iso_sdev = [y for y in chi_iso_sdev_csv if y is not None]
                if len(x_iso_sdev) >= 2:
                    chi_iso_sdev_pred, a_iso_sd, b_iso_sd = linreg_predict(x_iso_sdev, y_iso_sdev, inv_t)
                    # enforce non-negative predicted std dev
                    if chi_iso_sdev_pred is not None:
                        chi_iso_sdev_pred = [max(0.0, float(v)) for v in chi_iso_sdev_pred]
            except Exception as e:
                logging.warning(f"LR on chi_iso standard deviations failed: {e}")

            try:
                x_ax_sdev = [x for x, y in zip(inv_t_csv, chi_ax_sdev_csv) if y is not None]
                y_ax_sdev = [y for y in chi_ax_sdev_csv if y is not None]
                if len(x_ax_sdev) >= 2:
                    chi_ax_sdev_pred, a_ax_sd, b_ax_sd = linreg_predict(x_ax_sdev, y_ax_sdev, inv_t)
                    if chi_ax_sdev_pred is not None:
                        chi_ax_sdev_pred = [max(0.0, float(v)) for v in chi_ax_sdev_pred]
            except Exception as e:
                logging.warning(f"LR on chi_ax standard deviations failed: {e}")
        else:
            ut.cprint('Not enough CSV points for linear regression (need at least 2).', 'cian')

    # Analytical chi iso, ax and rho calculation:
    chi_iso_analytic = [
        g_sq_iso[i] - (f_S / (45 * k * temps[i])) * (D_J[i] * g_sq_ax[i] + 3 * E_J[i] * g_sq_rh[i]) # need to remove K
        for i in range(len(temps))
    ]

    chi_ax_analytic = [
        g_sq_ax[i] - (f_S / (30 * k * temps[i])) * ((D_J[i]) * (g_sq_ax[i] + 3 * g_sq_iso[i]) - 3 * E_J[i] * g_sq_rh[i]) # need to remove K
        for i in range(len(temps))
    ]

    chi_rho_analytic = [
        g_sq_rh[i] + (f_S / (30 * k * temps[i])) * (E_J[i] * (g_sq_ax[i] - 3 * g_sq_iso[i]) + D_J[i] * g_sq_rh[i]) # need to remove K
        for i in range(len(temps))
    ]

    # --- Write results CSV next to the PNG ---
    output_png = 'chi_plot.png'
    output_csv = output_png.replace('.png', '.csv')
    write_results_csv(
        output_csv,
        temps,
        inv_t,
        chi_iso_nevpt2_si,
        chi_ax_nevpt2_si,
        chi_rho_nevpt2_si,
        chi_iso_analytic,
        chi_ax_analytic,
        chi_rho_analytic,
        g_sq_iso,
        g_sq_ax,
        g_sq_rh,
        D_list,
        E_list,
        D_J,
        E_J,
        chi_iso_fit_pred,
        chi_ax_fit_pred,
        chi_rho_fit_pred,
        chi_iso_sdev_pred,
        chi_ax_sdev_pred,
        temps_csv,
        inv_t_csv,
        chi_iso_fit_csv,
        chi_ax_fit_csv,
        chi_rho_fit_csv,
        chi_iso_sdev_csv,
        chi_ax_sdev_csv,
        a_iso,
        b_iso,
        a_ax,
        b_ax,
        a_rho,
        b_rho,
        a_iso_sd,
        b_iso_sd,
        a_ax_sd,
        b_ax_sd,
    )

    # Plot isotropic, axial, and rhombic components against inverse temperature
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(inv_t, chi_iso_nevpt2_si, label=r'$\chi_{iso}$ NEVPT2', color='blue')
    ax.plot(inv_t, chi_ax_nevpt2_si,  label=r'$\chi_{ax}$ NEVPT2', color='green')
    ax.plot(inv_t, chi_rho_nevpt2_si, label=r'$\chi_{rho}$ NEVPT2', color='red')

    ax.plot(inv_t, chi_iso_analytic, label=r'$\chi_{iso}$ Analytical', color='blue', linestyle='--')
    ax.plot(inv_t, chi_ax_analytic, label=r'$\chi_{ax}$ Analytical', color='green', linestyle='--')
    ax.plot(inv_t, chi_rho_analytic, label=r'$\chi_{rho}$ Analytical', color='red', linestyle='--')
    
    ax.plot(inv_t, g_sq_iso, label=r'$(g^2)_{\mathrm{iso}}$', color='blue', linestyle=':')
    ax.plot(inv_t, g_sq_ax, label=r'$(g^2)_{\mathrm{ax}}$', color='green', linestyle=':')
    ax.plot(inv_t, g_sq_rh, label=r'$(g^2)_{\rho}$', color='red', linestyle=':')
    
    # Plot CSV-fitted series against their own inverse-temperature axis to avoid length mismatch
    if len(inv_t_csv) == len(chi_iso_fit_csv) == len(chi_ax_fit_csv) == len(chi_rho_fit_csv) and len(inv_t_csv) > 0:
        ax.plot(inv_t_csv, chi_iso_fit_csv, label=r'$\chi_{iso}$ Fitted', color='blue', marker='o', linestyle='', markersize=5)
        ax.plot(inv_t_csv, chi_ax_fit_csv,  label=r'$\chi_{ax}$ Fitted',  color='green', marker='o', linestyle='', markersize=5)
        ax.plot(inv_t_csv, chi_rho_fit_csv, label=r'$\chi_{rho}$ Fitted', color='red', marker='o', linestyle='', markersize=5)
        # Draw standard deviation as vertical error bars for chi_iso (CSV)
        if len(chi_iso_sdev_csv) == len(inv_t_csv) and any(v is not None for v in chi_iso_sdev_csv):
            # Replace None with 0 for yerr while keeping array length consistent
            yerr = [v if (v is not None) else 0.0 for v in chi_iso_sdev_csv]
            ax.errorbar(inv_t_csv, chi_iso_fit_csv, yerr=yerr, fmt='none', ecolor='blue', alpha = 0.5, capsize=2)

        # Draw standard deviation as vertical error bars for chi_ax (CSV)
        if len(chi_ax_sdev_csv) == len(inv_t_csv) and any(v is not None for v in chi_ax_sdev_csv):
            # Replace None with 0 for yerr while keeping array length consistent
            yerr = [v if (v is not None) else 0.0 for v in chi_ax_sdev_csv]
            ax.errorbar(inv_t_csv, chi_ax_fit_csv, yerr=yerr, fmt='none', ecolor='green', alpha = 0.5, capsize=2)

    elif len(inv_t_csv) > 0:
        logging.warning('CSV arrays have inconsistent lengths; skipping CSV plots.')

    # Plot linear-regression predictions as lines on the NEVPT2 grid
    if chi_iso_fit_pred is not None:
        ax.plot(inv_t, chi_iso_fit_pred, label=r'$\chi_{iso}$ Fitted (LR)', linestyle='-.', linewidth=1.5, color='blue')
    if chi_ax_fit_pred is not None:
        ax.plot(inv_t, chi_ax_fit_pred,  label=r'$\chi_{ax}$ Fitted (LR)',  linestyle='-.', linewidth=1.5, color='green')
    if chi_rho_fit_pred is not None:
        ax.plot(inv_t, chi_rho_fit_pred, label=r'$\chi_{rho}$ Fitted (LR)', linestyle='-.', linewidth=1.5, color='red')

    # Add LR-predicted sdev bands around LR lines where available
    import numpy as _np  # local alias to avoid shadowing
    if chi_iso_fit_pred is not None and chi_iso_sdev_pred is not None:
        iso_center = _np.array(chi_iso_fit_pred, dtype=float)
        iso_sdev   = _np.array(chi_iso_sdev_pred, dtype=float)
        ax.fill_between(inv_t, iso_center - iso_sdev, iso_center + iso_sdev, alpha=0.15, facecolor='blue')
    if chi_ax_fit_pred is not None and chi_ax_sdev_pred is not None:
        ax_center = _np.array(chi_ax_fit_pred, dtype=float)
        ax_sdev   = _np.array(chi_ax_sdev_pred, dtype=float)
        ax.fill_between(inv_t, ax_center - ax_sdev, ax_center + ax_sdev, alpha=0.15, facecolor='green')

    ax.set_xlabel(r'$1/\mathrm{Temperature}\ (1/\mathrm{K})$', fontsize=16)
    # ax.set_ylabel(r'$\mathrm{\chi}\,T\ (10^{-32}\ \mathrm{m^3\,K})$', fontsize=16)
    ax.set_ylabel(r'Normalized $\chi\,T$ (dimensionless)', fontsize=16)

    # Add secondary x-axis to show Temperature (K) corresponding to inverse temperature
    sec_ax = ax.secondary_xaxis(
        'top',
        functions=(
            lambda inv: np.divide(1, inv, out=np.full_like(inv, np.nan), where=inv!=0),
            lambda T: np.divide(1, T,   out=np.full_like(T,   np.nan), where=T!=0)
        )
    )
    sec_ax.set_xlabel('Temperature (K)', fontsize=16)

    # Add legend
    if len(inv_t_csv) > 0:
            legend = ax.legend(loc='upper left', fontsize=10, ncol=5)
    else:
        legend = ax.legend(loc='upper left', fontsize=10, ncol=3)

    fig.canvas.draw()
    legend_bbox = legend.get_window_extent()

    ax_bbox = ax.get_window_extent()

    if legend_bbox.y0 < ax_bbox.y1:
        y_min, y_max = ax.get_ylim()
        y_padding = (y_max - y_min) * 0.10
        ax.set_ylim(y_min, y_max + y_padding)

    # Add grid
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('chi_plot.png', dpi=600, bbox_inches='tight')

    plt.show()
    plt.close('all')
    return fig, ax

def main():
    """
    Parse command-line arguments and generate the χ(T) components plot.
    """

    # Define command-line interface for input file and section choice
    parser = argparse.ArgumentParser(
        description=(
            'Plot χ(T) components from ORCA output\n\n'
            'Example:\n'
            '  chi_plot filename.out nevpt2 [filename.csv]'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'susc_file',
        help='ORCA output file with susceptibility data'
    )
    parser.add_argument(
        'section',
        choices=['casscf', 'nevpt2'],
        help='Section to plot (casscf or nevpt2)'
    )
    parser.add_argument(
        'csv_file',
        nargs='?',
        default=None,
        help='Optional additional CSV file to read'
    )
    parser.add_argument(
        '--log-level',
        choices=['CRITICAL','ERROR','WARNING','INFO','DEBUG'],
        default='INFO',
        help='Logging verbosity'
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format='[%(levelname)s] %(message)s')

    # Save plot
    _fig, _ax = plot_chi_temperature_dependence(args.susc_file, args.section, args.csv_file)
    ut.cprint("Saved plot to 'chi_plot.png'",'cyan')
    ut.cprint("Saved table to 'chi_plot.csv'",'cyan')

if __name__ == '__main__':
    main()