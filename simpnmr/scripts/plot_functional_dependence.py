'''
            SimpNMR

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

This script contains a program to split rate (ac and dc *_params.csv) files
'''

import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import copy
import pandas as pd

import simpnmr.main as pnmr
import simpnmr.readers as rdrs


def load_hyperfine_data(sources: dict[str, str],
                        chem_labels: str) -> list[pnmr.Molecule]:
    '''
    Loads hyperfine data from a range of sources

    Parameters
    ----------
    sources: dict[str, str]
        Keys are name of sources e.g. functional name\n
        Values are file in which data is stored
    chem_labels: str
        File containing chemical labels and optionally math labels

    Returns
    -------
    dict[str, pnmr.Molecule]
        Keys are name of sources (copied from `sources`)\n
        Vvalues are Molecule objects for each source
    '''

    all_molecules = dict.fromkeys(sources, None)

    for source_name, source_file in sources.items():
        # Load quantum chemical hyperfine data
        calc_data = rdrs.QCA.guess_from_file(source_file)

        # Create molecule object from quantum chemical hyperfine data
        # to convert units
        molecule = pnmr.Molecule.from_QCA(
            calc_data, converter='MHz_to_Ang-3', elements='all_H'
        )

        molecule.add_chem_labels_from_file(chem_labels)

        all_molecules[source_name] = molecule

    return all_molecules


def plot_component(func_comps: dict[str, dict[str, float]], ylabel: str,
                   show: bool = True, save: bool = True,
                   fig: plt.Figure = None, ax: plt.Axes = None,
                   savename: str = 'hyperfines.png',
                   figure_title: str = 'Hyperfine coupling constants'):
    '''
    Plots hyperfine data for a set of different functionals

    Parameters
    ----------
    func_comps: dict[str, dict[str, float]]
        Outer dictionary keys are functional name\n
        Inner dictionary keys are chemical label or chemical math label\n
        values are HFCC
    ylabel: str
        ylabel of plot
    show: bool, default True
        Show plot
    show: bool, default True
        Save plot to `savename`
    figure_title: str, default 'Hyperfine coupling constants'
        Title of figure window

    Returns
    -------
    plt.Figure
        Figure
    plt.Axes
        Axes
    '''

    if None in [fig, ax]:
        fig, ax = plt.subplots(
            1,
            1,
            num=figure_title
        )

    # width of bars, and shift to apply for starting positions
    width = 1 / (len(func_comps) + 1)
    shifts = [width + width * it for it in range(len(func_comps))]

    _frst = list(func_comps.keys())[0]
    xvals = np.arange(1, len(func_comps[_frst]) + 1)

    for (functional, a_vals), shift in zip(func_comps.items(), shifts):
        if functional == 'pdip':
            ax.bar(
                xvals + shift,
                a_vals.values(),
                width=width,
                label='Point Dipole',
                color='k'
            )
        else:
            ax.bar(
                xvals + shift,
                a_vals.values(),
                width=width,
                label=functional
            )

    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.set_xticks(xvals + 0.5)
    ax.set_xticklabels(func_comps[_frst].keys())
    ax.grid(axis='x', ls='--', which='minor')
    ax.set_xlim(0.5, len(func_comps[_frst]) + 1.5)
    ax.xaxis.set_tick_params('major', length=0)

    ax.hlines(0, 0.5, len(func_comps[_frst]) + 1.5, lw=0.5, color='k')

    ax.legend()
    ax.set_ylabel(ylabel)
    fig.tight_layout()

    if save:
        plt.savefig(f'{savename}', dpi=500)
    if show:
        plt.show()

    return fig, ax


def plot_normalisation(norms: dict[str, float], save=True, show=True,
                       savename='normalisation.png',
                       figure_title='Normalisation'):

    fig, ax = plt.subplots(num=figure_title)

    for name, value in norms.items():
        print(name, value)

    for it, val in enumerate(norms.values()):
        ax.plot(it, val, lw=0, marker='x', color='k')

    ax.set_xticks(np.arange(len(norms)))
    ax.set_xticklabels(norms.keys(), rotation=45)

    ax.set_ylabel(r'$A_\mathregular{iso, max}$')

    fig.tight_layout()

    if save:
        plt.savefig(f'{savename}', dpi=500)
    if show:
        plt.show()

    return


def main():
    parser = argparse.ArgumentParser(
        description=(
            'This script allows you to plot multiple sets of Hyperfine data'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'input_file',
        type=str,
        help=(
            '.csv file with two columns "names" and "files", where "files"\n'
            'contains Quantum chemistry output file names'
        )
    )

    parser.add_argument(
        'chem_labels',
        type=str,
        help='.csv file containing chemical labels of each atom'
    )

    parser.add_argument(
        '-w',
        '--window_append',
        type=str,
        help='Appends to window titles'
    )

    uargs = parser.parse_args()

    # Load input file
    config = pd.read_csv(
        uargs.input_file,
        skip_blank_lines=True,
        skipinitialspace=True,
        comment='#'
    )

    sources = {
        name: file
        for name, file in zip(config['names'], config['files'])
    }

    molecules = load_hyperfine_data(
        sources,
        chem_labels=uargs.chem_labels
    )

    all_isos = {
        name: {
            nuc.chem_math_label: nuc.A.iso
            for nuc in molecule.nuclei
        }
        for name, molecule in molecules.items()
    }

    # Isotropic parts
    plot_component(
        all_isos,
        r'$A_\mathregular{iso} \mathregular{(ppm \ \AA^{-3})}$',
        figure_title=uargs.window_append
    )

    # Isotropic parts relative to largest value for that functional

    all_relative_isos = copy.deepcopy(all_isos)
    norm_vals = dict.fromkeys(all_isos, 0.)

    for name, relative_isos in all_isos.items():
        for lab in relative_isos.keys():
            all_relative_isos[name][lab] /= np.max(np.abs(list(relative_isos.values()))) # noqa
            norm_vals[name] = np.max(np.abs(list(relative_isos.values())))

    for name, valdict in all_relative_isos.items():
        all_relative_isos[name] = dict(
            sorted(valdict.items(), key=lambda item: item[1])
        )

    plot_normalisation(
        norm_vals,
        figure_title=uargs.window_append
    )

    plot_component(
        all_relative_isos,
        r'$A_\mathregular{iso}$ / $A_\mathregular{iso, max}$',
        figure_title=uargs.window_append
    )

    plot_component(
        all_relative_isos,
        r'$A_\mathregular{iso}$ / $A_\mathregular{iso, max}$',
        figure_title=uargs.window_append
    )

    all_ax = {
        name: {
            nuc.chem_math_label: nuc.A.dip[0, 0] - nuc.A.dip[1, 1]
            for nuc in molecule.nuclei
        }
        for name, molecule in molecules.items()
    }

    all_rho = {
        name: {
            nuc.chem_math_label: -nuc.A.dip[0, 0] - nuc.A.dip[1, 1]
            for nuc in molecule.nuclei
        }
        for name, molecule in molecules.items()
    }

    plot_component(
        all_ax,
        r'$A_\mathregular{ax}$',
        figure_title=uargs.window_append
    )

    plot_component(
        all_rho,
        r'$A_\mathregular{rho}$',
        figure_title=uargs.window_append
    )
