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
import numpy as np
import pandas as pd
import os


def load_susceptibility_data(sources: dict[str, str],
                             index: int) -> list[dict[str, float]]:
    '''
    Loads susceptibility data from a range of sources

    Parameters
    ----------
    sources: dict[str, str]
        Keys are name of sources e.g. functional name\n
        Values are file in which data is stored

    Returns
    -------
    list[dict[str, float]]
    '''

    all_molecules = dict.fromkeys(sources, None)

    for source_name, source_file in sources.items():
        # Load quantum chemical hyperfine data
        _data = np.loadtxt(source_file, skiprows=1)

        all_molecules[source_name] = _data[index]

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
        values are susceptibility component
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

    ax.plot(func_comps.values(), lw=0, marker='x', color='k')

    ax.set_xticks(np.arange(len(func_comps)))
    ax.set_xticklabels(func_comps.keys(), rotation=45)

    ax.set_ylabel(ylabel)
    fig.tight_layout()

    if save:
        plt.savefig(f'{savename}', dpi=500)
    if show:
        plt.show()

    return fig, ax


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
        help='.csv file containing source file information'
    )

    parser.add_argument(
        '-w',
        '--window_append',
        type=str,
        help='Appends to window titles',
        default=''
    )

    uargs = parser.parse_args()

    # Load input file
    config = pd.read_csv(
        uargs.input_file,
        skip_blank_lines=True,
        skipinitialspace=True,
        comment='#'
    )

    # Make table each functional name and chi, r2a, and resid
    table = {}
    for name, folder in zip(config['name'], config['folder']):
        # Load susceptibility data
        _susc = pd.read_csv(
            os.path.join(folder, 'susceptibility_tensor.csv'),
            skip_blank_lines=True,
            skipinitialspace=True,
            comment='#'
        )

        table[name] = {
            'chi_iso': _susc['chi_iso (Å^3)'][0],
            'chi_ax': _susc['chi_ax (Å^3)'][0],
            'chi_rho': _susc['chi_rho (Å^3)'][0],
            'r2_adjusted': _susc['r2_adjusted ()'][0],
            'MAE': _susc['MAE (ppm)'][0]
        }

    # Isotropic parts
    plot_component(
        {name: val['chi_iso'] for name, val in table.items()},
        r'$\chi_\mathregular{iso} \mathregular{(\AA^{3})}$',
        show=False,
        figure_title='isotropic susceptibility ' + uargs.window_append
    )

    # Ax parts
    plot_component(
        {name: val['chi_ax'] for name, val in table.items()},
        r'$\Delta\chi_\mathregular{ax} \mathregular{(\AA^{3})}$',
        show=False,
        figure_title='axial susceptibility ' + uargs.window_append
    )

    # rhombic parts
    plot_component(
        {name: val['chi_rho'] for name, val in table.items()},
        r'$\Delta\chi_\mathregular{rho} \mathregular{(\AA^{3})}$',
        show=False,
        figure_title='rhombic susceptibility ' + uargs.window_append
    )

    plt.show()
