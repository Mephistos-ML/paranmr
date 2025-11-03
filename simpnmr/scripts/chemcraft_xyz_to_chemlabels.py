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
import xyz_py as xyzp
from simpnmr import utils as ut
import csv


def load_chemcraft_xyz(file_name: str):
    '''
    Loads xyz file containing chemcraft atomic captions\n
    File is same as normal xyz but has an optional 5th column containing\n
    atomic caption/label in double quotes
    '''

    formatting = xyzp.detect_xyz_formatting(file_name)

    # Read labels and coordinates
    try:
        _labels, coords = xyzp.load_xyz(
            file_name,
            missing_headers=formatting['missing_headers'],
            check=False
        )
    except (ValueError, xyzp.XYZError) as vxe:
        ut.red_exit(str(vxe))

    if formatting['atomic_numbers']:
        indexed_labels = xyzp.add_label_indices(xyzp.num_to_lab(_labels))
    else:
        indexed_labels = xyzp.add_label_indices(_labels)

    chem_dict = dict.fromkeys(indexed_labels, '')

    # Read chemical labels
    with open(file_name, 'r') as f:
        for it, line in enumerate(f):
            if it < 2 and not formatting['missing_headers']:
                continue
            spl = line.split()
            if len(spl) > 4:
                chem_dict[indexed_labels[it]] = spl[4].replace('"', '')

    return chem_dict, indexed_labels, coords


def main():
    parser = argparse.ArgumentParser(
        description=(
            'This script converts annotated chemcraft .xyz files into a\n'
            'chemlabels csv file for use with SimpNMR'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'input_file',
        type=str,
        help='.xyz file containing atomic captions in Chemcraft style'
    )

    parser.add_argument(
        '-m',
        '--math_placeholder',
        action='store_true',
        help='Add placeholder column for chem_math_labels'
    )

    uargs = parser.parse_args()

    # Get chemical labels from chemcraft xyz file
    chem_dict, _, _ = load_chemcraft_xyz(uargs.input_file)

    # Remove entries with no chemlabel
    chem_dict = {k: v for k, v in chem_dict.items() if v}

    # Create placeholder math labels
    if uargs.math_placeholder:
        math_dict = {k: f'${v}$' for k, v in chem_dict.items()}

    # Save new chemlabels file
    with open('chemlabels.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if uargs.math_placeholder:
            writer.writerow(['atom_label', 'chem_label', 'chem_math_label'])
        else:
            writer.writerow(['atom_label', 'chem_label'])

        for k, v in chem_dict.items():
            row = [k, v]
            if uargs.math_placeholder:
                row.append(math_dict[k])
            writer.writerow(row)

    ut.cprint('Chemical labels written to\n chemlabels.csv', 'cyan')
