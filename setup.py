import setuptools

# with open('README.md', 'r', encoding='utf-8') as fh:
#     long_description = fh.read()

# DO NOT EDIT THIS NUMBER!
# IT IS AUTOMATICALLY CHANGED BY python-semantic-release
__version__ = '0.0.0'

setuptools.setup(
    name='SimpNMR',
    version=__version__,
    author='Suturina Group',
    author_email='',
    description='A package for fitting ', # noqa
    url='https://gitlab.com/suturina-group/simpnmr',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    package_dir={'': '.'},
    packages=setuptools.find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'xyz_py>=5.13.0',
        'pandas',
        'pathos',
        'pyyaml',
        'pyyaml-include',
        'adjustText',
        'extto >= 0.3.0'
    ],
    entry_points={
        'console_scripts': [
            'simpnmr = simpnmr.cli:interface',
            'plot_A_funcs = simpnmr.scripts.batch_hf_plot:main',
            'plot_chi_funcs = simpnmr.scripts.batch_susc_plot:main',
            'chi_plot = simpnmr.scripts.chi_plot:main',
            'get_susc = simpnmr.scripts.get_susc:main',
            'xyz_to_chemlabel = simpnmr.scripts.chemcraft_xyz_to_chemlabels:main' # noqa,
        ]
    }
)
