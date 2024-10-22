import setuptools

# with open('README.md', 'r', encoding='utf-8') as fh:
#     long_description = fh.read()

# DO NOT EDIT THIS NUMBER!
# IT IS AUTOMATICALLY CHANGED BY python-semantic-release
__version__ = '0.0.0'

setuptools.setup(
    name='simple_pnmr',
    version=__version__,
    author='Suturina Group',
    author_email='',
    description='A package for fitting ', # noqa
    url='https://github.com/jonkragskow/simple_pnmr',
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
        'adjustText'
    ],
    entry_points={
        'console_scripts': [
            'pnmr = simple_pnmr.cli:interface',
            'plot_A_funcs = simple_pnmr.scripts.batch_hf_plot:main',
            'plot_chi_funcs = simple_pnmr.scripts.batch_susc_plot:main',
            'xyz_to_chemlabel = simple_pnmr.scripts.chemcraft_xyz_to_chemlabels:main' # noqa
        ]
    }
)
