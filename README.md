# ParaNMR

> Fork of the original [suturina-group/simpnmr](https://gitlab.com/suturina-group/simpnmr). The original work is maintained by Suturina Group; this repository is a fork under the new `paranmr` name.

[![Docs](https://img.shields.io/badge/docs-paranmr.org-blue)](https://paranmr.org/)
[![PyPI](https://img.shields.io/pypi/v/paranmr.svg)](https://pypi.org/project/paranmr/)
[![License](https://img.shields.io/badge/license-GPL--3.0--or--later-green)](LICENSE)

**ParaNMR** is an open-source Python toolkit for prediction, fitting, and analysis
of paramagnetic NMR spectra based on experimental data and ab initio calculations.

## Features

- Prediction of paramagnetic NMR shifts and spectra
- Susceptibility tensor fitting from experimental measurements
- Integration with quantum chemistry outputs (ORCA, Gaussian, Molcas)

## Installation

```bash
pip install paranmr
paranmr --help
```

## Quick example

```bash
paranmr predict input.yml
```

## Documentation

👉 https://paranmr.org

## Project status

Active development. Public APIs and configuration schemas are considered stable;
breaking changes are coordinated with maintainers and documented in the changelog.

## Support

Please use GitHub Issues for bug reports and feature requests.

## Contributing

Development guidelines, architecture, and release policies are documented in the
Developer Guide:

👉 https://paranmr.org/developer_guide/

## Citation

If you use ParaNMR in academic work, please cite the software and the specific
version used. A DOI record will be added in the future.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
