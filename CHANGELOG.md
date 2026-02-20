# CHANGELOG

<!-- version list -->

## v1.4.0 (2026-02-20)

### Documentation

- **architecture**: Document application-level policy layer
  ([`dc18e3d`](https://gitlab.com/suturina-group/simpnmr/-/commit/dc18e3dfa1c2e00e394d55758db3bc84f5482c8a))

- **user**: Add standalone CLI utilities to user guide index
  ([`dc18e3d`](https://gitlab.com/suturina-group/simpnmr/-/commit/dc18e3dfa1c2e00e394d55758db3bc84f5482c8a))

- **user**: Mark susceptibility format as optional with automatic method selection
  ([`dc18e3d`](https://gitlab.com/suturina-group/simpnmr/-/commit/dc18e3dfa1c2e00e394d55758db3bc84f5482c8a))

### Features

- **policies**: Add susceptibility policy for backend and ORCA method resolution
  ([`dc18e3d`](https://gitlab.com/suturina-group/simpnmr/-/commit/dc18e3dfa1c2e00e394d55758db3bc84f5482c8a))

### Refactoring

- **config**: Make susceptibility format optional and legacy-compatible
  ([`dc18e3d`](https://gitlab.com/suturina-group/simpnmr/-/commit/dc18e3dfa1c2e00e394d55758db3bc84f5482c8a))

- **csv**: Avoid absolute paths in exported CSV comments
  ([`dc18e3d`](https://gitlab.com/suturina-group/simpnmr/-/commit/dc18e3dfa1c2e00e394d55758db3bc84f5482c8a))

- **loaders**: Delegate susceptibility backend and ORCA section resolution to policy
  ([`dc18e3d`](https://gitlab.com/suturina-group/simpnmr/-/commit/dc18e3dfa1c2e00e394d55758db3bc84f5482c8a))

- **pcs_iso**: Make susceptibility method autodetected
  ([`dc18e3d`](https://gitlab.com/suturina-group/simpnmr/-/commit/dc18e3dfa1c2e00e394d55758db3bc84f5482c8a))

- **predict**: Use susceptibility policy for ORCA routing
  ([`dc18e3d`](https://gitlab.com/suturina-group/simpnmr/-/commit/dc18e3dfa1c2e00e394d55758db3bc84f5482c8a))

- **susceptibility**: Autodetect backend and ORCA method via policy layer
  ([`dc18e3d`](https://gitlab.com/suturina-group/simpnmr/-/commit/dc18e3dfa1c2e00e394d55758db3bc84f5482c8a))


## v1.3.5 (2026-02-20)

### Bug Fixes

- **io**: Remove hardcoded linewidth and use relaxation-derived values when available
  ([`96b27da`](https://gitlab.com/suturina-group/simpnmr/-/commit/96b27da175af7c98e9ab5bba6345c599d861c095))

### Chores

- **app**: Delegate CSV comment prefix handling to write_csv_safe
  ([`509ba9d`](https://gitlab.com/suturina-group/simpnmr/-/commit/509ba9d10dc18ae1816452ae5c717935eaafcdc8))

- **app**: Delegate CSV comment prefix handling to write_csv_safe
  ([`f920b11`](https://gitlab.com/suturina-group/simpnmr/-/commit/f920b113a6ade5b25cc9a219051a5c95d2bd7889))

- **csv**: Introduce write_csv_safe for standardized CSV output
  ([`a5c6328`](https://gitlab.com/suturina-group/simpnmr/-/commit/a5c6328a4c4d8a895905c76962993f68a1143e26))

- **io**: Unify CSV exports via write_csv_safe for safe encoding
  ([`5fd6d40`](https://gitlab.com/suturina-group/simpnmr/-/commit/5fd6d404065c6c06708c86460a1d49d9d28aec34))

- **viz**: Add label font size to typography scale
  ([`afa2763`](https://gitlab.com/suturina-group/simpnmr/-/commit/afa276344f0f0ea318452259721ffbc6b8b701bb))

- **viz**: Increase spectrum label sizes and standardize CSV export via write_csv_safe
  ([`45e7472`](https://gitlab.com/suturina-group/simpnmr/-/commit/45e747232b9e04eeb2ba90b665b265198b246334))

- **viz**: Update legacy PNG references to PDF across the repository
  ([`7ef07a7`](https://gitlab.com/suturina-group/simpnmr/-/commit/7ef07a76c8901070427239a0c15ced06f4760be6))

### Refactoring

- **io**: Move write_spectrum helper into csv spec module
  ([`8dafc78`](https://gitlab.com/suturina-group/simpnmr/-/commit/8dafc7803711f5263a45b429de13ee122c6e946a))

### Testing

- **integration**: Add --hide flag to CLI example tests
  ([`1c537ec`](https://gitlab.com/suturina-group/simpnmr/-/commit/1c537ec97a1a1baed4d8a28abbf7073aa4daf9e3))

- **pytest**: Remove addopts marker filtering
  ([`256d640`](https://gitlab.com/suturina-group/simpnmr/-/commit/256d6400531df4c1685f43d6009bb82f665a9275))

- **unit**: Add encoding tests for CSV read/write helpers
  ([`be92d13`](https://gitlab.com/suturina-group/simpnmr/-/commit/be92d131bd682150abbef05f3e4dd351489d54ed))


## v1.3.4 (2026-02-16)

### Bug Fixes

- Trigger patch release
  ([`9439c7b`](https://gitlab.com/suturina-group/simpnmr/-/commit/9439c7b7cef89035310ec1dd1b0cfcfee76a25aa))


## v1.3.3 (2026-02-16)

### Bug Fixes

- **susc**: Correct VT fitting and uncertainty propagation
  ([`e3ce7d1`](https://gitlab.com/suturina-group/simpnmr/-/commit/e3ce7d170a500ae182deac6a95fda3bd29d00e88))


## v1.3.2 (2026-02-10)

### Bug Fixes

- Trigger patch release
  ([`c76aa10`](https://gitlab.com/suturina-group/simpnmr/-/commit/c76aa108bfd7de61676de4a780c4b9c2ab74494b))


## v1.3.1 (2026-02-05)

### Bug Fixes

- **csv**: Fix delimiter handling for raw experiment CSV
  ([`3d4c129`](https://gitlab.com/suturina-group/simpnmr/-/commit/3d4c129efbec704c70ad6859c2b2ab6e664e9453))

### Chores

- **docs**: Minor clarifications and wording improvements
  ([`c975f57`](https://gitlab.com/suturina-group/simpnmr/-/commit/c975f576a955aa903a1a79fbc800dd16c1d25c6f))

- **packaging**: Add missing __init__.py files to package directories
  ([`d139e0a`](https://gitlab.com/suturina-group/simpnmr/-/commit/d139e0a5867982ca8316201ff43bb15ecf040411))


## v1.3.0 (2026-02-01)

### Documentation

- Switch root_doc back to index
  ([`f86a926`](https://gitlab.com/suturina-group/simpnmr/-/commit/f86a926b9c0acc1c9a4688216ee0e7827376afbd))

### Features

- Architectural refactor of application and IO layers
  ([`5291fea`](https://gitlab.com/suturina-group/simpnmr/-/commit/5291fea3abfcac2951cae9ce14050d4500e869fc))


## v1.2.2 (2025-12-04)

### Bug Fixes

- Enable lanthanide support, Correct metadata handling, Add CSV export for fit_susc_corr_time
  ([`d948a49`](https://gitlab.com/suturina-group/simpnmr/-/commit/d948a4906604162ebc9f410d042a0bcca587f936))


## v1.2.1 (2025-12-04)


## v1.2.0 (2025-12-04)

### Bug Fixes

- Relaxation predictions printed
  ([`e761b87`](https://gitlab.com/suturina-group/simpnmr/-/commit/e761b87b7e2acbb1b3127dc8db46ea3991b0a9af))

### Chores

- Delete reduntand files
  ([`083dc93`](https://gitlab.com/suturina-group/simpnmr/-/commit/083dc9367b6d67f405aa60d8509a37922c1b2410))

### Features

- Correlation time fitting at multiple fields or temperatures
  ([`71daba5`](https://gitlab.com/suturina-group/simpnmr/-/commit/71daba583c8a9b28bff2935916fb258635d83ba2))

- Implement additional relaxation mechanism WIP
  ([`d206671`](https://gitlab.com/suturina-group/simpnmr/-/commit/d2066715873e73b43518a16e67c30fec15d50c30))

- Implement additional relaxation mechanism WIP
  ([`328029c`](https://gitlab.com/suturina-group/simpnmr/-/commit/328029c5183fbf1daa746a559cb3bb2784324626))


## v1.1.1 (2025-11-28)

### Bug Fixes

- **version**: Update SimpNMR version to 1.1.1
  ([`c6bef83`](https://gitlab.com/suturina-group/simpnmr/-/commit/c6bef83ce0886c9d614c494c43a8e710f25bf138))


## v1.1.0 (2025-11-28)

### Bug Fixes

- Update plots in visualise.py (plot_isoaxrho), improve chi_plot.py readability, and fix bug in
  readers.py (read_orca_spin) for reading XYZ files
  ([`9b35e85`](https://gitlab.com/suturina-group/simpnmr/-/commit/9b35e8555bc73785071844d66098490c5d4a5ef8))

### Features

- Add chi-frame geometry plotting and support single-point CSV data in chi_plot.py
  ([`c3580c4`](https://gitlab.com/suturina-group/simpnmr/-/commit/c3580c437e8a08262097715e27dbfee77f55798d))

- **core**: Improve HFC and chi inputs and fix CI pipeline
  ([`4a1d285`](https://gitlab.com/suturina-group/simpnmr/-/commit/4a1d285b31a6c10888b56eb865bfbc4e4f467d4b))


## v1.0.8 (2025-11-19)

### Bug Fixes

- Test semantic-release after fetching tags
  ([`764dd74`](https://gitlab.com/suturina-group/simpnmr/-/commit/764dd748d3dc1aa8893c0de7bf1edf0f16f542c7))


## v1.0.7 (2025-11-19)

### Bug Fixes

- Test semantic-release after fetching tags
  ([`057e35c`](https://gitlab.com/suturina-group/simpnmr/-/commit/057e35c1cb7705b3060f09b2074beac5a3e888ca))

- Test semantic-release after fetching tags
  ([`2828a87`](https://gitlab.com/suturina-group/simpnmr/-/commit/2828a87a13316e1b40bb51ad13d5db1729ffd3b2))


## v1.0.6 (2025-11-19)

### Bug Fixes

- Test semantic-release after fetching tags
  ([`362bceb`](https://gitlab.com/suturina-group/simpnmr/-/commit/362bceb07b4e93b62fbffe97823d8cddf55dd32a))


## v1.0.5 (2025-11-19)


## v1.0.4 (2025-11-19)


## v1.0.3 (2025-11-19)


## v1.0.2 (2025-11-19)

### Bug Fixes

- Test semantic-release after fetching tags
  ([`921e62c`](https://gitlab.com/suturina-group/simpnmr/-/commit/921e62ca907f6e35aec1e7a4b89269e11bf35320))


## v1.0.1 (2025-11-19)

### Bug Fixes

- Test semantic-release after fetching tags
  ([`12037ec`](https://gitlab.com/suturina-group/simpnmr/-/commit/12037ec502af7f3754a44fbdca96edab540ffd1c))


## v1.0.0 (2025-11-19)

- Initial Release

## v1.0.0 (2025-11-19)

- Initial Release

## v1.0.0 (2025-11-19)

- Initial Release

## v1.0.0 (2025-11-19)

- Initial Release

## v1.0.0 (2025-11-19)

- Initial Release

## v1.0.0 (2025-11-19)

- Initial Release
