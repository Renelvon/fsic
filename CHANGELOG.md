# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.3.1] - 2020-27-01
### Added
* Distribute using [wheels](https://pythonwheels.com/).

### Changed
* Updated some notebooks.
* Renamed notebooks folder to `tutorial`.
* Refined some dependencies.


## [0.3.0] - 2020-25-01
### Changed
- Updated and re-enabled Travis CI.
- Updated `README.md` and other parts of the repo to use and mention Python 3.
* Fixed some issues detected by Pylint.

### Removed
- Various unused functions and methods.
- Modules `glo` and `plot`.


## [0.2.0] - 2020-25-01
### Added
- Lint the test suite explicitly.
- `black` as code formatter.
- `pyproject.toml` to store tool configuration (except `pylint`).

### Changed
- Port the repository to Python 3 (only).
- Lint package more.

### Removed
- Various pieces of commented-out code.


## [0.1.1] - 2020-24-01
### Added
- Linting via [pylint](https://www.pylint.org/).
- Package layout checking via [pyroma](https://pypi.org/project/pyroma/).
- Source distribution completeness checking via [check-manifest](https://pypi.org/project/check-manifest/).

### Changed
- `print` statements to invocations of `print()`.
- Format repository using black
- Minor code refactorings.
- Euclidean division now explicitly uses the `//` operator.
- Moved `tests` to toplevel.

### Removed
- Traces of Python2-only features, like `xrange` and non-iterating `range`.


## [0.1.0] - 2020-23-01
### Added
- AUTHORS.md to make attribution easy; @renelvon as maintainer.
- Makefile to automate common targets.
- MANIFEST.in to enable proper source distribution.
- Setup script and configuration.
- This CHANGELOG.

### Changed
- Minimum versions of all required packages.

### Removed
- Unittest bash script; it is a Makefile target now.
