# Changelog

All notable changes to pyNeuroMatic will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GUI dependency checking in `pyneuromatic.gui.__init__.py`
- Optional `[gui]` extra for PyQt6 dependencies
- Separate CI/CD workflows for core and GUI tests
- Main package exports in `pyneuromatic.__init__.py`
- Support for Python 3.9-3.13

### Changed
- Migrated to modern `pyproject.toml` configuration
- GUI is now truly optional (core works without PyQt6)
- Enhanced test infrastructure with separate core/GUI test suites

### Fixed
- Core package can now be imported without GUI dependencies
- Improved error messages when GUI dependencies are missing

## [0.0.1] - Initial Development

### Added
- Core data structure implementation (NMManager, NMProject, NMFolder, NMData)
- Container hierarchy system with NMObjectContainer
- NMSets for grouping and selecting objects
- Basic I/O functionality for electrophysiology data
- Analysis tools for statistical measurements
- Test infrastructure with pytest
- Continuous integration with GitHub Actions
- Pre-commit hooks for code quality
- Code of Conduct and Contributing guidelines

### Project Structure
- `pyneuromatic/core/` - Core data structures and utilities
- `pyneuromatic/analysis/` - Analysis tools and statistics
- `pyneuromatic/gui/` - GUI components (future implementation)
- `tests/` - Comprehensive test suite

### Dependencies
- numpy - Array operations
- h5py - HDF5 file I/O
- colorama - Colored terminal output
- PyQt6 - GUI framework (optional)

---

## Release Notes

### Version 0.0.1 Notes

This is the initial development release of pyNeuroMatic, a Python implementation 
of the NeuroMatic toolkit originally developed for Igor Pro.

**Key Features:**
- Modern Python packaging structure
- Comprehensive data container hierarchy
- Flexible object selection system
- Automated testing and CI/CD
- Optional GUI support

**Known Limitations:**
- GUI implementation is in progress
- Not yet published to PyPI
- Documentation is minimal
- Some advanced features from Igor version not yet implemented

**Installation:**
```bash
# Core functionality
pip install -e .

# With GUI support
pip install -e ".[gui]"

# For development
pip install -e ".[dev]"
```

**Migration from Igor:**
The Python implementation maintains conceptual compatibility with the Igor 
version but uses Python idioms and modern packaging practices.

---

## Future Plans

### Version 0.1.0 (Planned)
- [ ] Complete GUI implementation with PyQt6
- [ ] Channel visualization and plotting
- [ ] Main analysis module/tab
- [ ] Documentation with Sphinx
- [ ] Example notebooks and scripts
- [ ] Publish to PyPI

### Version 0.2.0 (Planned)
- [ ] Spike module for raster plots and histograms
- [ ] Event detection module
- [ ] Enhanced data acquisition support
- [ ] ROI analysis for fluorescence imaging
- [ ] Curve fitting module

### Long-term Goals
- [ ] Real-time data acquisition with National Instruments hardware
- [ ] Advanced pulse generation
- [ ] Artifact subtraction
- [ ] Short-term plasticity simulations
- [ ] Comprehensive documentation and tutorials
- [ ] Active community contributions

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.
