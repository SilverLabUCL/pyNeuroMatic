pyNeuroMatic
==============

<!--
[![PyPI](https://img.shields.io/pypi/v/pyneuromatic)](https://pypi.org/project/pyneuromatic/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyneuromatic)](https://pypi.org/project/pyneuromatic/)
-->
[![GitHub CI](https://github.com/SilverLabUCL/pyNeuroMatic/actions/workflows/ci.yml/badge.svg)](https://github.com/SilverLabUCL/pyNeuroMatic/actions/workflows/ci.yml)
[![GitHub](https://img.shields.io/github/license/SilverLabUCL/pyNeuroMatic)](https://github.com/SilverLabUCL/pyNeuroMatic/blob/master/LICENSE)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/SilverLabUCL/pyNeuroMatic)](https://github.com/SilverLabUCL/pyNeuroMatic/pulls)
[![GitHub issues](https://img.shields.io/github/issues/SilverLabUCL/pyNeuroMatic)](https://github.com/SilverLabUCL/pyNeuroMatic/issues)

Python implementation of [NeuroMatic](https://github.com/SilverLabUCL/NeuroMatic).

pyNeuroMatic is a collection of Python tools for acquiring, analysing and simulating electrophysiological data.

**Note:** This is currently a work in progress. Core data structures are functional, with analysis tools and GUI implementation in development.

Requirements
------------

- **Python**: 3.9 or higher (3.11 recommended)
- **Core dependencies**: numpy, h5py, colorama
- **GUI dependencies** (optional): PyQt6

Installation
------------

### Recommended: Virtual Environment

We strongly recommend using a virtual environment:

```bash
# Create virtual environment
python3 -m venv pyneuromatic_env

# Activate it
source pyneuromatic_env/bin/activate  # On macOS/Linux
# or
pyneuromatic_env\Scripts\activate     # On Windows
```

### Option 1: Install from PyPI (when available)

```bash
# Core functionality only
pip install pyneuromatic

# With GUI support
pip install pyneuromatic[gui]

# For development
pip install pyneuromatic[dev]
```

### Option 2: Install from Source

Clone the repository:

```bash
git clone https://github.com/SilverLabUCL/pyNeuroMatic.git
cd pyNeuroMatic
```

**Core installation** (analysis and data structures, no GUI):

```bash
pip install -e .
```

**With GUI support** (includes PyQt6):

```bash
pip install -e ".[gui]"
```

**For development** (includes testing and formatting tools):

```bash
pip install -e ".[dev]"
```

### Verifying Installation

```python
# Test core functionality
import pyneuromatic as pnm
print(pnm.__version__)

# Test GUI availability (only if installed with [gui])
from pyneuromatic.gui import check_gui_available
check_gui_available()
```

Quick Start
-----------

```python
import pyneuromatic as pnm

# Create a new project
nm = pnm.NMManager()
nm.projects.new('MyProject0')

# Work with the container hierarchy
# Manager > Project > Folder > Data
```

Development Setup
-----------------

For contributors:

```bash
# Clone and checkout development branch
git clone https://github.com/SilverLabUCL/pyNeuroMatic.git
cd pyNeuroMatic
git checkout development

# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

The pre-commit hooks will run automatically before each commit to ensure code quality.

Running Tests
-------------

```bash
# Run all tests
pytest

# Run only core tests (fast, no GUI required)
pytest tests/test_core/ tests/test_analysis/

# Run with coverage
pytest --cov=pyneuromatic --cov-report=html
```

Package Structure
-----------------

```
pyNeuroMatic/
├── pyneuromatic/          # Main package
│   ├── core/             # Core data structures (NMManager, NMProject, NMFolder, NMData)
│   ├── analysis/         # Analysis tools and statistics
│   ├── gui/             # GUI components (optional, requires PyQt6)
│   └── __init__.py      # Package exports
├── tests/                # Test suite
├── examples/             # Example scripts and notebooks (coming soon)
└── docs/                 # Documentation (coming soon)
```

Implemented Features
--------------------

### Core Architecture (✅ Implemented)

The foundational structure of NeuroMatic has been implemented using Python containers (Mutable Mappings). The structural container hierarchy is: **Manager > Project > Folder > Data**.

The hierarchy is currently accessed via a command line interface (CLI) and will also be accessible via a GUI.

**Example:**

```python
import pyneuromatic as pnm

# Create manager
nm = pnm.NMManager()

# Create a new project
nm.projects.new('MyProject0')
```

**Container Classes:**

- `NMObjectContainer` is the parent class of all NM containers
- Each `NMObjectContainer` holds one or more `NMObjects` (NMProject, NMFolder, NMData...)
- Each `NMObjectContainer` contains sets (`NMSets`) for grouping and filtering objects
- `NMSets` can be functions of each other, for example: `Set3 = ['Set1', '&', 'Set2']`
- `NMObjects` contain functions for creating notes and log histories

**Data Series:**

`NMDataSeries` defines data acquired from data acquisition (DAQ) devices, supporting multiple ADC input channels (A, B, C...) and epochs/episodes (E0, E1, E2...).

**Container Hierarchy:**

```
NMManager (nm)
    NMProject (e.g. 'MyProject0')
        NMFolderContainer
            NMFolder (e.g. 'MyFolder0', 'MyFolder1'...)
                NMDataContainer
                    NMData (e.g. 'RecordA0', 'RecordA1'... 'AvgA0', 'AvgB0')
                NMDataSeriesContainer
                    NMDataSeries (e.g. 'Record', 'Avg'...)
                        NMChannelContainer
                            NMChannel ('A', 'B', 'C'...)
                        NMEpochContainer
                            NMEpoch ('E0', 'E1', 'E2'...)
```

### Selection and Execution (✅ Implemented)

**Select Items:**

Each `NMObjectContainer` has one selected item (e.g., `nm.projects.select_key = 'MyProject0'`). The selected items create a unique path through the container tree, accessible via the manager:

```python
nm.select_values  # or nm.select_keys
# Example output: {'project': 'MyProject0', 'folder': 'MyFolder3', 'data': 'RecordA5'}
```

Users can perform tasks on selected items, such as baselining or filtering.

**Execute Items:**

Each `NMObjectContainer` has one 'execute' item. By default, this is the 'select' item, but users can set the execute item to a container key (e.g., `nm.projects.execute_key = 'project0'`) or a `NMSet` (e.g., `nm.projects.execute_key = 'set3'`).

### Analysis Tools

**Stats module** - Compute statistical data measures such as Max, Min, Average, Slope, Rise Time, etc., within predefined time windows.

Planned Features
----------------

The following features are planned for future releases:

### GUI (🔧 In Progress)

**(4)** GUI built with PyQt6 and channel graphs.

**(5)** Demo module/tab - A template for creating user-defined modules.

### Analysis Modules

**(6)** **Main module/tab** - Basic Display, Edit and X-scale data functions, data operations such as Scale and Normalize, and analysis functions such as Average and Sum.

**(8)** **Spike module/tab** - Spike analysis for computing raster plots, peristimulus time (PST) histograms and inter-spike-interval (ISI) histograms. Spike detection using y-threshold level detector on positive or negative slope deflections.

**(9)** **Event module/tab** - Detection of spontaneous events such as excitatory post-synaptic currents (EPSCs). Search algorithms include:
  - Simple level detector
  - Threshold-above-baseline detector (Kudoh and Taguchi 2002)
  - Template-matching detector (Clements and Bekkers 1997)

**(10)** **ROI module/tab** - Fluorescence image region-of-interest (ROI) analysis, including line scans with graphical interface for ROI definition.

**(11)** **Fit module/tab** - Curve fitting tools.

### Data Generation and Acquisition

**(12)** **Pulse module/tab** - Generate waves with pulse waveforms (square, ramp, exponential, alpha, sine, cosine, etc.). Simulate stochastic (binomial) synaptic release and trains of synaptic currents/conductances with short-term plasticity (facilitation/depression).

**(13)** **Art module/tab** - Artifact subtraction.

**(14)** **Clamp module/tab** - Data acquisition with National Instruments (NI) boards.

Troubleshooting
---------------

### GUI Import Errors

If you see:
```
ImportError: GUI dependencies not available
```

Install GUI dependencies:
```bash
pip install -e ".[gui]"
```

### Python Version Issues

PyQt6 requires Python 3.8 or higher. If you're using Python 3.7 or older, please upgrade:

```bash
# Check your Python version
python --version

# Upgrade if needed (example with Homebrew on macOS)
brew install python@3.11
```

Contributing
------------

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

For major changes, please open an issue first to discuss what you would like to change.

License
-------

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Citation
--------

If you use pyNeuroMatic in your research, please cite the original NeuroMatic paper:

Rothman JS and Silver RA (2018) NeuroMatic: An Integrated Open-Source Software Toolkit for Acquisition, Analysis and Simulation of Electrophysiological Data. *Front. Neuroinform.* 12:14. doi: [10.3389/fninf.2018.00014](https://doi.org/10.3389/fninf.2018.00014)

**BibTeX:**

```bibtex
@article{rothman2018neuromatic,
  title={NeuroMatic: An Integrated Open-Source Software Toolkit for Acquisition, Analysis and Simulation of Electrophysiological Data},
  author={Rothman, Jason S and Silver, R Angus},
  journal={Frontiers in Neuroinformatics},
  volume={12},
  pages={14},
  year={2018},
  publisher={Frontiers Media SA},
  doi={10.3389/fninf.2018.00014}
}
```

Acknowledgments
---------------

pyNeuroMatic is a Python implementation of [NeuroMatic](https://github.com/SilverLabUCL/NeuroMatic), originally developed for Igor Pro by Jason Rothman and the Silver Lab at UCL.

Contact
-------

- **Author**: Jason Rothman
- **Email**: j.rothman@ucl.ac.uk
- **Issues**: [GitHub Issues](https://github.com/SilverLabUCL/pyNeuroMatic/issues)
