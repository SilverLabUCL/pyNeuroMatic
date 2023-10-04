PyNeuroMatic
==============

[![GitHub CI](https://github.com/jasonsethrothman/NMPY/actions/workflows/ci.yml/badge.svg)](https://github.com/jasonsethrothman/NMPY/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/NMPY)](https://pypi.org/project/NMPY/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/NMPY)](https://pypi.org/project/NMPY/)
[![GitHub](https://img.shields.io/github/license/jasonsethrothman/NMPY)](https://github.com/jasonsethrothman/NMPY/blob/master/LICENSE)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/jasonsethrothman/NMPY)](https://github.com/jasonsethrothman/NMPY/pulls)
[![GitHub issues](https://img.shields.io/github/issues/jasonsethrothman/NMPY)](https://github.com/jasonsethrothman/NMPY/issues)

Python implementation of [NeuroMatic](https://github.com/SilverLabUCL/NeuroMatic).
This is currently a work in progress.

Installation
------------

### Pip

PyNeuroMatic releases, when made, can be installed with pip (preferably in a [virtual environment](https://docs.python.org/3/tutorial/venv.html)):

    pip install pyneuromatic


### Installation from source

Clone the repository:

    git clone https://github.com/jasonsethrothman/pyneuromatic.git
    cd pyneuromatic

It should be possible to install NMPY using just:

    pip install .

To develop NMPY, you can use the `dev` extra and the `development` branch:

    git clone https://github.com/jasonsethrothman/pyneuromatic.git
    cd pyneuromatic
    git checkout development
    pip install .[dev]

Please use pre-commit to run the pre-commit hooks.
You will need to install the hooks once:

    pre-commit install

They will then run before each commit.
