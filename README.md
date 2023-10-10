pyNeuroMatic
==============

[![GitHub CI](https://github.com/jasonsethrothman/NMPY/actions/workflows/ci.yml/badge.svg)](https://github.com/jasonsethrothman/NMPY/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pyneuromatic)](https://pypi.org/project/pyneuromatic/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyneuromatic)](https://pypi.org/project/pyneuromatic/)
[![GitHub](https://img.shields.io/github/license/jasonsethrothman/NMPY)](https://github.com/jasonsethrothman/NMPY/blob/master/LICENSE)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/jasonsethrothman/NMPY)](https://github.com/jasonsethrothman/NMPY/pulls)
[![GitHub issues](https://img.shields.io/github/issues/jasonsethrothman/NMPY)](https://github.com/jasonsethrothman/NMPY/issues)

Python implementation of [NeuroMatic](https://github.com/SilverLabUCL/NeuroMatic).
This is currently a work in progress.

pyNeuroMatic is a collection of Python tools for acquiring, analysing and simulating electrophysiological data.

Installation
------------

### Pip

pyNeuroMatic releases, when made, can be installed with pip (preferably in a [virtual environment](https://docs.python.org/3/tutorial/venv.html)):

    pip install pyneuromatic


### Installation from source

Clone the repository:

    git clone https://github.com/jasonsethrothman/pyneuromatic.git
    cd 

It should be possible to install NMPY using just:

    pip install .

To develop NMPY, you can use the `dev` extra and the `development` branch:

    git clone https://github.com/jasonsethrothman/.git
    cd 
    git checkout development
    pip install .[dev]

Please use pre-commit to run the pre-commit hooks.
You will need to install the hooks once:

    pre-commit install

They will then run before each commit.

Current/Future Features
-----------------------

(1) The foundational structure of NeuroMatic (NM) has been created using Python containers (Mutable Mappings). NM's structural container hierarchy is as follows: Project > Folder > Data. The hierarchy is currently accessed via a command line interface (CLI) but will also be accessible via a GUI. Here is a command to create a new project:

    In [1]: nm.projects.new('MyProject0')

NMObjectContainer is the parent class of all NM containers (NMProjectContainer, NMFolderContainer, NMDataContainer...). Each NMObjectContainer holds one or more NMObjects (NMProject, NMFolder, NMData...).

Each NMObjectContainer also contains sets (NMSets). NMObjects can be placed in NMSets and these NMSets can then be used to specify what projects, folders and data is to be analysed. NMSets can be functions of each other (.e.g. 'Set3' = ['Set1', '&', 'Set2'])

NMObjects contain functions for creating notes and log histories.

A NMDataSeries define data acquired from data acquisition (DAQ) devices, allowing multiple ADC input channels (A, B, C...) and epochs/episodes (E0, E1, E2...).

NM's container tree hierarchy is as follows:

    NMManager (nm)
        NMProjectContainer
            NMProject (e.g. 'MyProject0', 'MyProject1'...)
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

(2) NM container 'select' items. Each NMObjectContainer has one selected item (e.g. nm.projects.select_key = 'MyProject0'). The selected items in NM's container tree create a unique path through the tree. The selected items are accessible via NM's manager (nm): nm.select_values or nm.select_keys. Here is an example of printing nm.select_keys via CLI:

    {'project': 'MyProject0', 'folder': 'MyFolder3', 'data': 'RecordA5'}.

In the future, users will be able to instruct NM to perform a given task on the selected items, such as baselining or filtering.

(3) NM container 'execute' items. Each NMObjectContainer has one 'execute' item. By default the execute item is the 'select' item, just described (2). However, users have the option to set the 'execute' item to a NM container key (e.g. nm.projects.execute_key = 'project0') or a NMSet (e.g. nm.projects.execute_key = 'set3').

(4) GUI and channel graphs.

(5) Demo module/tab. A simple module/tab that provides a template for creating a user-defined module/tab.

(6) Main module/tab. NM's main module/tab that performs basic Display, Edit and X-scale data functions, basic data operations such as Scale and Normalize, and data analysis functions such as Avarage and Sum.

(7) Stats module/tab. Compute statistical data measures such as Max, Min, Average, Slope, Rise Time, etc, within any number of predefined time windows.

(8) Spike module/tab. Spike analysis module/tab for computing raster plots, peristimulus time (PST) histograms and inter-spike-interval (ISI) histograms. Spike occurrences are determined by a y-threshold level detector on positive or negative slope deflections.

(9) Event module/tab. Module/tab for detection of spontaneous events such as excitory post-synaptic currents (EPSCs). The search algorithm can be either a simple level detector, a threshold-above-baseline detector similar to that described by Kudoh and Taguchi 2002, or a template-matching detector as described by Clements and Bekkers 1997. 

(10) ROI module/tab. Florescence image region-of-interest (ROI) analysis, including line scans. Users will be able to define ROIs using a graphical interface.

(11) Fit module/tab. Curve fitting.

(12) Pulse module/tab. Generate waves with added pulse waveforms such as a square, ramp, exponential, alpha, sine, cosine, etc. Simulate stochastic (binomial) synaptic release using synaptic-like exponential waveforms. Simulate trains of synaptic currents/conductances that exhibit short-term plasticity (i.e. facilitation and/or depression).

(13) Art module/tab. Artifact subtraction.

(14) Clamp module/tab. Data acquisition. National Instruments (NI) boards.
