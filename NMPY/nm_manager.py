# -*- coding: utf-8 -*-
"""
NMPY - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import nm_experiment

nmexp = None

def main():
    nmexp = nm_experiment.Experiment()
    if nmexp.file is not None:
        nmexp.file.wave_prefix_new(prefix="Record")
        nmexp.file_open_hdf5()
    print("initialized NM")

if __name__ == '__main__':
    main()
