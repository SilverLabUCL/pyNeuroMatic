# -*- coding: utf-8 -*-
"""
NMPY - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import nm_experiment

def main():
    exp = nm_experiment.Experiment()
    exp.file.wave_prefix_new(prefix="Record")
    print("initialized NM")

if __name__ == '__main__':
    main()
