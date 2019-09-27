# -*- coding: utf-8 -*-
"""
NMPY - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import nm_experiment


def main():
    experiment = nm_experiment.Experiment()
    experiment_folder_test(experiment, 1)
    print("initialized NM")


def experiment_folder_test(experiment, select):
    if experiment.folder is None:
        return False
    if select == 0:
        experiment.folder.wave_prefix_new(prefix="Record")
    elif select == 1:
        experiment.folder_open_hdf5()
    else:
        return False
    return True


if __name__ == '__main__':
    main()
