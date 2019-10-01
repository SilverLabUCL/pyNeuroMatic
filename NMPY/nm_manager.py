# -*- coding: utf-8 -*-
"""
NMPY - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import nm_experiment

exp = nm_experiment.Experiment()
experiment_folder_test(exp, 1)


def main():
    #experiment = nm_experiment.Experiment()
    #experiment_folder_test(experiment, 1)
    print("initialized NM")


def experiment_folder_test(experiment, select):
    if experiment.folder_select is None:
        return False
    if select == 0:
        f = experiment.folder_select.wave_prefix_new(prefix="Record")
    elif select == 1:
        experiment.folder_open_hdf5()
    else:
        return False
    return True


if __name__ == '__main__':
    main()
