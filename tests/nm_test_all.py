#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 10:30:47 2022

@author: jason
"""
import unittest

import tests.nm_channel_test as nm_channel_test
import tests.nm_data_test as nm_data_test
import tests.nm_dataseries_test as nm_dataseries_test
import tests.nm_epoch_test as nm_epoch_test
import tests.nm_folder_test as nm_folder_test
import tests.nm_manager_test as nm_manager_test
import tests.nm_object_test as nm_object_test
import tests.nm_object_container_test as nm_object_container_test
import tests.nm_project_test as nm_project_test
import tests.nm_scale_test as nm_scale_test
import tests.nm_sets_test as nm_sets_test
import tests.nm_utilities_test as nm_utilities_test


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # c = 'nm_object_test.NMObjectTest.'
    # suite.addTests(loader.loadTestsFromName(c+'test00_init'))

    suite.addTests(loader.loadTestsFromModule(nm_utilities_test))
    # suite.addTests(loader.loadTestsFromModule(nm_object_test))
    # suite.addTests(loader.loadTestsFromModule(nm_object_container_test))
    # suite.addTests(loader.loadTestsFromModule(nm_manager_test))
    # suite.addTests(loader.loadTestsFromModule(nm_scale_test))
    # suite.addTests(loader.loadTestsFromModule(nm_sets_test))
    # suite.addTests(loader.loadTestsFromModule(nm_data_test))
    # suite.addTests(loader.loadTestsFromModule(nm_dataseries_test))
    # suite.addTests(loader.loadTestsFromModule(nm_folder_test))
    # suite.addTests(loader.loadTestsFromModule(nm_project_test))
    # suite.addTests(loader.loadTestsFromModule(nm_channel_test))
    # suite.addTests(loader.loadTestsFromModule(nm_epoch_test))

    runner = unittest.TextTestRunner(verbosity=3)
    result = runner.run(suite)
