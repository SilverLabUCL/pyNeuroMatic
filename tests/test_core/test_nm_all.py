#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 10:30:47 2022

@author: jason
"""
import unittest

import tests.test_core.test_nm_channel as test_nm_channel
import tests.test_core.test_nm_data as test_nm_data
import tests.test_core.test_nm_dataseries as test_nm_dataseries
import tests.test_core.test_nm_dimension as test_nm_dimension
import tests.test_core.test_nm_epoch as test_nm_epoch
import tests.test_core.test_nm_folder as test_nm_folder
import tests.test_core.test_nm_manager as test_nm_manager
import tests.test_core.test_nm_object as test_nm_object
import tests.test_core.test_nm_object_container as test_nm_object_container
import tests.test_core.test_nm_project as test_nm_project
import tests.test_core.test_nm_sets as test_nm_sets
import tests.test_core.test_nm_utilities as test_nm_utilities


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # c = 'nm_object_test.NMObjectTest.'
    # suite.addTests(loader.loadTestsFromName(c+'test00_init'))

    suite.addTests(loader.loadTestsFromModule(test_nm_utilities))
    # suite.addTests(loader.loadTestsFromModule(nm_object_test))
    # suite.addTests(loader.loadTestsFromModule(nm_object_container_test))
    # suite.addTests(loader.loadTestsFromModule(nm_manager_test))
    # suite.addTests(loader.loadTestsFromModule(nm_dimension_test))
    # suite.addTests(loader.loadTestsFromModule(nm_sets_test))
    # suite.addTests(loader.loadTestsFromModule(nm_data_test))
    # suite.addTests(loader.loadTestsFromModule(nm_dataseries_test))
    # suite.addTests(loader.loadTestsFromModule(nm_folder_test))
    # suite.addTests(loader.loadTestsFromModule(nm_project_test))
    # suite.addTests(loader.loadTestsFromModule(test_nm_channel))
    # suite.addTests(loader.loadTestsFromModule(nm_epoch_test))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
