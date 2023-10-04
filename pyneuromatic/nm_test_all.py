#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 10:30:47 2022

@author: jason
"""
import unittest

import nm_channel_test
import nm_data_test
import nm_dataseries_test
import nm_epoch_test
import nm_folder_test
import nm_manager_test
import nm_object_test
import nm_object_container_test
import nm_project_test
import nm_scale_test
import nm_sets_test
import nm_utilities_test


if __name__ == '__main__':

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # c = 'nm_object_test.NMObjectTest.'
    # suite.addTests(loader.loadTestsFromName(c+'test00_init'))

    # suite.addTests(loader.loadTestsFromModule(nm_utilities_test))
    suite.addTests(loader.loadTestsFromModule(nm_object_test))
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
