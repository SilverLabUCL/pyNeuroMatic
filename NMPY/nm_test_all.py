#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 10:30:47 2022

@author: jason
"""
import unittest
import nm_object_test
# import nm_object_container_test
import nm_object_mapping_test

if __name__ == '__main__':

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # suite.addTests(loader.loadTestsFromModule(nm_object_test))
    # c = 'nm_object_test.NMObjectTest.'
    # suite.addTests(loader.loadTestsFromName(c+'test00_init'))

    suite.addTests(loader.loadTestsFromModule(nm_object_mapping_test))

    '''
    # suite.addTests(loader.loadTestsFromModule(nm_object_container_test))
    # c = 'nm_object_container_test.NMObjectContainerTest.'
    # suite.addTests(loader.loadTestsFromName(c+'test00_init'))

    suite.addTests(loader.loadTestsFromName(c+'test01_eq_ne'))
    suite.addTests(loader.loadTestsFromName(c+'test02_copy'))
    suite.addTests(loader.loadTestsFromName(c+'test03_parameters'))
    suite.addTests(loader.loadTestsFromName(c+'test04_content_type'))
    suite.addTests(loader.loadTestsFromName(c+'test05_prefix_set'))
    suite.addTests(loader.loadTestsFromName(c+'test06_name_next'))
    suite.addTests(loader.loadTestsFromName(c+'test07_new'))
    suite.addTests(loader.loadTestsFromName(c+'test08_append'))
    suite.addTests(loader.loadTestsFromName(c+'test09_names'))
    suite.addTests(loader.loadTestsFromName(c+'test10_content'))
    suite.addTests(loader.loadTestsFromName(c+'test11_index'))
    suite.addTests(loader.loadTestsFromName(c+'test12_exists'))
    suite.addTests(loader.loadTestsFromName(c+'test13_getitem'))
    suite.addTests(loader.loadTestsFromName(c+'test14_getitems'))
    '''
    
    runner = unittest.TextTestRunner(verbosity=3)
    result = runner.run(suite)
