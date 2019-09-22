# -*- coding: utf-8 -*-
"""
Main.py -  Defines NM Main class
Copyright 2019 Jason Rothman 
"""


import os, sys, gc

class Main:
    """NM Main class"""
    i = 12345

    def __init__(self, configFile=None, argv=None):
        self.gui = None


