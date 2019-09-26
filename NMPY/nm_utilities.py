# -*- coding: utf-8 -*-
"""
NMPY - NeuroMatic in Python
NM utility functions
Copyright 2019 Jason Rothman
"""

def quotes(text: str) -> str:
    return "\"" + text + "\""

def removeSpecialChars(text: str) -> str:
    if text is None or len(text) == 0:
        return ""
    temp = ""
    for c in text:
        if c.isalnum() or c == "_":
            temp += c  # only alpha-numeric or "_"
    return temp
