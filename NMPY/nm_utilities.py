# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
NM utility functions
Copyright 2019 Jason Rothman
"""


def quotes(text: str) -> str:
    return "\"" + text + "\""


def removeSpecialChars(text: str) -> str:
    if text is None or not text:
        return ""
    temp = ""
    for c in text:
        if c.isalnum() or c == "_":
            temp += c  # only alpha-numeric or "_"
    return temp


def name_ok(name: str) -> str:
    ok = ["_"]  # symbols that are OK to include in names
    if name is None or not name:
        return False
    for c in ok:
        name = name.replace(c, "")
    return name.isalnum()


def error(error=0, text=""):
    print("NM error: " + text)
