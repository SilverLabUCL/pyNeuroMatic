# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
NM utility functions
Copyright 2019 Jason Rothman
"""
import inspect


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


def error(text):
    fxn = inspect.stack()[1][3]
    print("ERROR nm." + fxn + " : " + text)


def history(text=""):
    fxn = inspect.stack()[1][3]
    print("nm." + fxn + " : " + text)
