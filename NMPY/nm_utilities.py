# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
NM utility functions
Copyright 2019 Jason Rothman
"""
import inspect


def quotes(text):
    if not text:
        text = ""
    return "\"" + text + "\""


def removeSpecialChars(text):
    if not text:
        return ""
    temp = ""
    for c in text:
        if c.isalnum() or c == "_":
            temp += c  # only alpha-numeric or "_"
    return temp


def name_ok(name, alert=True):
    ok = ["_"]  # list of symbols OK to include in names
    if not name:
        if alert:
            error("encountered empty name")
        return False
    for c in ok:
        name = name.replace(c, "")
    if name.isalnum():
        return True
    elif alert:
        error("bad name " + quotes(name))
    return False


def name_list(objlist):
    nlist = []
    if objlist:
        for o in objlist:
            nlist.append(o.name)
    return nlist


def exists(objlist, name):
    if objlist and name_ok(name):
        for o in objlist:
            if name.casefold() == o.name.casefold():
                return True
    return False


def error(text):
    if not text:
        return False
    fxn = inspect.stack()[1][3]
    print("ERROR nm." + fxn + " : " + text)
    return True


def history(text):
    if not text:
        return False
    fxn = inspect.stack()[1][3]
    print("nm." + fxn + " : " + text)
    return True
