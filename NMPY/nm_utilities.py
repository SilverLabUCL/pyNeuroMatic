# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
NM utility functions
Copyright 2019 Jason Rothman
"""
import inspect
from colorama import Fore, Back, Style


def chan_char(chanNum):
    c = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
         'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    if chanNum >= 0 and chanNum < 26:
        return c[chanNum]
    error("channel number must be in the range 0-25")
    return ''
    

def quotes(text):
    if not text:
        text = ""
    return "\"" + text + "\""


def remove_special_chars(text):
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
        return False
    for c in ok:
        name = name.replace(c, "")
    if name.isalnum():
        return True
    return False


def exists(nm_obj_list, name):
    if nm_obj_list and name_ok(name):
        for o in nm_obj_list:
            if name.casefold() == o.name.casefold():
                return True
    return False


def get_names(nm_obj_list):
    nlist = []
    if nm_obj_list:
        for o in nm_obj_list:
            nlist.append(o.name)
    return nlist


def get_items(nm_obj_list, prefix, chan_char=""):
    if nm_obj_list and name_ok(prefix):
        olist = []
        numchar = len(prefix)
        for o in nm_obj_list:
            if prefix.casefold() == o.name[:numchar].casefold():
                if chan_char:
                    if chan_char_exists(o.name[numchar:], chan_char):
                        olist.append(o.name)
                else:
                    olist.append(o.name)
        return olist
    return None


def chan_char_exists(text, chan_char):
    if text and chan_char:
        numchar = len(text)
        for i in reversed(range(numchar)):  # search backwards
            if text[i].isdigit():  # skip thru seq number
                continue
            if text[i] == chan_char:  # first char before seq #
                return True
    return False


def error(text):
    if not text:
        return False
    stack = inspect.stack()
    child = stack_get_class(stack)
    method = stack_get_method(stack)
    print(Fore.RED + "ERROR." + child + "." + method + ": " + text + Style.RESET_ALL)
    return False


def history(text):
    if not text:
        return False
    stack = inspect.stack()
    child = stack_get_class(stack)
    method = stack_get_method(stack)
    print(child + "." + method + ": " + text)
    return True


def stack_get_class(stack, child=True):
    class_tree = str(stack[1][0].f_locals["self"].__class__)
    class_tree = class_tree.replace("<class ", "")
    class_tree = class_tree.replace("\'", "")
    class_tree = class_tree.replace(">", "")
    class_tree = class_tree.split(".")
    class_child = class_tree[0]
    class_parent = class_tree[1]
    if child:
        return class_child
    return class_child + "." + class_parent


def stack_get_method(stack):
    # return inspect.stack()[1][3]
    return stack[1][0].f_code.co_name
