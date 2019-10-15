# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
NM utility functions
Copyright 2019 Jason Rothman
"""
import inspect
from colorama import Fore, Back, Style
import nm_configs as nmconfig


def channel_char(chan_num):
    clist = nmconfig.CHAN_LIST
    if chan_num >= 0 and chan_num < len(clist):
        return clist[chan_num]
    return ''


def channel_num(chan_char):
    clist = nmconfig.CHAN_LIST
    try:
        chan_num = clist.index(chan_char)
        return chan_num
    except ValueError:
        return -1


def quotes(text, single=True):
    if not text:
        text = ""
    if single:
        return "'" + text + "'"
    return "\"" + text + "\""


def remove_special_chars(text):
    if not text:
        return ""
    temp = ""
    for c in text:
        if c.isalnum() or c == "_":
            temp += c  # only alpha-numeric or "_"
    return temp


def name_ok(name):
    ok = ["_"]  # list of symbols OK to include in names
    if not name:
        return False
    for c in ok:
        name = name.replace(c, "")
    if name.isalnum():
        return True
    return False


def exists(nm_obj_list, name):
    if not nm_obj_list or not name_ok(name):
        return False
    for o in nm_obj_list:
        if name.casefold() == o.name.casefold():
            return True
    return False


def get_names(nm_obj_list):
    if not nm_obj_list:
        return []
    nlist = []
    for o in nm_obj_list:
        nlist.append(o.name)
    return nlist


def get_items(nm_obj_list, prefix, chan_char=""):
    if not nm_obj_list or not name_ok(prefix) or not chan_char:
        return []
    olist = []
    numchar = len(prefix)
    for o in nm_obj_list:
        if prefix.casefold() == o.name[:numchar].casefold():
            if chan_char:
                if chan_char_exists(o.name[numchar:], chan_char):
                    olist.append(o)
                else:
                    pass
            else:
                olist.append(o)
    return olist


def chan_char_exists(text, chan_char):
    if not text or not chan_char or len(chan_char) > 1:
        return False
    numchar = len(text)
    for i in reversed(range(numchar)):  # search backwards
        if text[i].isdigit():  # skip thru seq number
            continue
        if text[i].casefold() == chan_char.casefold():  # first char before seq #
            return True
    return False


def alert(text):
    if not text:
        return False
    stack = inspect.stack()
    child = stack_get_class(stack)
    method = stack_get_method(stack)
    print("ALERT." + child + "." + method +
          ": " + text)
    return False


def error(text):
    if not text:
        return False
    stack = inspect.stack()
    child = stack_get_class(stack)
    method = stack_get_method(stack)
    print(Fore.RED + "ERROR." + child + "." + method +
          ": " + text + Fore.BLACK)
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
    if not stack:
        return "None"
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
    if not stack:
        return "None"
    # return inspect.stack()[1][3]
    return stack[1][0].f_code.co_name
