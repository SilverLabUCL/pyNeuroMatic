# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
NM utility functions
Copyright 2019 Jason Rothman
"""
from colorama import Fore, Back, Style

import nm_configs as nmc


def channel_char(chan_num):
    clist = nmc.CHAN_LIST
    if chan_num >= 0 and chan_num < len(clist):
        return clist[chan_num]
    return ''


def channel_num(chan_char):
    if not chan_char:
        return -1
    for i in range(0, len(nmc.CHAN_LIST)):
        if nmc.CHAN_LIST[i].upper() == chan_char.upper():
            return i
    return -1


def chan_char_exists(text, chan_char):
    if not text or not chan_char or len(chan_char) > 1:
        return False
    for i in reversed(range(len(text))):  # search backwards
        if text[i].isdigit():  # skip thru seq number
            continue
        if text[i].upper() == chan_char.upper():  # first char before seq #
            return True
    return False


def quotes(text, single=True):
    if not text:
        text = ''
    if single:
        return "'" + text + "'"
    return '"' + text + '"'


def remove_special_chars(text):
    if not text:
        return ''
    temp = ''
    for c in text:
        if c.isalnum() or c == '_':
            temp += c  # only alpha-numeric or '_'
    return temp


def exists(nm_obj_list, name):
    if not nm_obj_list or not name:
        return False
    for o in nm_obj_list:
        if name.casefold() == o.name.casefold():
            return True
    return False


def get_name_list(nm_obj_list):
    if not nm_obj_list:
        return []
    nlist = []
    for o in nm_obj_list:
        nlist.append(o.name)
    return nlist


def get_items(nm_obj_list, prefix, chan_char=''):
    if not nm_obj_list or not prefix:
        return []
    olist = []
    i = len(prefix)
    for o in nm_obj_list:
        if prefix.casefold() == o.name[:i].casefold():
            if chan_char:
                if chan_char_exists(o.name[i:], chan_char):
                    olist.append(o)
                else:
                    pass
            else:
                olist.append(o)
    return olist


def child_method(stack):
    if not stack:
        return ''
    child = stack_get_class(stack)
    method = stack_get_method(stack)
    return child + '.' + method


def stack_get_class(stack, child=True):
    if not stack:
        return 'None'
    class_tree = str(stack[1][0].f_locals['self'].__class__)
    class_tree = class_tree.replace('<class ', '')
    class_tree = class_tree.replace("\'", '')
    class_tree = class_tree.replace('>', '')
    class_tree = class_tree.split('.')
    class_child = class_tree[0]
    class_parent = class_tree[1]
    if child:
        return class_child
    return class_child + '.' + class_parent


def stack_get_method(stack):
    if not stack:
        return 'None'
    # return inspect.stack()[1][3]
    return stack[1][0].f_code.co_name
