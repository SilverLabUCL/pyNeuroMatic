# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
NM utility functions
Copyright 2019 Jason Rothman
"""
import inspect
from colorama import Fore, Back, Style

import nm_configs as nmc


def name_ok(name):
    ok = ['_']  # list of symbols OK to include in names
    if not name:
        return False
    for c in ok:
        name = name.replace(c, '')
    if name.isalnum():
        return True
    return False


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


def get_names(nm_obj_list):
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


def alert(message, red=True, tree=True, quiet=False):
    if quiet:
        return False
    if not message:
        return False
    if tree:
        stack = inspect.stack()
        if stack:
            message = child_method(stack) + ': ' + message
    message = 'ALERT: ' + message
    if red:
        print(Fore.RED + message + Fore.BLACK)
    else:
        print(message)
    return False


def error(message, red=True, tree=True, quiet=False):
    if quiet:
        return False
    if not message:
        return False
    if tree:
        stack = inspect.stack()
        if stack:
            message = child_method(stack) + ': ' + message
    message = 'ERROR: ' + message
    if red:
        print(Fore.RED + message + Fore.BLACK)
    else:
        print(message)
    return False


def history(message, quiet=False):
    if quiet:
        return False
    if not message:
        return False
    stack = inspect.stack()
    if stack:
        message = child_method(stack) + ': ' + message
    print(message)
    return True


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


def input_yesno(question, title='nm', cancel=False):
    if not question:
        return ''
    txt = ''
    if cancel:
        txt = question + '\n(y)es (n)o (c)ancel >> '
        ok = ['y', 'n', 'c']
    else:
        txt = question + '\n(y)es, (n)o >> '
        ok = ['y', 'n']
    if title:
        txt = title + ': ' + txt
    answer = input(txt)
    a = answer[:1].lower()
    if a in ok:
        return a
    return ''
