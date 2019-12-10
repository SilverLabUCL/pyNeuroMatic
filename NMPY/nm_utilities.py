# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
NM utility functions
Copyright 2019 Jason Rothman
"""
import math
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


def num_ok(number, no_inf=True, no_nan=True, no_neg=False, no_pos=False,
           no_zero=False):
    if no_inf and math.isinf(number):
        return False
    if no_nan and math.isnan(number):
        return False
    if no_neg and number < 0:
        return False
    if no_pos and number > 0:
        return False
    if no_zero and number == 0:
        return False
    return True


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


def alert(message, red=True, tree=True, quiet=False, frame=1):
    if quiet:
        return False
    if not message:
        return False
    if tree:
        stack = inspect.stack()
        if stack:
            message = get_class_method(stack, frame=frame) + ': ' + message
    message = 'ALERT: ' + message
    if red:
        print(Fore.RED + message + Fore.BLACK)
    else:
        print(message)
    return False


def error(message, red=True, tree=True, quiet=False, frame=1):
    if quiet:
        return False
    if not message:
        return False
    if tree:
        stack = inspect.stack()
        if stack:
            message = get_class_method(stack, frame=frame) + ': ' + message
    message = 'ERROR: ' + message
    if red:
        print(Fore.RED + message + Fore.BLACK)
    else:
        print(message)
    return False


def history(message, quiet=False, frame=1):
    if quiet:
        return False
    if not message:
        return False
    stack = inspect.stack()
    if stack:
        message = get_class_method(stack, frame=frame) + ': ' + message
    print(message)
    return True


def get_class_method(stack, nm=True, parent=False, child=True, frame=1):
    if not stack:
        return ''
    cm = []
    if nm:
        cm.append('NM')
    child = get_class(stack, parent=parent, child=child, frame=frame)
    method = get_method(stack, frame=frame)
    if child:
        cm.append(child)
    if method:
        cm.append(method)
    return '.'.join(cm)


def get_class(stack, parent=True, child=True, frame=1):
    if len(stack) <= 1 or len(stack[0]) == 0:
        return ''
    f = stack[frame][0]
    if not inspect.isframe(f):
        return ''
    if 'self' not in f.f_locals:
        return ''
    class_tree = str(stack[frame][0].f_locals['self'].__class__)
    class_tree = class_tree.replace('<class ', '')
    class_tree = class_tree.replace("\'", '')
    class_tree = class_tree.replace('>', '')
    class_tree = class_tree.split('.')
    class_parent = class_tree[0]
    class_child = class_tree[1]
    if parent and child:
        return class_parent + '.' + class_child
    if parent:
        return class_parent
    if child:
        return class_child
    return ''


def get_method(stack, frame=1):
    if len(stack) <= 1 or len(stack[0]) == 0:
        return ''
    if not inspect.isframe(stack[frame][0]):
        return ''
    return stack[frame][0].f_code.co_name


def input_yesno(question, title='nm', cancel=False):
    if not question:
        return ''
    txt = ''
    if cancel:
        txt = question + '\n(y)es (n)o (c)ancel: '
        ok = ['y', 'n', 'c']
    else:
        txt = question + '\n(y)es, (n)o: '
        ok = ['y', 'n']
    if title:
        txt = title + ': ' + txt
    answer = input(txt)
    a = answer[:1].lower()
    if a in ok:
        return a
    return ''


def input_default(prompt, default=''):
    if default:
        txt = prompt + ' [' + default + ']: '
    else:
        txt = prompt + ':'
    a = input(txt) or default
    return a
