# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
NM utility functions
Copyright 2019 Jason Rothman
"""
import math
import inspect
from colorama import Fore, Back, Style

import nm_preferences as nmp


def name_ok(name):
    ok = ['_']  # list of symbols OK to include in names
    if isinstance(name, list):
        if not name:
            return False
    else:
        name = [name]
    for n in name:
        if not isinstance(n, str):
            return False
        if len(n) == 0:
            continue  # empty string is OK
        for c in ok:
            n = n.replace(c, '')
        if len(n) == 0:
            return False
        if not n.isalnum():
            return False
    return True


def number_ok(number, only_integer=False, no_boolean=True, no_inf=True,
              no_nan=True, no_neg=False, no_pos=False, no_zero=False):
    only_integer = check_bool(only_integer, False)
    no_boolean = check_bool(no_boolean, True)
    no_inf = check_bool(no_inf, True)
    no_nan = check_bool(no_nan, True)
    no_neg = check_bool(no_neg, False)
    no_pos = check_bool(no_pos, False)
    no_zero = check_bool(no_zero, False)
    if isinstance(number, list):
        if not number:
            return False
    else:
        number = [number]
    for i in number:
        if only_integer and not isinstance(i, int):
            return False
        if no_boolean and isinstance(i, bool):
            return False
        if not isinstance(i, int) and not isinstance(i, float):
            return False
        if no_inf and math.isinf(i):
            return False
        if no_nan and math.isnan(i):
            return False
        if no_neg and i < 0:
            return False
        if no_pos and i > 0:
            return False
        if no_zero and i == 0:
            return False
    return True


def quotes(text, single=True):
    single = check_bool(single, True)
    if not isinstance(text, str):
        text = str(text)
    if single:
        return "'" + str(text) + "'"
    return '"' + str(text) + '"'


def remove_special_chars(text):
    if not text or not isinstance(text, str):
        return ''
    temp = ''
    for c in text:
        if c.isalnum() or c == '_':
            temp += c  # only alpha-numeric or '_'
    return temp


def int_list_to_seq_str(int_list, space=True):
    # e.g. [0,1,5,6,7,12,19,20,21,22,24] -> 0,1,5-7,12,19-22,24
    space = check_bool(space, True)
    if not int_list or not isinstance(int_list, list):
        return ''
    for i in int_list:
        if not isinstance(i, int):
            return ''
    if len(int_list) == 1:
        return str(int_list[0])
    imin = min(int_list)
    imax = max(int_list)
    seq_started = False
    sfirst = int_list[0]
    slist = []
    for i in range(imin, imax + 1):  # [0,1,2,3,4,5,6,7,8]
        if i in int_list:
            if seq_started:
                if i == imax:  # last integer
                    if sfirst + 1 == i:
                        slist.append(str(sfirst))
                        slist.append(str(i))
                    else:
                        slist.append(str(sfirst) + '-' + str(i))
            else:
                if i == imax:  # last integer
                    slist.append(str(i))
                else:
                    sfirst = i
                    seq_started = True
        elif seq_started:
            slast = i - 1
            if sfirst == slast:
                slist.append(str(sfirst))
            elif sfirst + 1 == slast:
                slist.append(str(sfirst))
                slist.append(str(slast))
            else:
                slist.append(str(sfirst) + '-' + str(slast))
            seq_started = False
    if space:
        return ', '.join(slist)
    return ','.join(slist)


def channel_char(chan_num):
    if not isinstance(chan_num, int) and not isinstance(chan_num, float):
        return ''
    if not number_ok(chan_num, no_neg=True):
        return ''
    clist = nmp.CHAN_LIST
    if chan_num >= 0 and chan_num < len(clist):
        return clist[chan_num]
    return ''


def channel_num(chan_char):
    if not chan_char or not isinstance(chan_char, str) or len(chan_char) > 1:
        return -1
    for i, c in enumerate(nmp.CHAN_LIST):
        if chan_char.lower() == c.lower():
            return i
    return -1


def channel_char_exists(text, chan_char):
    if not text or not isinstance(text, str):
        return False
    if not chan_char or not isinstance(chan_char, str) or len(chan_char) > 1:
        return False
    for i in reversed(range(len(text))):  # search backwards, chan char at end
        if text[i].isdigit():  # skip thru seq number
            continue
        if text[i].lower() == chan_char.lower():  # first char before seq #
            return True
        return False  # check only last character
    return False


def check_bool(var_bool, default_value):
    if not isinstance(var_bool, bool):
        return default_value
    return var_bool


def type_error(object, varType):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    var_names = [var_name for var_name, var_val in callers_local_vars if
                 var_val is object]
    count = len(var_names)  # may be multiple variables with same value
    if count > 0:
        vname = var_names[0]  # take first name
    else:
        vname = 'FAILEDTOFINDVARIABLENAME'
    got = str(type(object))
    got = got.replace('<class ', '').replace('>', '').replace("'", "")
    return 'bad ' + vname + ': expected ' + varType + ', but got ' + got


def history_change(var_name, from_value, to_value):
    if isinstance(from_value, str):
        old = from_value
    else:
        old = str(from_value)
    if isinstance(to_value, str):
        new = to_value
    else:
        new = str(to_value)
    return ('changed ' + var_name + ' from ' + quotes(old) + ' to ' +
            quotes(new))


def history(message, title='', tp='', frame=1, red=False, quiet=False):
    red = check_bool(red, False)
    quiet = check_bool(quiet, False)
    if tp.lower() == 'none':
        path = ''
    else:
        path = get_treepath(inspect.stack(), tp=tp, frame=frame)
    if path:
        h = path + ': ' + message
    else:
        h = message
    if title:
        h = title + ': ' + h
    if not quiet:
        if red:
            print(Fore.RED + h + Fore.BLACK)
        else:
            print(h)
    return h


def get_treepath(stack, tp='', frame=1):
    if not stack:
        return ''
    method = get_method(stack, frame=frame)
    if not tp:
        tp = get_class(stack, frame=frame)
    path = ['nm']
    if tp:
        path.append(tp)
    if method:
        path.append(method)
    return '.'.join(path)


def get_class(stack, frame=1, module=False):
    module = check_bool(module, False)
    if len(stack) <= frame or len(stack[0]) == 0:
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
    m = class_tree[0]
    c = class_tree[1]
    if module:
        return m + '.' + c
    return c


def get_method(stack, frame=1):
    if len(stack) <= frame or len(stack[0]) == 0:
        return ''
    f = stack[frame][0]
    if not inspect.isframe(f):
        return ''
    return f.f_code.co_name


def input_yesno(prompt, title='', tp='', frame=1, cancel=False):
    cancel = check_bool(cancel, False)
    if not prompt:
        return ''
    if cancel:
        txt = prompt + '\n' + '(y)es (n)o (c)ancel: '
        ok = ['y', 'yes', 'n', 'no', 'c', 'cancel']
    else:
        txt = prompt + '\n' + '(y)es, (n)o: '
        ok = ['y', 'yes', 'n', 'no']
    if tp.lower() == 'none':
        path = ''
    else:
        path = get_treepath(inspect.stack(), tp=tp, frame=frame)
    if path:
        txt = path + ': ' + txt
    if title:
        txt = title + ': ' + txt
    answer = input(txt)
    a = answer.lower()
    if a in ok:
        return a[:1]
    return ''
