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


def name_ok(  # check if name is OK to use
    name_or_listofnames  # string name to check
):
    ok = ['_']  # list of symbols OK to include in names
    if isinstance(name_or_listofnames, list):
        if not name_or_listofnames:  # empty list
            return False
        listofnames = name_or_listofnames
    else:
        listofnames = [name_or_listofnames]
    for name in listofnames:
        if not isinstance(name, str):
            return False
        if len(name) == 0:
            continue  # empty string is OK
        for c in ok:
            name = name.replace(c, '')
        if len(name) == 0:
            return False
        if not name.isalnum():
            return False
    return True


def number_ok(  # check if number is OK to use
    number_or_listofnumbers,  # the number to check
    only_integer=False,  # numbers must be integer type
    no_boolean=True,  # numbers must be boolean type
    no_inf=True,  # infinity is not OK
    no_nan=True,  # NaN is not OK
    no_neg=False,  # negative numbers are not OK
    no_pos=False,  # positive numbers are not OK
    no_zero=False  # 0 is not OK
):
    only_integer = bool_check(only_integer, False)
    no_boolean = bool_check(no_boolean, True)
    no_inf = bool_check(no_inf, True)
    no_nan = bool_check(no_nan, True)
    no_neg = bool_check(no_neg, False)
    no_pos = bool_check(no_pos, False)
    no_zero = bool_check(no_zero, False)
    if isinstance(number_or_listofnumbers, list):
        if not number_or_listofnumbers:  # empty list
            return False
        listofnumbers = number_or_listofnumbers
    else:
        listofnumbers = [number_or_listofnumbers]
    for i in listofnumbers:
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


def bool_check(  # check boolean value is OK
    bool_value,  # the boolean value
    default_value  # default value to return if bool-value is not boolean
):
    if not isinstance(bool_value, bool):
        if isinstance(default_value, bool):
            return default_value
        else:
            e = ("ERROR: nm.Utilities.bool_check: bad bool_value: " +
                 "expected boolean but got " + str(type(default_value)))
            raise TypeError(e)  # raise error since default is bad
    return bool_value


def quotes(  # add string quotes around text
    text,  # the string text
    single=True  # True: single quotes, False: double quotes
):
    single = bool_check(single, True)
    if not isinstance(text, str):
        text = str(text)
    if single:
        return "'" + str(text) + "'"
    return '"' + str(text) + '"'


def remove_special_chars(  # remove non-alpha-numeric characters
    text  # the string text
):
    if not text or not isinstance(text, str):
        return ''
    temp = ''
    for c in text:
        if c.isalnum() or c == '_':
            temp += c  # only alpha-numeric or '_'
    return temp


def int_list_to_seq_str(  # convert list of integers to a sequence string
    int_list,  # list of integers
    space=True  # True: add space after comma, False: no space
):
    # e.g. [0,1,5,6,7,12,19,20,21,22,24] -> 0,1,5-7,12,19-22,24
    space = bool_check(space, True)
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


def chan_char(  # convert channel number to character
    chan_num  # channel number
):
    if not isinstance(chan_num, int) and not isinstance(chan_num, float):
        return ''
    if not number_ok(chan_num, no_neg=True):
        return ''
    clist = nmp.CHANNEL_LIST
    if chan_num >= 0 and chan_num < len(clist):
        cc = clist[chan_num]
        return cc.upper()
    return ''


def chan_num(  # convert channel character to number
    chan_char  # channel character
):
    if not chan_char or not isinstance(chan_char, str) or len(chan_char) > 1:
        return -1
    for i, c in enumerate(nmp.CHANNEL_LIST):
        if chan_char.upper() == c.upper():
            return i
    return -1


def chan_char_check(  # check channel character is OK
    chan_char  # channel character
):
    if chan_num(chan_char) >= 0:
        return chan_char.upper()  # enforce upper case
    return ''


def chan_char_exists(  # search for channel character in text string
    text,  # text to search
    chan_char,  # channel character
):
    if not text or not isinstance(text, str):
        return False
    if not chan_char or not isinstance(chan_char, str) or len(chan_char) > 1:
        return False
    irange = reversed(range(len(text)))  # search backwards thru text
    for i in irange:
        if text[i].isdigit():
            continue
        if text[i].upper() == chan_char.upper():
            return True
        return False  # check only last character
    return False


def history_change(  # create history text for variables that have changed
    var_name,  # variable name
    from_value,  # changed from this value
    to_value  # changed to this value
):
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


def history(  # print message to history
    message,  # string message
    title='',  # message title (e.g. 'ALERT' or 'ERROR')
    tp='',  # treepath, pass 'none' for none
    frame=1,  # inspect frame # for creating treepath
    red=False,  # True: print red, False: print black
    quiet=False  # True: no print, False: print message
):
    red = bool_check(red, False)
    quiet = bool_check(quiet, False)
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


def get_treepath(  # create ancestry treepath
    stack,  # stack, e.g. inspect.stack()
    tp='',  # treepath
    frame=1  # inspect frame # for creating treepath
):
    if not stack:
        return ''
    method = get_method(stack, frame=frame)
    if not tp:
        tp = get_class(stack, frame=frame)
    path = ['nm']  # NeuroMatic
    if tp:
        path.append(tp)  # class
    if method:
        path.append(method)
    return '.'.join(path)


def get_class(
    stack,  # stack, e.g. inspect.stack()
    frame=1,  # inspect frame
    module=False
):
    module = bool_check(module, False)
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


def get_method(  # get method from inspect stack
    stack,  # stack, e.g. inspect.stack()
    frame=1  # inspect frame
):
    if len(stack) <= frame or len(stack[0]) == 0:
        return ''
    f = stack[frame][0]
    if not inspect.isframe(f):
        return ''
    return f.f_code.co_name


def input_yesno(  # get user yes/no/cancel input
    prompt,  # yes/no prompt message
    title='',  # prompt title
    tp='',  # treepath
    frame=1,  # inspect frame # for creating treepath
    cancel=False  # include cancel option
):
    cancel = bool_check(cancel, False)
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
