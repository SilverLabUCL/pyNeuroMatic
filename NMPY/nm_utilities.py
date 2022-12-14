# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
NM utility functions
Copyright 2019 Jason Rothman
"""
import math
import inspect
from colorama import Fore, Back, Style
from typing import Union, List, NewType

import nm_preferences as nmp

NMManagerType = NewType('NMManager', object)
NMObjectType = NewType('NMObject', object)
NMObjectContainerType = NewType('NMObjectContainer', NMObjectType)
NMProjectType = NewType('NMProject', NMObjectType)
NMFolderType = NewType('NMFolder', NMObjectType)
NMFolderContainerType = NewType('NMFolderContainer', NMObjectContainerType)
NMDataType = NewType('NMData', NMObjectType)
NMDataContainerType = NewType('NMDataContainer', NMObjectContainerType)
NMDataSeriesType = NewType('NMDataSeries', NMObjectType)
NMDataSeriesContainerType = NewType('NMDataSeriesContainer',
                                    NMObjectContainerType)
NMChannelType = NewType('NMChannel', NMObjectType)
NMChannelContainerType = NewType('NMChannelContainer', NMObjectContainerType)
NMScaleType = NewType('NMScale', NMObjectType)
NMScaleXType = NewType('NMScaleX', NMScaleType)
NMDataSeriesSetType = NewType('NMDataSeriesSet', NMObjectType)
NMDataSeriesSetContainerType = NewType('NMDataSeriesContainerSet',
                                       NMObjectContainerType)


def name_ok(
    name: Union[str, List[str]],
    ok_list: List[str] = nmp.NAME_SYMBOLS_OK
) -> bool:
    """Check if name(s) is alpha-numeric.

    :param name: name or list of names to check.
    :type name: str or list
    :param is_ok: list of symbols that are ok to include in names.
    :type is_ok: list, optional
    :return: True if name ok, otherwise False.
    :rtype: bool
    """
    if not isinstance(name, list):
        name = [name]  # convert to list of names
    if len(name) == 0:  # no name is not OK
        return False
    if not isinstance(ok_list, list):
        ok_list = [ok_list]
    for n in name:
        if not isinstance(n, str) or len(n) == 0:
            return False
        if not n[0].isalpha():
            return False
        for ok_str in ok_list:
            if isinstance(ok_str, str):
                n = n.replace(ok_str, '')  # remove ok strings
        if not n.isalnum():
            return False
    return True


def number_ok(
    number: Union[object, List[object]],
    must_be_integer: bool = False,
    inf_is_ok: bool = False,
    nan_is_ok: bool = False,
    zero_is_ok: bool = True,
    neg_is_ok: bool = True,
    pos_is_ok: bool = True,
    complex_is_ok: bool = False
) -> bool:
    """Check if object(s) is a number.

    :param number: object or list of objects to check.
    :type number: object or list
    :param must_be_integer: object must be an integer.
    :type must_be_integer: bool, optional
    :param inf_is_ok: infinity is ok.
    :type inf_is_ok: bool, optional
    :param nan_is_ok: NaN is ok.
    :type nan_is_ok: bool, optional
    :param zero_is_ok: 0 is ok.
    :type zero_is_ok: bool, optional
    :param neg_is_ok: negative number is ok.
    :type neg_is_ok: bool, optional
    :param pos_is_ok: positive number is ok.
    :type pos_is_ok: bool, optional
    :param complex_is_ok: complex number is ok.
    :type complex_is_ok: bool, optional
    :return: True if number is ok, otherwise False.
    :rtype: bool
    """
    if number is None:
        return False
    if not isinstance(number, list):
        number = [number]  # convert to list of numbers
    if len(number) == 0:  # no number is not OK
        return False
    for n in number:
        if must_be_integer and not isinstance(n, int):
            return False
        if isinstance(n, bool):
            return False
        if isinstance(n, complex):
            if complex_is_ok:
                continue
            else:
                return False
        if not isinstance(n, int) and not isinstance(n, float):
            return False
        if math.isinf(n):
            if inf_is_ok:
                continue
            else:
                return False
        if math.isnan(n):
            if nan_is_ok:
                continue
            else:
                return False
        if not zero_is_ok and n == 0:
            return False
        if not neg_is_ok and n < 0:
            return False
        if not pos_is_ok and n > 0:
            return False
    return True


def quotes(
    text: str,
    single: bool = True
) -> str:
    """Add quotes around text.

    :param text: text.
    :type text: str
    :param single: True for single quotes, False for double quotes.
    :type single: bool, optional
    :return: quoted text.
    :rtype: str
    """
    if not isinstance(text, str):
        text = str(text)
    if single:
        return "'" + str(text) + "'"
    return '"' + str(text) + '"'


def remove_special_char(
    text: Union[str, List[str]],
    ok_char: List[str] = [],
    bad_char: List[str] = []
) -> Union[str, List[str]]:
    """Remove non-alpha-numeric characters from text.

    :param text: text.
    :type text: str or list
    :param ok_char: list of ok characters, e.g. ['_', '.'].
    :type ok_char: list, optional
    :param bad_char: list of bad characters, e.g. ['3', 'n'].
    :type bad_char: list, optional
    :return: text without alpha-numeric characters.
    :rtype: str or list
    """
    if not text:
        return ''
    if isinstance(text, str):
        ostr = ''
        for c in text:
            if c in bad_char:
                continue
            if c in ok_char or c.isalnum():
                ostr += c
        return ostr
    if isinstance(text, list):
        olist = []
        for t in text:
            ostr = ''
            if isinstance(t, str):
                for c in t:
                    if c in bad_char:
                        continue
                    if c in ok_char or c.isalnum():
                        ostr += c
            olist.append(ostr)
        return olist
    return ''


def int_list_to_seq_str(
    int_list: List[int],
    seperator: str = ', ',
    seperator_at_end: bool = False
) -> str:
    """Convert list of integers to a sequence string.

    :param int_list: list of integers, e.g. [0,1,5,6,7,12,19,20,21,22].
    :type int_list: list
    :param seperator: sequence seperator string, e.g. ', ' or ';'.
    :type seperator: str, optional
    :param seperator_at_end: include seperator at end of sequence.
    :type seperator_at_end: bool, optional
    :return: sequence string, e.g. '0, 1, 5-7, 12, 19-22'.
    :rtype: str
    """
    if not isinstance(seperator, str):
        seperator = ', '
    if not isinstance(int_list, list):
        if isinstance(int_list, int):
            return str(int_list)
        else:
            return ''
    if len(int_list) == 1:
        i = int_list[0]
        if isinstance(i, int):
            return str(i)
        else:
            return ''
    ilist = []
    for i in int_list:
        if isinstance(i, int):
            ilist.append(i)
    imin = min(ilist)
    imax = max(ilist)
    seq_started = False
    sfirst = ilist[0]
    slist = []
    for i in range(imin, imax + 1):  # [0,1,2,3,4,5,6,7,8]
        if i in ilist:
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
    seqstr = seperator.join(slist)
    if seperator_at_end:
        return seqstr + seperator
    return seqstr


def channel_char(
    chan_num: Union[int, List[int]],
    char_list: List[str] = nmp.CHANNEL_LIST
) -> Union[str, List[str]]:
    """Convert channel number(s) to character.

    :param chan_num: channel number, e.g. 0.
    :type chan_num: int or list
    :param char_list: list of channel characters, e.g. ['A', 'B', 'C', 'D'].
    :type char_list: list, optional
    :return: channel character, e.g. 'A'. UNKNOWN: ''.
    :rtype: str or list
    """
    if not isinstance(char_list, list):
        char_list = nmp.CHANNEL_LIST  # use default NM list
    if isinstance(chan_num, int):
        if chan_num >= 0 and chan_num < len(char_list):
            cc = char_list[chan_num]
            return cc.upper()
        else:
            return ''
    if isinstance(chan_num, list):
        clist = []
        for i in chan_num:
            if i >= 0 and i < len(char_list):
                cc = char_list[i]
                clist.append(cc.upper())
            else:
                clist.append('')
        return clist
    return ''


def channel_num(
    chan_char: Union[str, List[str]],
    char_list: List[str] = nmp.CHANNEL_LIST
) -> Union[int, List[int]]:
    """Convert channel character(s) to number.

    :param chan_char: channel character, e.g. 'A', or list of characters.
    :type chan_char: str or list
    :param char_list: list of channel characters, e.g. ['A', 'B', 'C', 'D'].
    :type char_list: list, optional
    :return: channel number(s), e.g. 0. UNKNOWN: -1.
    :rtype: int or list
    """
    if not isinstance(char_list, list):
        char_list = nmp.CHANNEL_LIST
    if isinstance(chan_char, str):
        for i, c in enumerate(char_list):
            if chan_char.upper() == c.upper():
                return i
        return -1
    if isinstance(chan_char, list):
        ilist = []
        for c1 in chan_char:
            if isinstance(c1, str) and len(c1) == 1:
                i1 = -1
                for i2, c2 in enumerate(char_list):
                    if c1.upper() == c2.upper():
                        i1 = i2
                        break
                ilist.append(i1)
            else:
                ilist.append(-1)
        return ilist
    return -1


def channel_char_check(
    chan_char: Union[str, List[str]],
    char_list: List[str] = nmp.CHANNEL_LIST
) -> Union[str, List[str]]:
    """Check channel character

    :param chan_char: channel character, e.g. 'A', or list of characters.
    :type chan_char: str or list
    :param char_list: list of channel characters, e.g. ['A', 'B', 'C', 'D'].
    :type char_list: list, optional
    :return: the channel character if ok. UNKNOWN: ''.
    :rtype: str or list
    """
    if isinstance(chan_char, str):
        if channel_num(chan_char, char_list=char_list) >= 0:
            return chan_char.upper()  # enforce upper case
    if isinstance(chan_char, list):
        clist = []
        for c in chan_char:
            if channel_num(c, char_list=char_list) >= 0:
                clist.append(c)
            else:
                clist.append('')
        return clist
    return ''


def channel_char_search(
    text: str,
    chan_char: str
) -> int:
    """Search for channel character in text (backwards search).

    :param text: text to search, e.g. 'RecordA127'.
    :type text: str
    :param chan_char: channel character to find, e.g. 'A'.
    :type chan_char: str
    :return: string location of channel character. NOT FOUND: -1.
    :rtype: int
    :raises TypeError: if chan_char is not alphabetical.
    """
    if not text or not isinstance(text, str):
        return -1
    if not chan_char or not isinstance(chan_char, str):
        return -1
    if not chan_char.isalpha():
        e = ("bad chan_char: " + "channel character is not alphabetical")
        raise TypeError(e)
    irange = reversed(range(len(text)))  # search backwards for first alpha
    ifound = -1
    for i in irange:
        if text[i].isalpha():
            ifound = i
            break
    i1 = ifound - len(chan_char) + 1
    i2 = i1 + len(chan_char)
    if text[i1:i2].upper() == chan_char.upper():
        return i1
    return -1


def history_change(
    param_name: str,
    old_value: object,
    new_value: object
) -> str:
    """Create history text for variables that have changed

    :param param_name: name of parameter that has changed.
    :type param_name: str
    :param old_value: old value of parameter.
    :type old_value: object
    :param new_value: new value of parameter.
    :type new_value: object
    :return: history string containing the change.
    :rtype: str
    """
    if not isinstance(param_name, str):
        param_name = str(param_name)
    if not isinstance(old_value, str):
        if old_value is None:
            old_value = str(old_value)
        else:
            old_value = quotes(str(old_value))
    if not isinstance(new_value, str):
        if new_value is None:
            new_value = str(new_value)
        else:
            new_value = quotes(str(new_value))
    h = 'changed ' + param_name + ' from ' + old_value + ' to ' + new_value
    return h


def history(
    message: str,
    title: str = '',
    tp: str = '',
    frame: int = 1,
    red: bool = False,
    quiet: bool = False
) -> str:
    """Print message to NM history.

    :param message: message to print.
    :type message: str
    :param title: message title (e.g. 'ALERT' or 'ERROR').
    :type title: str
    :param tp: function treepath, pass '' for default or 'NONE' for none.
    :type tp: str
    :param frame: inspect frame # for creating treepath.
    :type frame: int
    :param red: True to print red, False to print black.
    :type red: bool
    :param quiet: True to not print message, False to print.
    :type quiet: bool
    :return: history string.
    :rtype: str
    """
    if not isinstance(message, str):
        return ''
    if not isinstance(frame, int) or frame < 0:
        frame = 1
    if tp.upper() == 'NONE':
        path = ''
    elif len(tp) == 0 or tp.upper() == 'DEFAULT':
        path = get_treepath(inspect.stack(), frame=frame)
    else:
        path = tp
    if path:
        h = path + ': ' + message
    else:
        h = message
    if isinstance(title, str) and len(title) > 0:
        h = title + ': ' + h
    if not quiet:
        if red:
            print(Fore.RED + h + Fore.BLACK)
        else:
            print(h)
        # TODO: print to NM history
    return h


def get_treepath(
    stack: list,
    frame: int = 1,
    package: str = 'nm'
) -> str:
    """Create function ancestry treepath.

    :param stack: stack list.
    :type stack: list
    :param frame: inspect frame # for creating treepath.
    :type frame: int
    :param package: package, e.g. 'nm'
    :type package: str
    :return: treepath string.
    :rtype: str
    """
    if not stack:
        return ''
    if not isinstance(frame, int) or frame < 0:
        frame = 1
    if isinstance(package, str) and len(package) > 0:
        path = [package]
    else:
        path = []
    c = get_class(stack, frame=frame)  # class ancestry
    m = get_method(stack, frame=frame)
    if c:
        path.append(c)
    if m:
        path.append(m)
    return '.'.join(path)  # e.g. 'nm.myparent.mychild.mymethod


def get_class(
    stack: list,
    frame: int = 1,
    module: bool = False
) -> str:
    """Extract class from stack

    :param stack: stack list.
    :type stack: list
    :param frame: inspect frame # for creating treepath.
    :type frame: int
    :param module: include module name.
    :type module: bool
    :return: class name.
    :rtype: str
    """
    if not stack:
        return ''
    if not isinstance(frame, int) or frame < 0:
        frame = 1
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


def get_method(
    stack: list,
    frame: int = 1,
) -> str:
    """Extract method from stack

    :param stack: stack list.
    :type stack: list
    :param frame: inspect frame # for creating treepath.
    :type frame: int
    :return: method name.
    :rtype: str
    """
    if not stack:
        return ''
    if not isinstance(frame, int) or frame < 0:
        frame = 1
    if len(stack) <= frame or len(stack[0]) == 0:
        return ''
    f = stack[frame][0]
    if not inspect.isframe(f):
        return ''
    return f.f_code.co_name


def input_yesno(
    prompt: str,
    title: str = '',
    tp: str = '',
    frame: int = 1,
    cancel: bool = False  # include cancel option
):
    """Get user yes/no/cancel input

    :param prompt: prompt message.
    :type prompt: str
    :param title: prompt title.
    :type title: str
    :param tp: function treepath.
    :type tp: str
    :param frame: inspect frame # for creating treepath.
    :type frame: int
    :param cancel: include 'cancel'.
    :type cancel: bool
    :return: user input, 'yes', 'no' or 'cancel'.
    :rtype: str
    """
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
        path = get_treepath(inspect.stack(), frame=frame)
    if path:
        txt = path + ': ' + txt
    if title:
        txt = title + ': ' + txt
    answer = input(txt)
    a = answer.lower()
    if a in ok:
        return a[:1]
    return ''
