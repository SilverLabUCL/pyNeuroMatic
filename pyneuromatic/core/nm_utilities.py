# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
NM utility functions
Copyright 2019 Jason Rothman
"""
from __future__ import annotations
from colorama import Fore, Back, Style
import inspect
import math
from typing import Callable

CHANNEL_CHARS: tuple = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K",
                "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
                "W", "X", "Y", "Z")

CONFIRM_YNC: tuple = ("y", "yes", "n", "no", "c", "cancel")  # see input_yesno()

# for testing
BADTYPES: tuple = (None, 3, 3.14, True, [], (), {}, set(), "string", Fore)
BADNAMES: tuple = ("", "all", "default", "none", "select", "self", "nan",
            "inf", "-inf", "b&dn@me!")




def badtypes(
        ok: list = []  # list of items that are ok (not bad)
) -> list:
    badlist: list = [None, 3, 3.14, True, [], (), {}, set(), "string", Callable]
    for o in ok:
        badlist.remove(o)
    return badlist


def number_ok(
    number: object | list[object],
    must_be_integer: bool = False,
    inf_is_ok: bool = False,
    nan_is_ok: bool = False,
    zero_is_ok: bool = True,
    neg_is_ok: bool = True,
    pos_is_ok: bool = True,
    complex_is_ok: bool = False,
) -> bool:
    """
    Check if object(s) is a number.

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
    if len(number) == 0:  # no number
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


def name_ok(
    name: str,
    ok_names: str | list[str] = [],
    ok_strings: str | list[str] = ["_"],
) -> bool:
    """Check if name is alpha-numeric.

    :param name: name to check.
    :type name: str
    :param ok_names: list of names that are ok.
    :type ok_names: str or List[str]
    :param ok_strings: list of strings/symbols that are ok to include in name.
    :type ok_strings: str or List[str]
    :return: True if name ok, otherwise False.
    :rtype: bool
    """
    if not isinstance(name, str):
        return False
    if len(name) > 0 and not name[0].isalpha():  # first char must be alpha
        return False

    if isinstance(ok_names, str):
        ok_names = [ok_names]
    elif not isinstance(ok_names, list):
        e = typeerror(ok_names, "ok_names", "string or list")
        raise TypeError(e)
    for ok_name in ok_names:
        if not isinstance(ok_name, str):
            e = typeerror(ok_name, "ok_names: list item", "string")
            raise TypeError(e)
        if name.lower() == ok_name.lower():
            return True

    if isinstance(ok_strings, str):
        ok_strings = [ok_strings]
    elif not isinstance(ok_strings, list):
        e = typeerror(ok_strings, "ok_strings", "string or list")
        raise TypeError(e)
    for ok_str in ok_strings:
        if not isinstance(ok_str, str):
            e = typeerror(ok_strings, "ok_strings: list item", "string")
            raise TypeError(e)
        name = name.replace(ok_str, "")  # remove ok strings from name

    if not name.isalnum():
        return False

    return name.lower() not in BADNAMES  # compare lower-case name


def name_next_seq(
    self,
    names: list[str],  # existing names, e.g. ['RecordA0', 'RecordA1'...]
    prefix: str,  # prefix of names, e.g. 'A'
    first: int = 0,  # first number of sequence
) -> int:  # e.g. 3
    """Find next sequence number of a list of names.
    Names are case insensitive.

    :param names: list of names (keys) with format PREFIX + SEQ#.
    :type names: List[str]
    :param prefix: True for single quotes, False for double quotes.
    :type prefix: str
    :param first: first number of sequence.
    :type first: int
    :return: next unused sequence number.
    :rtype: int
    """
    if not isinstance(names, list):
        e = self._type_error("names", "List[string]")
        raise TypeError(e)
    if not isinstance(prefix, str):
        e = self._type_error("prefix", "string")
        raise TypeError(e)
    if not prefix or not name_ok(prefix):
        e = self._value_error("prefix")
        raise ValueError(e)
    if not isinstance(first, int):
        e = self._type_error("first", "integer")
        raise TypeError(e)
    if first < 0:
        e = self._value_error("first")
        raise ValueError(e)

    imax = -1
    for name in names:
        if not isinstance(name, str):
            e = self._type_error("name", "string")
            raise TypeError(e)
        name = name.lower()
        istr = name.replace(prefix.lower(), "")
        if str.isdigit(istr):
            imax = max(imax, int(istr))
    if imax >= first:
        return imax + 1
    return first


def keys_are_equal(
    keys1: str | list[str],
    keys2: str | list[str],
    case_sensitive: bool = False,
) -> bool:
    """Determine if two lists contain the same keys.
    Comparison can be either case sensitive or insensitive (default).
    Order does not matter.

    :param keys1: first key list.
    :type keys1: List[str]
    :param keys2: second key list.
    :type prefix: List[str]
    :param keys2: second key list.
    :type prefix: List[str]
    :return: True if key lists are the same, otherwise False.
    :rtype: bool
    """
    if not hasattr(keys1, "__iter__"):
        return False
    if not hasattr(keys2, "__iter__"):
        return False
    if not isinstance(keys1, list):
        keys1 = list(keys1)
    if not isinstance(keys2, list):
        keys2 = list(keys2)
    if len(keys1) != len(keys2):
        return False
    for k1 in keys1:
        if not isinstance(k1, str):
            return False
        found = False
        for k2 in keys2:
            if not isinstance(k2, str):
                return False
            if case_sensitive:
                if k1 == k2:
                    found = True
                    break
            else:
                if k1.lower() == k2.lower():
                    found = True
                    break
        if not found:
            return False
    return True


def remove_special_char(
    text: str | list[str], ok_char: list[str] = [], bad_char: list[str] = []
) -> str | list[str]:
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
        return ""
    if isinstance(text, str):
        ostr = ""
        for c in text:
            if c in bad_char:
                continue
            if c in ok_char or c.isalnum():
                ostr += c
        return ostr
    if isinstance(text, list):
        olist = []
        for t in text:
            ostr = ""
            if isinstance(t, str):
                for c in t:
                    if c in bad_char:
                        continue
                    if c in ok_char or c.isalnum():
                        ostr += c
            olist.append(ostr)
        return olist
    return ""


def int_list_to_seq_str(
    int_list: list[int], seperator: str = ", ", seperator_at_end: bool = False
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
        seperator = ", "
    if not isinstance(int_list, list):
        if isinstance(int_list, int):
            return str(int_list)
        else:
            return ""
    if len(int_list) == 1:
        i = int_list[0]
        if isinstance(i, int):
            return str(i)
        else:
            return ""
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
                        slist.append(str(sfirst) + "-" + str(i))
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
                slist.append(str(sfirst) + "-" + str(slast))
            seq_started = False
    seqstr = seperator.join(slist)
    if seperator_at_end:
        return seqstr + seperator
    return seqstr


def channel_char(
    chan_num: int | list[int], char_list: tuple[str] = CHANNEL_CHARS
) -> str | list[str]:
    """Convert channel number(s) to character.

    :param chan_num: channel number, e.g. 0.
    :type chan_num: int or list
    :param char_list: list of channel characters, e.g. ['A', 'B', 'C', 'D'].
    :type char_list: list, optional
    :return: channel character, e.g. 'A'. UNKNOWN: ''.
    :rtype: str or list
    """
    if not isinstance(char_list, list):
        char_list = CHANNEL_CHARS  # use default NM list
    if isinstance(chan_num, int):
        if chan_num >= 0 and chan_num < len(char_list):
            cc = char_list[chan_num]
            return cc.upper()
        else:
            return ""
    if isinstance(chan_num, list):
        clist = []
        for i in chan_num:
            if i >= 0 and i < len(char_list):
                cc = char_list[i]
                clist.append(cc.upper())
            else:
                clist.append("")
        return clist
    return ""


def channel_num(
    chan_char: str | list[str], char_list: tuple[str] = CHANNEL_CHARS
) -> int | list[int]:
    """Convert channel character(s) to number.

    :param chan_char: channel character, e.g. 'A', or list of characters.
    :type chan_char: str or list
    :param char_list: list of channel characters, e.g. ['A', 'B', 'C', 'D'].
    :type char_list: list, optional
    :return: channel number(s), e.g. 0. UNKNOWN: -1.
    :rtype: int or list
    """
    if not isinstance(char_list, list):
        char_list = CHANNEL_CHARS
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
    chan_char: str | list[str], char_list: tuple[str] = CHANNEL_CHARS
) -> str | list[str]:
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
                clist.append("")
        return clist
    return ""


def channel_char_search(text: str, chan_char: str) -> int:
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
        e = "bad chan_char: " + "channel character is not alphabetical"
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


def typeerror(obj: object, obj_name: str, type_str: str) -> str:
    return (
        obj_name
        + ": expected "
        + type_str
        + " but got %s %s" % (type(obj).__name__, obj)
    )


def history_change(param_name: str, old_value: object, new_value: object) -> str:
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
            old_value = "None"
        else:
            old_value = "'%s'" % old_value
    if not isinstance(new_value, str):
        if new_value is None:
            new_value = "None"
        else:
            new_value = "'%s'" % new_value
    h = "changed " + param_name + " from " + old_value + " to " + new_value
    return h


def history(
    message: str,
    title: str = "",
    tp: str = "",
    frame: int = 1,
    red: bool = False,
    quiet: bool = False,
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
        return ""
    if not isinstance(frame, int) or frame < 0:
        frame = 1
    if tp.upper() == "NONE":
        path = ""
    elif len(tp) == 0 or tp.upper() == "DEFAULT":
        path = get_treepath(inspect.stack(), frame=frame)
    else:
        path = tp
    if path:
        h = path + ": " + message
    else:
        h = message
    if isinstance(title, str) and len(title) > 0:
        h = title + ": " + h
    if not quiet:
        if red:
            print(Fore.RED + h + Fore.BLACK)
        else:
            print(h)
        # TODO: print to NM history
    return h


def get_treepath(
    stack: list, frame: int = 1, package: str = "nm"  # stack frame
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
        return ""
    if not isinstance(frame, int) or frame < 0:
        frame = 1
    if isinstance(package, str) and len(package) > 0:
        path = [package]
    else:
        path = []
    c = get_class_from_stack(stack, frame=frame)  # class ancestry
    m = get_method_from_stack(stack, frame=frame)
    if c:
        path.append(c)
    if m:
        path.append(m)
    return ".".join(path)  # e.g. 'nm.myparent.mychild.mymethod


def get_class_from_stack(stack: list, frame: int = 1, module: bool = False) -> str:
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
        return ""
    if not isinstance(frame, int) or frame < 0:
        frame = 1
    if len(stack) <= frame or len(stack[0]) == 0:
        return ""
    f = stack[frame][0]
    if not inspect.isframe(f):
        return ""
    if "self" not in f.f_locals:
        return ""
    class_tree = str(stack[frame][0].f_locals["self"].__class__)
    class_tree = class_tree.replace("<class ", "")
    class_tree = class_tree.replace("'", "")
    class_tree = class_tree.replace(">", "")
    class_tree = class_tree.split(".")
    m = class_tree[0]
    c = class_tree[1]
    if module:
        return m + "." + c
    return c


def get_method_from_stack(
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
        return ""
    if not isinstance(frame, int) or frame < 0:
        frame = 1
    if len(stack) <= frame or len(stack[0]) == 0:
        return ""
    f = stack[frame][0]
    if not inspect.isframe(f):
        return ""
    return f.f_code.co_name


def input_yesno(
    prompt: str,
    title: str = "",
    treepath: str = "default",
    frame: int = 1,
    cancel: bool = True,
    answer: str | None = None,  # for testing, bypasses input()
):
    """Get user yes/no/cancel input

    :param prompt: prompt message.
    :type prompt: str
    :param title: prompt title.
    :type title: str
    :param treepath: function treepath.
    :type treepath: str
    :param frame: inspect frame # for creating treepath.
    :type frame: int
    :param cancel: include 'cancel' option.
    :type cancel: bool
    :param answer: answer that will bypass input, for testing purposes.
    :type answer: str
    :return: user input, 'y', 'n', 'c' or 'error'
    :rtype: str
    """
    ok = list(CONFIRM_YNC)
    if not isinstance(prompt, str):
        prompt = ""
    if cancel:
        txt = prompt + "\n" + "(y)es (n)o (c)ancel: "
    else:
        txt = prompt + "\n" + "(y)es, (n)o: "
        ok.remove("c")
        ok.remove("cancel")
    if not isinstance(treepath, str):
        path = ""
    elif treepath.lower() == "default":
        path = get_treepath(inspect.stack(), frame=frame)
    else:
        path = treepath  # + '.userinput'
    if path:
        txt = path + ":\n" + txt
    if title:
        txt = title + ":\n" + txt
    if prompt.lower() == "testprompt":
        return txt
    if not isinstance(answer, str):
        answer = input(txt)
    a = answer.lower()
    if a in ok:
        return a[:1]  # 'y', 'n' or 'c'
    return "error"
