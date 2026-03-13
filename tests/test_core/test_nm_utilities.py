#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 17:23:44 2023

@author: jason
"""
import unittest

from pyneuromatic.core.nm_manager import NMManager
import pyneuromatic.core.nm_history as nmh
import pyneuromatic.core.nm_utilities as nmu

QUIET = True
NM = NMManager(quiet=QUIET)


class NMUtilitiesTest(unittest.TestCase):

    def setUp(self):  # executed before each test
        pass

    def test00_name_ok(self):
        bad = list(nmu.BADTYPES)
        bad.remove("string")
        for b in bad:
            self.assertFalse(nmu.name_ok(b))

        bad = list(nmu.BADTYPES)
        bad.remove("string")
        bad.remove([])
        for b in bad:
            with self.assertRaises(TypeError):
                nmu.name_ok("test", ok_names=b)
            with self.assertRaises(TypeError):
                nmu.name_ok("test", ok_strings=b)

        bad = list(nmu.BADNAMES)
        for b in bad:
            self.assertFalse(nmu.name_ok(b))

        self.assertFalse(nmu.name_ok("_"))
        self.assertFalse(nmu.name_ok("*"))
        self.assertFalse(nmu.name_ok("0"))
        self.assertTrue(nmu.name_ok("test"))
        self.assertTrue(nmu.name_ok("test1234567890"))

        # NM preferences: NAME_SYMBOLS_OK = ['_']
        self.assertTrue(nmu.name_ok("test_"))
        self.assertTrue(nmu.name_ok("test_OK"))
        self.assertTrue(nmu.name_ok("test_OK_OK"))

        self.assertFalse(nmu.name_ok("test*"))
        self.assertFalse(nmu.name_ok("@test"))
        self.assertFalse(nmu.name_ok("te.st"))
        self.assertTrue(nmu.name_ok("te.st", ok_strings=["."]))

    def test01_number_ok(self):
        bad = list(nmu.BADTYPES)
        bad.remove(3)
        bad.remove(3.14)
        for b in bad:
            self.assertFalse(nmu.number_ok(b))
            self.assertFalse(nmu.number_ok([1, 3.14, b]))

        self.assertTrue(nmu.number_ok(1))
        self.assertTrue(nmu.number_ok(3.14))
        self.assertTrue(nmu.number_ok(-3.14))
        self.assertTrue(nmu.number_ok(0))
        self.assertFalse(nmu.number_ok(1.34, must_be_integer=True))
        self.assertTrue(nmu.number_ok(1, must_be_integer=True))
        self.assertFalse(nmu.number_ok(False, must_be_integer=True))
        self.assertFalse(nmu.number_ok(float("inf")))
        self.assertTrue(nmu.number_ok(float("inf"), inf_is_ok=True))
        self.assertFalse(nmu.number_ok(float("-inf")))
        self.assertTrue(nmu.number_ok(float("-inf"), inf_is_ok=True))
        self.assertFalse(nmu.number_ok(float("nan")))
        self.assertTrue(nmu.number_ok(float("nan"), nan_is_ok=True))
        self.assertTrue(nmu.number_ok(0, neg_is_ok=False))
        self.assertTrue(nmu.number_ok(1.34, neg_is_ok=False))
        self.assertFalse(nmu.number_ok(-1.34, neg_is_ok=False))
        self.assertTrue(nmu.number_ok(0, pos_is_ok=False))
        self.assertFalse(nmu.number_ok(1.34, pos_is_ok=False))
        self.assertTrue(nmu.number_ok(-1.34, pos_is_ok=False))
        self.assertTrue(nmu.number_ok(0))
        self.assertFalse(nmu.number_ok(0, zero_is_ok=False))
        self.assertTrue(nmu.number_ok(1.34, zero_is_ok=False))
        self.assertTrue(nmu.number_ok(-1.34, zero_is_ok=False))
        self.assertFalse(nmu.number_ok(complex(1, -1)))
        self.assertTrue(nmu.number_ok(complex(1, -1), complex_is_ok=True))
        self.assertFalse(nmu.number_ok([0, -5, 1.34, "one"]))
        self.assertFalse(nmu.number_ok([0, -5, 1.34, None]))
        self.assertFalse(nmu.number_ok([0, -5, 1.34, False]))
        self.assertTrue(nmu.number_ok([0, -5, 1.34]))
        self.assertFalse(nmu.number_ok([0, -5, 1.34], must_be_integer=True))
        self.assertFalse(nmu.number_ok([0, -5, 1.34, float("inf")]))
        self.assertTrue(nmu.number_ok([0, -5, float("inf")], inf_is_ok=True))
        self.assertFalse(nmu.number_ok([0, -5, 1.34, float("-inf")]))
        self.assertTrue(nmu.number_ok([0, -5, float("-inf")], inf_is_ok=True))
        self.assertFalse(nmu.number_ok([0, -5, 1.34, float("nan")]))
        self.assertTrue(nmu.number_ok([0, -5, float("nan")], nan_is_ok=True))
        self.assertTrue(nmu.number_ok([0, 3, 4], neg_is_ok=False))
        self.assertFalse(nmu.number_ok([-1, 3, 4], neg_is_ok=False))
        self.assertFalse(nmu.number_ok([0, 3, 4], pos_is_ok=False))
        self.assertTrue(nmu.number_ok([0, -3, -4], pos_is_ok=False))
        self.assertFalse(nmu.number_ok([0, 3, 4], zero_is_ok=False))
        self.assertTrue(nmu.number_ok([-4, 4], zero_is_ok=False))
        self.assertFalse(nmu.number_ok([0, -5, 1.34, complex(1, -1)]))
        self.assertTrue(
            nmu.number_ok([0, -5, 1.34, complex(1, -1)], complex_is_ok=True)
        )

    def test02_keys_equal(self):
        klist1 = ["one", "two", "three"]
        klist2 = ["ONE", "TWO", "THREE"]

        bad = list(nmu.BADTYPES)
        for b in bad:
            self.assertFalse(nmu.keys_are_equal(b, klist2))
            self.assertFalse(nmu.keys_are_equal(klist1, b))

        self.assertTrue(nmu.keys_are_equal(klist1, klist2))
        self.assertFalse(nmu.keys_are_equal(klist1, klist2, case_sensitive=True))
        klist2.reverse()
        self.assertTrue(nmu.keys_are_equal(klist1, klist2))

        dlist1 = {}
        dlist1.update({"one": 1})
        dlist1.update({"two": 2})
        dlist1.update({"three": 3})
        klist1 = dlist1.keys()

        dlist2 = {}
        dlist2.update({"ONE": 1})
        dlist2.update({"TWO": 2})
        dlist2.update({"THREE": 3})
        klist2 = dlist2.keys()

        self.assertTrue(nmu.keys_are_equal(klist1, klist2))
        self.assertFalse(nmu.keys_are_equal(klist1, klist2, case_sensitive=True))

        klist2 = list(klist2)
        klist2.reverse()
        self.assertTrue(nmu.keys_are_equal(klist1, klist2))

        klist1 = ["one", "two", 3]
        klist2 = ["ONE", "TWO", "THREE"]
        self.assertFalse(nmu.keys_are_equal(klist1, klist2))

    def test03_input_yesno(self):
        self.assertEqual(nmu.prompt_yes_no("test", answer=""), "error")
        self.assertEqual(nmu.prompt_yes_no("test", answer="YES"), "y")
        self.assertEqual(nmu.prompt_yes_no("test", answer="Y"), "y")
        self.assertEqual(nmu.prompt_yes_no("test", answer="NO"), "n")
        self.assertEqual(nmu.prompt_yes_no("test", answer="N"), "n")
        self.assertEqual(nmu.prompt_yes_no("test", answer="CANCEL"), "c")
        self.assertEqual(nmu.prompt_yes_no("test", answer="C"), "c")
        self.assertEqual(nmu.prompt_yes_no("test", cancel=False, answer="C"), "error")
        p1 = nmu.prompt_yes_no("testprompt", title='MyTitle', path='my.path')
        p2 = "MyTitle:" + "\n" + "my.path:" + "\n" + "testprompt" + "\n" + "(y)es (n)o (c)ancel: "
        self.assertEqual(p1, p2)
        # print(p1)

    """
        # remove_special_char
        self.assertEqual(nmu.remove_special_char('test*'), 'test')
        self.assertEqual(nmu.remove_special_char('test'), 'test')
        self.assertEqual(nmu.remove_special_char('t@st*'), 'tst')
        self.assertEqual(nmu.remove_special_char(''), '')
        self.assertEqual(
            nmu.remove_special_char(['test*', 't@st*']), ['test', 'tst'])
        self.assertEqual(
            nmu.remove_special_char(['test', None, False]), ['test', '', ''])
        self.assertEqual(nmu.remove_special_char(None), '')
        self.assertEqual(nmu.remove_special_char(False), '')
        self.assertEqual(nmu.remove_special_char('test_*'), 'test')
        self.assertEqual(
            nmu.remove_special_char('test_*', ok_char=['_']), 'test_')
        self.assertEqual(
            nmu.remove_special_char('test_*', ok_char=['_', '*']), 'test_*')
        self.assertEqual(
            nmu.remove_special_char(
                'test_*',
                ok_char=['_', '*'],
                bad_char=['_', '*']),
            'test')
        self.assertEqual(
            nmu.remove_special_char('test0_*', bad_char=['t', '0']), 'es')
        self.assertEqual(
            nmu.remove_special_char('test0_*', ok_char=['_', '*'],
                                    bad_char=['t', '0']), 'es_*')
        # int_list_to_seq_str
        i = [1, 2, 3, 4, 6]
        s = '1-4, 6'
        self.assertEqual(nmu.int_list_to_seq_str(i), s)
        i = [0, 1, 2, 16, 145]
        s = '0-2, 16, 145'
        self.assertEqual(nmu.int_list_to_seq_str(i), s)
        i = [0, 2, 4, 6]
        s = '0, 2, 4, 6'
        self.assertEqual(nmu.int_list_to_seq_str(i), s)
        i = [0, 1, 5, 6, 7, 12, 19, 20, 21, 22, 124]
        s = '0, 1, 5-7, 12, 19-22, 124'
        self.assertEqual(nmu.int_list_to_seq_str(i), s)
        i = [0, 4, 5.0, 6]
        s = '0, 4, 6'
        self.assertEqual(nmu.int_list_to_seq_str(i), s)
        i = [0, None, 4, 5, 6]
        s = '0, 4-6'
        self.assertEqual(nmu.int_list_to_seq_str(i), s)
        i = [0, float('nan'), 4, 5, 6]
        s = '0, 4-6'
        self.assertEqual(nmu.int_list_to_seq_str(i), s)
        i = [0, 2, 4, 5, 6]
        s = '0,2,4-6'
        self.assertEqual(nmu.int_list_to_seq_str(i, seperator=','), s)
        i = [0, 2, 3, 4, 6]
        s = '0$2-4$6'
        self.assertEqual(nmu.int_list_to_seq_str(i, seperator='$'), s)
        i = [0, 2, 3, 4, 6]
        s = '0, 2-4, 6, '
        self.assertEqual(nmu.int_list_to_seq_str(i, seperator_at_end=True), s)
        # channel_char
        self.assertEqual(nmu.channel_char(0), 'A')
        self.assertEqual(nmu.channel_char(1), 'B')
        self.assertEqual(nmu.channel_char(2), 'C')
        self.assertEqual(nmu.channel_char(10), 'K')
        self.assertEqual(nmu.channel_char(26), '')
        self.assertEqual(nmu.channel_char(-2), '')
        self.assertEqual(nmu.channel_char(float('inf')), '')
        self.assertEqual(nmu.channel_char(float('nan')), '')
        self.assertEqual(nmu.channel_char(None), '')
        self.assertEqual(nmu.channel_char([]), [])
        self.assertEqual(nmu.channel_char([0, 1, 3]), ['A', 'B', 'D'])
        self.assertEqual(nmu.channel_char([0, 1, 26]), ['A', 'B', ''])
        clist = ['w', 'x', 'y', 'z']
        self.assertEqual(nmu.channel_char(0, char_list=clist), 'W')
        self.assertEqual(nmu.channel_char(3, char_list=clist), 'Z')
        self.assertEqual(nmu.channel_char(4, char_list=clist), '')
        self.assertEqual(nmu.channel_char([0, 1, 3], char_list=clist),
                         ['W', 'X', 'Z'])
        clist = ['AA', 'BB', 'CC', 'DD']
        self.assertEqual(nmu.channel_char(3, char_list=clist), 'DD')
        # channel_num
        self.assertEqual(nmu.channel_num(None), -1)
        self.assertEqual(nmu.channel_num('A'), 0)
        self.assertEqual(nmu.channel_num('a'), 0)
        self.assertEqual(nmu.channel_num('b'), 1)
        self.assertEqual(nmu.channel_num('K'), 10)
        self.assertEqual(nmu.channel_num(''), -1)
        self.assertEqual(nmu.channel_num('AA'), -1)
        clist = ['AA', 'BB', 'CC', 'DD']
        self.assertEqual(nmu.channel_num('AA', char_list=clist), 0)
        clist = ['w', 'x', 'y', 'z']
        self.assertEqual(nmu.channel_num('A', char_list=clist), -1)
        self.assertEqual(nmu.channel_num('Y', char_list=clist), 2)
        self.assertEqual(nmu.channel_num([]), [])
        self.assertEqual(nmu.channel_num(['A', 'B', 'D']), [0, 1, 3])
        self.assertEqual(nmu.channel_num(['A', 'B', '@']), [0, 1, -1])
        self.assertEqual(nmu.channel_num(['c', 'a', 'f']), [2, 0, 5])
        self.assertEqual(nmu.channel_num(['A', 'B', 'C'], char_list=clist),
                         [-1, -1, -1])
        self.assertEqual(nmu.channel_num(['w', 'z', 'x'], char_list=clist),
                         [0, 3, 1])
        # channel_char_check
        self.assertEqual(nmu.channel_char_check(None), '')
        self.assertEqual(nmu.channel_char_check('A'), 'A')
        self.assertEqual(nmu.channel_char_check('Z'), 'Z')
        clist = ['w', 'x', 'y', 'z']
        self.assertEqual(nmu.channel_char_check('A', char_list=clist), '')
        self.assertEqual(nmu.channel_char_check('Z', char_list=clist), 'Z')
        self.assertEqual(nmu.channel_char_check([]), [])
        self.assertEqual(nmu.channel_char_check(['A', 'B', 'D']),
                         ['A', 'B', 'D'])
        self.assertEqual(nmu.channel_char_check(['A', 'B', 'DD']),
                         ['A', 'B', ''])
        self.assertEqual(nmu.channel_char_check(['A', 'B', 'D'],
                         char_list=clist), ['', '', ''])
        # channel_char_search
        self.assertEqual(nmu.channel_char_search(None, 'A'), -1)
        self.assertEqual(nmu.channel_char_search('testA1', None), -1)
        with self.assertRaises(TypeError):
            self.assertEqual(nmu.channel_char_search('testA1', 'A1'), -1)
        with self.assertRaises(TypeError):
            self.assertEqual(nmu.channel_char_search('testA1', 'A$'), -1)
        self.assertEqual(nmu.channel_char_search('testA1', 'a'), 4)
        self.assertEqual(nmu.channel_char_search('testa111', 'A'), 4)
        self.assertEqual(nmu.channel_char_search('testA', 'A'), 4)
        self.assertEqual(nmu.channel_char_search('A', 'A'), 0)
        self.assertEqual(nmu.channel_char_search('testA111', 'B'), -1)
        self.assertEqual(nmu.channel_char_search('A', 'B'), -1)
        self.assertEqual(nmu.channel_char_search('taste', 'A'), -1)
        self.assertEqual(nmu.channel_char_search('testAA12', 'AA'), 4)
        self.assertEqual(nmu.channel_char_search('testAAA1267', 'AAA'), 4)
        self.assertEqual(nmu.channel_char_search('testAAA@1267', 'AAA'), 4)
        self.assertEqual(nmu.channel_char_search('testA@1267', 'A'), 4)
        self.assertEqual(nmu.channel_char_search('A@1267', 'A'), 0)
        # history_change
        # history
        quiet = False
        fxn = '_test_utilities'
        c = 'Test'  # this class
        h = 'history message'
        r = 'nm.' + c + '.' + fxn + ': ' + h
        self.assertEqual(nmh.history(h, quiet=quiet), r)
        tp = 'nm.one.two.three'
        r = tp + ': ' + h
        self.assertEqual(nmh.history(h, tp=tp, quiet=quiet), r)

        # get_path
        stack = inspect.stack()
        fxn = 'test_all'  # calling fxn
        r = 'nm.' + c + '.' + fxn
        self.assertEqual(nmh.get_path(stack), r)
        tp = 'one.two.three'
        r = 'nm.one.two.three.' + fxn
        self.assertEqual(nmh.get_path(stack, path=tp), r)
        # get_class
        stack = inspect.stack()
        self.assertEqual(nmh.get_class_from_stack(stack), c)
        self.assertEqual(nmh.get_class_from_stack(stack, module=True), '__main__.' + c)
        # get_method
        stack = inspect.stack()
        self.assertEqual(nmh.get_method_from_stack(stack), fxn)
    """


class TestParseDataName(unittest.TestCase):
    """Tests for nmu.parse_data_name() utility function."""

    def test_standard_name(self):
        result = nmu.parse_data_name("RecordA0")
        self.assertEqual(result, ("Record", "A", 0))

    def test_multi_digit_epoch(self):
        result = nmu.parse_data_name("RecordB123")
        self.assertEqual(result, ("Record", "B", 123))

    def test_lowercase_prefix(self):
        result = nmu.parse_data_name("avgC5")
        self.assertEqual(result, ("avg", "C", 5))

    def test_long_prefix(self):
        result = nmu.parse_data_name("MyLongPrefixD99")
        self.assertEqual(result, ("MyLongPrefix", "D", 99))

    def test_no_trailing_digits(self):
        result = nmu.parse_data_name("RecordA")
        self.assertIsNone(result)

    def test_no_channel_char(self):
        # "test_123" has underscore before digits, not a channel letter
        result = nmu.parse_data_name("test_123")
        self.assertIsNone(result)

    def test_record123_parses_d_as_channel(self):
        # "Record123" actually has "D" as channel (Recor-D-123)
        result = nmu.parse_data_name("Record123")
        self.assertEqual(result, ("Recor", "D", 123))

    def test_empty_prefix(self):
        result = nmu.parse_data_name("A0")
        self.assertIsNone(result)

    def test_empty_string(self):
        result = nmu.parse_data_name("")
        self.assertIsNone(result)

    def test_none_input(self):
        result = nmu.parse_data_name(None)
        self.assertIsNone(result)

    def test_only_digits(self):
        result = nmu.parse_data_name("12345")
        self.assertIsNone(result)

    def test_channel_z(self):
        result = nmu.parse_data_name("WaveZ42")
        self.assertEqual(result, ("Wave", "Z", 42))

    def test_lowercase_channel_normalized(self):
        # Channel should be uppercase in result
        result = nmu.parse_data_name("Recorda0")
        self.assertEqual(result, ("Record", "A", 0))


# ===========================================================================
# TestFormatEpochString
# ===========================================================================

class TestFormatEpochString(unittest.TestCase):
    """Tests for nmu.format_epoch_string()."""

    # --- consecutive epochs → range notation ---

    def test_consecutive_returns_range(self):
        self.assertEqual(nmu.format_epoch_string(["RecordA0", "RecordA1", "RecordA2"]), "0-2")

    def test_two_consecutive(self):
        self.assertEqual(nmu.format_epoch_string(["RecordA4", "RecordA5"]), "4-5")

    def test_large_consecutive_range(self):
        names = ["RecordA%d" % i for i in range(10)]
        self.assertEqual(nmu.format_epoch_string(names), "0-9")

    # --- non-consecutive epochs → bracket notation ---

    def test_non_consecutive_returns_brackets(self):
        self.assertEqual(nmu.format_epoch_string(["RecordA0", "RecordA2", "RecordA5"]), "[0,2,5]")

    def test_single_gap(self):
        self.assertEqual(nmu.format_epoch_string(["RecordA0", "RecordA2"]), "[0,2]")

    # --- single array ---

    def test_single_array_returns_single_number(self):
        self.assertEqual(nmu.format_epoch_string(["RecordA3"]), "[3]")

    # --- unparseable names fall back to the raw name ---

    def test_unparseable_name_uses_raw(self):
        result = nmu.format_epoch_string(["nopattern"])
        self.assertIn("nopattern", result)

    def test_mixed_parseable_and_not(self):
        # One parseable, one not — should not raise
        result = nmu.format_epoch_string(["RecordA0", "nopattern"])
        self.assertIsInstance(result, str)

    # --- empty list ---

    def test_empty_list_returns_string(self):
        # min/max on empty list raises ValueError, caught internally
        result = nmu.format_epoch_string([])
        self.assertIsInstance(result, str)

    # --- channel/prefix independence ---

    def test_different_prefix_same_epochs(self):
        self.assertEqual(nmu.format_epoch_string(["StimulusA0", "StimulusA1"]), "0-1")

    def test_multi_digit_epoch(self):
        self.assertEqual(nmu.format_epoch_string(["RecordA10", "RecordA11", "RecordA12"]), "10-12")

    # --- arithmetic series (constant step > 1) ---

    def test_arithmetic_series_step3(self):
        names = ["RecordA%d" % i for i in [1, 4, 7, 10, 13]]
        self.assertEqual(nmu.format_epoch_string(names), "1-13:3")

    def test_arithmetic_series_step2(self):
        names = ["RecordA%d" % i for i in [0, 2, 4, 6]]
        self.assertEqual(nmu.format_epoch_string(names), "0-6:2")

    def test_arithmetic_series_step5(self):
        names = ["RecordA%d" % i for i in [5, 10, 15]]
        self.assertEqual(nmu.format_epoch_string(names), "5-15:5")

    def test_arithmetic_series_requires_n3(self):
        # n=2 with step > 1 should NOT use arithmetic notation
        self.assertEqual(nmu.format_epoch_string(["RecordA1", "RecordA4"]), "[1,4]")

    def test_non_constant_step_not_arithmetic(self):
        # steps 2, 3 — not constant, falls through to grouped format
        names = ["RecordA%d" % i for i in [0, 2, 5]]
        self.assertEqual(nmu.format_epoch_string(names), "[0,2,5]")

    # --- grouped consecutive runs ---

    def test_two_groups(self):
        names = ["RecordA%d" % i for i in [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]]
        self.assertEqual(nmu.format_epoch_string(names), "[0-4,10-14]")

    def test_two_groups_short(self):
        # 2-element runs listed individually, not as range
        names = ["RecordA%d" % i for i in [0, 1, 5, 6, 7]]
        self.assertEqual(nmu.format_epoch_string(names), "[0,1,5-7]")

    def test_three_groups(self):
        names = ["RecordA%d" % i for i in [0, 1, 2, 5, 6, 10]]
        self.assertEqual(nmu.format_epoch_string(names), "[0-2,5,6,10]")

    def test_groups_with_singletons(self):
        # Mix of ranges and single points
        names = ["RecordA%d" % i for i in [0, 1, 2, 5, 8, 11]]
        self.assertEqual(nmu.format_epoch_string(names), "[0-2,5,8,11]")

    # --- sorting ---

    def test_unsorted_input_sorted_in_output(self):
        # Names given out of order — result should be sorted
        names = ["RecordA2", "RecordA0", "RecordA1"]
        self.assertEqual(nmu.format_epoch_string(names), "0-2")


if __name__ == "__main__":
    unittest.main(verbosity=2)
