# -*- coding: utf-8 -*-
import copy
import unittest

from pyneuromatic.core.nm_notes import NMNotes


class TestNMNotes(unittest.TestCase):
    """Tests for NMNotes class."""

    def setUp(self):
        self.notes = NMNotes()

    def test_initially_empty(self):
        self.assertIsInstance(self.notes, NMNotes)
        self.assertEqual(len(self.notes), 0)

    def test_note_getter_empty(self):
        self.assertEqual(self.notes.note, "")

    def test_add(self):
        self.notes.add("first note")
        self.notes.add("second note")
        self.assertEqual(len(self.notes), 2)
        self.assertEqual(self.notes[0].get("note"), "first note")
        self.assertEqual(self.notes[1].get("note"), "second note")

    def test_note_getter_returns_last(self):
        self.notes.add("first")
        self.notes.add("last")
        self.assertEqual(self.notes.note, "last")

    def test_note_setter(self):
        self.notes.note = "added TTX"
        self.assertEqual(len(self.notes), 1)
        self.assertEqual(self.notes[0].get("note"), "added TTX")

    def test_timestamps(self):
        self.notes.add("test note")
        self.assertIn("date", self.notes[0])
        self.assertIsInstance(self.notes[0]["date"], str)

    def test_clear(self):
        self.notes.add("note1")
        self.notes.add("note2")
        self.notes.clear()
        self.assertEqual(len(self.notes), 0)
        self.assertEqual(self.notes.note, "")

    def test_add_none_ignored(self):
        self.notes.add(None)
        self.assertEqual(len(self.notes), 0)

    def test_add_non_string_converted(self):
        self.notes.add(42)
        self.assertEqual(len(self.notes), 1)
        self.assertEqual(self.notes[0].get("note"), "42")

    def test_len(self):
        self.assertEqual(len(self.notes), 0)
        self.notes.add("a")
        self.assertEqual(len(self.notes), 1)
        self.notes.add("b")
        self.assertEqual(len(self.notes), 2)

    def test_iter(self):
        self.notes.add("a")
        self.notes.add("b")
        texts = [n.get("note") for n in self.notes]
        self.assertEqual(texts, ["a", "b"])

    def test_getitem(self):
        self.notes.add("first")
        self.notes.add("second")
        self.assertEqual(self.notes[0].get("note"), "first")
        self.assertEqual(self.notes[1].get("note"), "second")
        self.assertEqual(self.notes[-1].get("note"), "second")

    def test_ok_valid(self):
        self.assertTrue(NMNotes.ok([{"note": "hey", "date": "111"}]))
        self.assertTrue(NMNotes.ok([{"date": "111", "note": "hey"}]))
        self.assertFalse(NMNotes.ok(None))
        self.assertFalse(NMNotes.ok([{"note": 123, "date": "111"}]))

    def test_ok_empty_list(self):
        self.assertTrue(NMNotes.ok([]))

    def test_ok_invalid_keys(self):
        self.assertFalse(NMNotes.ok([{"n": "hey", "date": "111"}]))
        self.assertFalse(NMNotes.ok([{"note": "hey", "d": "111"}]))

    def test_ok_invalid_values(self):
        self.assertFalse(NMNotes.ok([{"note": "hey", "date": None}]))
        self.assertTrue(NMNotes.ok([{"note": "hey", "date": "None"}]))

    def test_ok_missing_keys(self):
        self.assertFalse(NMNotes.ok([{"note": "hey"}]))
        self.assertFalse(NMNotes.ok([{"date": "111"}]))

    def test_ok_extra_keys(self):
        self.assertFalse(
            NMNotes.ok([{"note": "hey", "date": "111", "more": "1"}])
        )

    def test_ok_not_list(self):
        self.assertFalse(NMNotes.ok("not a list"))
        self.assertFalse(NMNotes.ok({"note": "hey", "date": "111"}))

    def test_equality(self):
        other = NMNotes()
        self.assertEqual(self.notes, other)
        self.notes.add("test")
        self.assertNotEqual(self.notes, other)

    def test_equality_after_deepcopy(self):
        self.notes.add("test")
        copied = copy.deepcopy(self.notes)
        self.assertEqual(self.notes, copied)

    def test_equality_not_nmnotes(self):
        self.assertNotEqual(self.notes, "not NMNotes")
        self.assertNotEqual(self.notes, [])

    def test_deepcopy(self):
        self.notes.add("original note")
        copied = copy.deepcopy(self.notes)
        self.assertEqual(len(copied), 1)
        self.assertEqual(copied[0].get("note"), "original note")
        # Verify independence
        copied.add("new note")
        self.assertEqual(len(self.notes), 1)
        self.assertEqual(len(copied), 2)


if __name__ == "__main__":
    unittest.main()
