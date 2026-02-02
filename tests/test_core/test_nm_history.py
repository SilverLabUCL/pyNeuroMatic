#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for NMHistory centralized logging.

@author: jason
"""
import logging
import unittest

from pyneuromatic.core.nm_history import (
    NMHistory,
    NMHistoryBufferHandler,
    NMConsoleHandler,
)
from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.core.nm_object import NMObject
import pyneuromatic.core.nm_history as nmh
import pyneuromatic.core.nm_utilities as nmu

QUIET = True


class NMHistoryBufferHandlerTest(unittest.TestCase):
    """Test the custom buffer handler independently."""

    def setUp(self):
        self.handler = NMHistoryBufferHandler(maxlen=100)

    def test00_init(self):
        self.assertEqual(len(self.handler.buffer), 0)
        h = NMHistoryBufferHandler(maxlen=5)
        self.assertEqual(h._buffer.maxlen, 5)

    def test01_emit(self):
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=None,
            exc_info=None,
        )
        record.path = "nm.test"
        self.handler.emit(record)
        self.assertEqual(len(self.handler.buffer), 1)
        entry = self.handler.buffer[0]
        self.assertEqual(entry["message"], "test message")
        self.assertEqual(entry["path"], "nm.test")
        self.assertEqual(entry["level"], "INFO")
        self.assertIn("date", entry)

    def test02_buffer_overflow(self):
        h = NMHistoryBufferHandler(maxlen=3)
        for i in range(5):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="msg%d" % i,
                args=None,
                exc_info=None,
            )
            record.path = ""
            h.emit(record)
        self.assertEqual(len(h.buffer), 3)
        self.assertEqual(h.buffer[0]["message"], "msg2")
        self.assertEqual(h.buffer[2]["message"], "msg4")

    def test03_clear(self):
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=None,
            exc_info=None,
        )
        record.path = ""
        self.handler.emit(record)
        self.assertEqual(len(self.handler.buffer), 1)
        self.handler.clear()
        self.assertEqual(len(self.handler.buffer), 0)


class NMHistoryTest(unittest.TestCase):
    """Test the NMHistory wrapper class."""

    def setUp(self):
        self.h = NMHistory(buffer_size=1000, quiet=True)

    def tearDown(self):
        self.h.clear()

    def test00_init(self):
        self.assertIsInstance(self.h, NMHistory)
        self.assertEqual(len(self.h.buffer), 0)
        self.assertTrue(self.h.quiet)
        self.assertEqual(self.h.buffer_size, 1000)

    def test01_log_info(self):
        result = self.h.log("test message", path="nm.test")
        self.assertEqual(result, "nm.test: test message")
        self.assertEqual(len(self.h.buffer), 1)
        entry = self.h.buffer[0]
        self.assertEqual(entry["level"], "INFO")
        self.assertEqual(entry["message"], "test message")
        self.assertEqual(entry["path"], "nm.test")

    def test02_log_alert(self):
        result = self.h.log(
            "alert msg", title="ALERT", path="nm.test", level=logging.WARNING
        )
        self.assertEqual(result, "ALERT: nm.test: alert msg")
        entry = self.h.buffer[-1]
        self.assertEqual(entry["level"], "WARNING")
        self.assertEqual(entry["message"], "alert msg")

    def test03_log_error(self):
        result = self.h.log(
            "error msg", title="ERROR", path="nm.test", level=logging.ERROR
        )
        self.assertEqual(result, "ERROR: nm.test: error msg")
        entry = self.h.buffer[-1]
        self.assertEqual(entry["level"], "ERROR")
        self.assertEqual(entry["message"], "error msg")

    def test04_quiet_per_call(self):
        # messages should still go to buffer even when quiet=True
        self.h.log("quiet msg", quiet=True)
        self.assertEqual(len(self.h.buffer), 1)
        self.assertEqual(self.h.buffer[0]["message"], "quiet msg")

    def test05_quiet_property(self):
        self.assertTrue(self.h.quiet)
        self.h.quiet = False
        self.assertFalse(self.h.quiet)
        self.h.quiet = True
        self.assertTrue(self.h.quiet)

    def test06_clear(self):
        self.h.log("msg1")
        self.h.log("msg2")
        self.assertEqual(len(self.h.buffer), 2)
        self.h.clear()
        self.assertEqual(len(self.h.buffer), 0)

    def test07_buffer_overflow(self):
        h = NMHistory(buffer_size=5, quiet=True)
        for i in range(10):
            h.log("msg%d" % i)
        self.assertEqual(len(h.buffer), 5)
        self.assertEqual(h.buffer[0]["message"], "msg5")
        self.assertEqual(h.buffer[4]["message"], "msg9")

    def test08_no_path(self):
        result = self.h.log("bare message")
        self.assertEqual(result, "bare message")
        self.assertEqual(self.h.buffer[0]["path"], "")

    def test09_path_none(self):
        result = self.h.log("msg", path="NONE")
        self.assertEqual(result, "msg")
        self.assertEqual(self.h.buffer[0]["path"], "NONE")

    def test10_empty_title(self):
        result = self.h.log("msg", title="", path="nm.test")
        self.assertEqual(result, "nm.test: msg")

    def test11_multiple_messages(self):
        self.h.log("first")
        self.h.log("second")
        self.h.log("third")
        self.assertEqual(len(self.h.buffer), 3)
        messages = [e["message"] for e in self.h.buffer]
        self.assertEqual(messages, ["first", "second", "third"])


class NMHistoryIntegrationTest(unittest.TestCase):
    """Test NMHistory integration with NMManager and NMObject."""

    def setUp(self):
        self.nm = NMManager(quiet=QUIET)

    def test00_manager_has_history(self):
        self.assertIsInstance(self.nm.history, NMHistory)
        self.assertTrue(self.nm.history.quiet)

    def test01_history_captures_init(self):
        # NMManager.__init__ should have logged creation messages
        buf = self.nm.history.buffer
        self.assertTrue(len(buf) > 0)
        messages = [e["message"] for e in buf]
        found_created = any("created NM manager" in m for m in messages)
        found_project = any("current NM project" in m for m in messages)
        found_tool = any("current NM tool" in m for m in messages)
        self.assertTrue(found_created)
        self.assertTrue(found_project)
        self.assertTrue(found_tool)

    def test02_nmu_history_routes_to_buffer(self):
        self.nm.history.clear()
        nmh.history("test via nmu", path="nm.test", quiet=True)
        buf = self.nm.history.buffer
        self.assertEqual(len(buf), 1)
        self.assertEqual(buf[0]["message"], "test via nmu")
        self.assertEqual(buf[0]["path"], "nm.test")

    def test03_history_routes_to_buffer(self):
        """Test that nmh.history() routes INFO messages to buffer."""
        self.nm.history.clear()
        nmh.history("history msg", path="nm.test", quiet=True)
        buf = self.nm.history.buffer
        self.assertEqual(len(buf), 1)
        self.assertIn("history msg", buf[0]["message"])
        self.assertEqual(buf[0]["level"], "INFO")

    def test04_alert_routes_to_buffer(self):
        """Test that nmh.history() with ALERT title routes to buffer with WARNING level."""
        self.nm.history.clear()
        nmh.history("alert msg", title="ALERT", red=True, quiet=True)
        buf = self.nm.history.buffer
        self.assertEqual(len(buf), 1)
        self.assertEqual(buf[0]["level"], "WARNING")

    def test05_error_routes_to_buffer(self):
        """Test that nmh.history() with ERROR title routes to buffer with ERROR level."""
        self.nm.history.clear()
        nmh.history("error msg", title="ERROR", red=True, quiet=True)
        buf = self.nm.history.buffer
        self.assertEqual(len(buf), 1)
        self.assertEqual(buf[0]["level"], "ERROR")

    def test06_buffer_preserves_order(self):
        self.nm.history.clear()
        nmh.history("first", quiet=True)
        nmh.history("second", quiet=True)
        nmh.history("third", quiet=True)
        buf = self.nm.history.buffer
        messages = [e["message"] for e in buf]
        self.assertEqual(messages, ["first", "second", "third"])


if __name__ == "__main__":
    unittest.main()
