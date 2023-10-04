from collections.abc import Mapping, Iterable
from sys import getsizeof
from timeit import default_timer
import unittest


from orderedset import OrderedSet


class OrderedSetTests(unittest.TestCase):

    """Tests for OrderedSet."""

    def test_constructor_with_iterables(self):
        OrderedSet([1, 2, 3, 4])
        OrderedSet(n**2 for n in [1, 2, 3])

    def test_constructor_with_nothing(self):
        OrderedSet()

    def test_string_representations(self):
        numbers = OrderedSet(n**2 for n in [1, 2, 3, 4])
        empty = OrderedSet()
        self.assertEqual(str(numbers), "OrderedSet([1, 4, 9, 16])")
        self.assertEqual(repr(empty), "OrderedSet([])")

    def test_iterable(self):
        numbers = OrderedSet([1, 2, 3, 4])
        self.assertEqual(set(numbers), {1, 2, 3, 4})

    def test_uniqueness(self):
        numbers = OrderedSet([1, 3, 2, 4, 2, 1, 4, 5])
        self.assertEqual(sorted(numbers), [1, 2, 3, 4, 5])

    def test_maintains_order_and_uniqueness(self):
        string = "Hello world.  This string contains many characters in it."
        expected = "Helo wrd.Thistngcamy"
        characters = OrderedSet(string)
        self.assertEqual("".join(characters), expected)

    def test_length(self):
        numbers = OrderedSet([1, 2, 4, 2, 1, 4, 5])
        self.assertEqual(len(numbers), 4)
        self.assertEqual(len(OrderedSet('hiya')), 4)
        self.assertEqual(len(OrderedSet('hello there')), 7)

    def test_containment(self):
        numbers = OrderedSet([1, 2, 4, 2, 1, 4, 5])
        self.assertIn(2, numbers)
        self.assertNotIn(3, numbers)

    # To test the Bonus part of this exercise, comment out the following line
    # @unittest.expectedFailure
    def test_memory_and_time_efficient(self):
        with Timer() as small_set_timer:
            numbers = OrderedSet(9999 for _ in range(2000))
        with Timer() as large_set_timer:
            numbers2 = OrderedSet(9999 + i for i in range(2000))

        # Time efficient initializer
        self.assertGreater(
            small_set_timer.elapsed * 5,
            large_set_timer.elapsed,
        )

        # Memory efficient
        self.assertLess(get_size(numbers) * 100, get_size(numbers2))
        self.assertLess(get_size(numbers), 2000)

        with Timer() as beginning_lookup:
            9999 in numbers2
        with Timer() as end_lookup:
            -1 in numbers2

        # Time efficient containment checks
        self.assertGreater(
            beginning_lookup.elapsed * 1.5,
            end_lookup.elapsed,
        )

    # To test the Bonus part of this exercise, comment out the following line
    # @unittest.expectedFailure
    def test_equality(self):
        self.assertEqual(OrderedSet('abc'), OrderedSet('abc'))
        self.assertNotEqual(OrderedSet('abc'), OrderedSet('bac'))
        self.assertNotEqual(OrderedSet('abc'), OrderedSet('abcd'))
        self.assertEqual(OrderedSet('abc'), set('abc'))
        self.assertEqual(OrderedSet('bac'), set('abc'))
        self.assertNotEqual(OrderedSet('abc'), 'abc')
        self.assertNotEqual(OrderedSet('abc'), ['a', 'b', 'c'])

    # To test the Bonus part of this exercise, comment out the following line
    # @unittest.expectedFailure
    def test_add_and_discard(self):
        numbers = OrderedSet([1, 2, 3])
        numbers.add(3)
        self.assertEqual(len(numbers), 3)
        numbers.add(4)
        self.assertEqual(len(numbers), 4)
        numbers.discard(4)
        self.assertEqual(len(numbers), 3)
        numbers.discard(4)
        self.assertEqual(len(numbers), 3)

        # Time efficient adding
        with Timer() as small_set_timer:
            numbers = OrderedSet()
            for n in [9999 for _ in range(2000)]:
                numbers.add(n)
        with Timer() as large_set_timer:
            numbers2 = OrderedSet()
            for n in [9999 + i for i in range(2000)]:
                numbers2.add(n)

        # Time efficient adding
        self.assertGreater(
            small_set_timer.elapsed * 5,
            large_set_timer.elapsed,
        )


def get_size(obj, seen=None):
    """Return size of any Python object."""
    if seen is None:
        seen = set()
    size = getsizeof(obj)
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    if hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    if hasattr(obj, '__slots__'):
        size += sum(
            get_size(getattr(obj, attr), seen)
            for attr in obj.__slots__
            if hasattr(obj, attr)
        )
    if isinstance(obj, Mapping):
        size += sum(
            get_size(k, seen) + get_size(v, seen)
            for k, v in obj.items()
        )
    elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
        size += sum(get_size(item, seen) for item in obj)
    return size


class Timer:

    """Context manager to time a code block."""

    def __enter__(self):
        self.start = default_timer()
        return self

    def __exit__(self, *args):
        self.end = default_timer()
        self.elapsed = self.end - self.start


if __name__ == "__main__":
    unittest.main(verbosity=3)
