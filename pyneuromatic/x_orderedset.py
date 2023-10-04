from collections.abc import MutableSet


class OrderedSet(MutableSet):

    """Set-like object that maintains insertion order of items."""

    def __init__(self, iterable=()):
        self.items = dict.fromkeys(iterable, None)  # None results in Set?

    def __contains__(self, item):
        return item in self.items

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items.keys())

    def __repr__(self):
        return f"{type(self).__name__}({list(self.items)})"

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return len(self) == len(other) and all(x == y for x, y in zip(self, other))
        return super().__eq__(other)

    def add(self, item):
        self.items[item] = None

    def discard(self, item):
        self.items.pop(item, None)
