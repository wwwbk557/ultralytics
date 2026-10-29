# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Attribute mapping for the 14-class Pico pose head.

The 14 classes are the Cartesian product of a 2-way colour and a 7-way
``number`` (digit 1..5, guard, outpost)::

    class_id = colour * 7 + number

Keep this mapping in one place: the loss, export decoder, tests and the C++
contract must all use the same order.
"""

from __future__ import annotations

NUM_COLOR = 2
NUM_NUMBER = 7
NUM_ATTR_CLASSES = NUM_COLOR * NUM_NUMBER

# Current 14-class order:
# 0..4 blue_1..blue_5, 5 blue_guard, 6 blue_outpost,
# 7..11 red_1..red_5, 12 red_guard, 13 red_outpost.
COLOR_BY_CLASS = (0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1)
NUMBER_BY_CLASS = (0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6)

# number: 0..4 = digit 1..5, 5 = guard, 6 = outpost.
NUMBER_NAMES = ("1", "2", "3", "4", "5", "guard", "outpost")
COLOR_NAMES = ("blue", "red")


def class_to_attrs(class_id: int) -> tuple[int, int]:
    """Return ``(colour, number)`` for a 14-class id."""
    return COLOR_BY_CLASS[class_id], NUMBER_BY_CLASS[class_id]


def attrs_to_class(colour: int, number: int) -> int:
    """Return the 14-class id for ``(colour, number)``."""
    return int(colour) * NUM_NUMBER + int(number)
