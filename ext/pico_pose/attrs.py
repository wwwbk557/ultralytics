# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Attribute mapping for the Scheme-C 14-class Pico pose head.

The 14 classes are the Cartesian product of a 2-way colour and a 7-way
``role`` (sentry, hero, engineer, 3, 4, outpost, base)::

    class_id = colour * 7 + role

Order (flat class ids):
    0 blue_sentry, 1 blue_hero, 2 blue_engineer, 3 blue_3,
    4 blue_4, 5 blue_outpost, 6 blue_base,
    7 red_sentry, 8 red_hero, 9 red_engineer, 10 red_3,
    11 red_4, 12 red_outpost, 13 red_base.

Keep this mapping in one place: loss, export decoder, tests and the C++
contract must all use the same order.
"""

from __future__ import annotations

NUM_COLOR = 2
NUM_ROLE = 7
NUM_ATTR_CLASSES = NUM_COLOR * NUM_ROLE

# Scheme-C order: blue 0..6, red 7..13.
COLOR_BY_CLASS = (0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1)
ROLE_BY_CLASS = (0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6)

# role: 0=sentry, 1=hero, 2=engineer, 3=3号, 4=4号, 5=outpost, 6=base.
ROLE_NAMES = ("sentry", "hero", "engineer", "3", "4", "outpost", "base")
COLOR_NAMES = ("blue", "red")


def class_to_attrs(class_id: int) -> tuple[int, int]:
    """Return ``(colour, role)`` for a Scheme-C class id."""
    return COLOR_BY_CLASS[class_id], ROLE_BY_CLASS[class_id]


def attrs_to_class(colour: int, role: int) -> int:
    """Return the Scheme-C class id for ``(colour, role)``."""
    return int(colour) * NUM_ROLE + int(role)
