#!/usr/bin/env python3

from enum import Enum
import os
import consts.resource_paths

class RangeFloat(object):
    def __init__(self, start, end):
        """
        Initialize the start end

        Args:
            self: (todo): write your description
            start: (int): write your description
            end: (int): write your description
        """
        self.start = start
        self.end = end

    def __eq__(self, other):
        """
        Determine whether two intervals are equal.

        Args:
            self: (todo): write your description
            other: (todo): write your description
        """
        return self.start <= other <= self.end

    def __contains__(self, item):
        """
        Return true if item is contained in the list.

        Args:
            self: (todo): write your description
            item: (str): write your description
        """
        return self.__eq__(item)

    def __iter__(self):
        """
        Iterate over all iterates of the iterable.

        Args:
            self: (todo): write your description
        """
        yield self

    def __str__(self):
        """
        Return the string representation of this object.

        Args:
            self: (todo): write your description
        """
        return '[{0},{1}]'.format(self.start, self.end)

class PrintColors(Enum):
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    WARNING = "\033[1;5;31m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

def cli_print(msg, print_color):
    """
    Prints to console with input print color type
    """
    if not isinstance(print_color, PrintColors):
        raise ValueError("Must use PrintColors type in cli_print")
    print("{0}{1}{2}".format(print_color.value, msg, PrintColors.ENDC.value))
