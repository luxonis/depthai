#!/usr/bin/env python3

from types import SimpleNamespace


class RangeFloat(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end

    def __contains__(self, item):
        return self.__eq__(item)

    def __iter__(self):
        yield self

    def __str__(self):
        return '[{0},{1}]'.format(self.start, self.end)


PrintColors = SimpleNamespace(
    HEADER="\033[95m",
    BLUE="\033[94m",
    GREEN="\033[92m",
    RED="\033[91m",
    WARNING="\033[1;5;31m",
    FAIL="\033[91m",
    ENDC="\033[0m",
    BOLD="\033[1m",
    UNDERLINE="\033[4m",
    BLACK_BG_RED="\033[1;31;40m",
    BLACK_BG_GREEN="\033[1;32;40m",
    BLACK_BG_BLUE="\033[1;34;40m",
)


def cliPrint(msg, print_color):
    print("{0}{1}{2}".format(print_color, msg, PrintColors.ENDC))
