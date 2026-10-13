from collections import defaultdict
from time import perf_counter

TIMERS = defaultdict(float)
COUNTERS = defaultdict(int)


def reset():
    TIMERS.clear()
    COUNTERS.clear()


def add_time(key, seconds):
    TIMERS[key] += seconds


def count(key, n=1):
    COUNTERS[key] += n


def snapshot():
    return {'timers': dict(TIMERS), 'counters': dict(COUNTERS)}


__all__ = ['perf_counter', 'reset', 'add_time', 'count', 'snapshot', 'TIMERS', 'COUNTERS']
