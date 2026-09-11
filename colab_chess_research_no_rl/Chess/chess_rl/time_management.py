import time


class SearchTimeout(Exception):
    """Internal unwind; every search push must have a finally-pop."""


def allocate_ms(time_left_ms, fraction=0.05, cap_ms=1000.0):
    reserve = max(10.0, min(50.0, 0.1 * time_left_ms))
    return min(max(0.0, time_left_ms - reserve), fraction * time_left_ms + 250.0, cap_ms)


class Deadline:
    def __init__(self, seconds):
        self.end = seconds

    def check(self):
        if time.monotonic() >= self.end:
            raise SearchTimeout()

    def remaining_ms(self):
        return max(0.0, (self.end - time.monotonic()) * 1000)
