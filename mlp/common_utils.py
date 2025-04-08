import math
import time


# Simple formating of a time in human redable.
def format_time(ns: int) -> str:
    if ns >= 60 * 1e9:
        # Over a minute
        return time.strftime("%H:%M:%S", time.gmtime(ns / 1e9))
    exponent = math.floor(math.log10(ns))
    units = ["ns", "μs", "ms", "s"]
    order = exponent // 3
    unit = units[order]
    precision = 3 if order >= 2 else 0
    return f"{ns * 10 ** (-3 * order):.{precision}f}{unit}"