import numpy as np

def debug(fn):
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        if args[2] < 965 and args[1] ==3:
            print(*args[1:4])
            print(f"returned {result[:3]}, {result[4:]}")
        return result
    return wrapper

