import numpy as np

def debug(fn):
    def wrapper(*args, **kwargs):
        print(*args[1:4])
        result = fn(*args, **kwargs)
        print(f"returned {result}")
        return result
    return wrapper
