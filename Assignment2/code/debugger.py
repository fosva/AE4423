def debug(fn):
    def wrapper(*args, **kwargs):
        print(f"Invoking {fn.__name__}")
        print(f"  args: {args[:-1]}")
        print(f"  kwargs: {kwargs}")
        result = fn(*args, **kwargs)
        print(f"  returned {result}")
        return result
    return wrapper
