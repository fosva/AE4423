import logging


def debug(fn):
    def wrapper(*args, **kwargs):
        logging.debug(f"Invoking {fn.__name__}")
        logging.debug(f"  args: {args}")
        logging.debug(f"  kwargs: {kwargs}")
        result = fn(*args, **kwargs)
        logging.debug(f"  returned {result}")
        return result
    return wrapper
