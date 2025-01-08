import numpy as np

def debug(fn):
    def wrapper(*args, **kwargs):
        #print(f"Invoking {fn.__name__}")
        time = args[2]
        timeslot = time//40
        depth = 0
        if "depth" in kwargs.keys():
            depth = kwargs["depth"]
        #print(" "*(int(depth)//40) + f"depth: {depth}, aircraft type: {args[0].ac_type}, time: {time}, route: {args[1]} -> {args[4]}")
        result = fn(*args, **kwargs)
        if result[0] > -np.inf:
            print(" "*(depth//40) + f"depth: {depth}, aircraft type: {args[0].ac_type}, time: {time}, route: {args[1]} -> {args[4]}")
            print(" "*(depth//40) + f"returned {result[:-1]}")
        return result
    return wrapper
