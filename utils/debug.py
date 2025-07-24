# Simple global switch for verbose prints
VERBOSE = True

def dprint(msg: str):
    if VERBOSE:
        print(msg)
