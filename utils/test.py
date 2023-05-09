import time
from typing import Callable

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def test(func: Callable):
    print("Running:", func.__name__)
    t0 = time.time()
    try:
        func()
    except AssertionError as e:
        print(bcolors.FAIL + f"Failed error: {e}" + bcolors.ENDC)
    else:
        print(bcolors.OKGREEN + "Successful" + bcolors.ENDC)
    print(f"Time taken: {time.time() - t0:.2f}s")
    print(50*"-")