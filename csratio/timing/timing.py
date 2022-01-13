from time import process_time as timer


def timeit(func, *args):
    start = timer()
    result = func(*args)
    end = timer()
    print(f"time for {func.__name__}: ", end - start)
    return result
