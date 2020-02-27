import time

def timer(func):
    def wrapper(*args, **kw):
        since = time.time()
        result = func(*args, **kw)
        time_elapsed = time.time() - since
        if time_elapsed // 60 > 1:
            print(f'function {func.__name__} running time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        return result
    return wrapper