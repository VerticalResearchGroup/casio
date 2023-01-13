import os
import functools
import pickle

def cache_list(etype):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            argss = '_'.join(str(arg) for arg in args)
            filename = f'cache/{func.__name__}.{argss}.cache'
            if os.path.exists(filename):
                # print(f'Loading {func.__name__} from cache')
                with open(filename, 'r') as f:
                    return [
                        etype(line.strip())
                        for line in f
                    ]
            else:
                print(f'Regenerating {func.__name__}...')
                ret = func(*args, **kwargs)
                assert isinstance(ret, list)
                # assert all(isinstance(x, etype) for x in ret), ret
                os.makedirs('cache', exist_ok=True)
                with open(filename, 'w') as f:
                    for x in ret:
                        print(x, file=f)
                return ret

        return wrapper
    return decorator

def cache_pickle(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        argss = '_'.join(str(arg) for arg in args)
        filename = f'cache/{func.__name__}.{argss}.cache.pkl'
        if os.path.exists(filename):
            # print(f'Loading {func.__name__} from cache')
            with open(filename, 'rb') as f:
                return pickle.load(f)
        else:
            print(f'Regenerating {func.__name__}...')
            ret = func(*args, **kwargs)

            os.makedirs('cache', exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(ret, f)

            return ret

    return wrapper
