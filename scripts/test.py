import utils
import utils.cache


@utils.cache.cache_pickle
def some_function(app, plat):
    return {
        'p100': app,
        'v100': app,
        'a100': plat,
    }


print(some_function('resnet50', 'p100'))
