from .common import *


def get_bench_file(plat, app, batch):
    pat = f'{CASIO}/casio-results/{plat}/{app}/bench-{app}-train-b{batch}-n*.txt'

    ret_niter = 0
    ret_filename = None

    for filename in glob.glob(pat):
        niter = int(filename.replace('.txt', '').split('-')[-1][1:])
        if ret_filename is None or niter >  ret_niter:
            ret_niter = niter
            ret_filename = filename

    if ret_filename is not None: return ret_filename

    print(pat)
    assert False, f'Could not find bench file for {app} {plat} {batch}'

def throughput(plat, app, batch):
    bench_file = get_bench_file(plat, app, batch)
    with open(bench_file, 'r') as f:
        for line in f:
            if line.startswith('Throughput'): return float(line.split()[1])

    assert False, f'Could not find throughput in {bench_file}'
