import os

print(f'==================================================================')
print(f'Benchmark Parameters:')
casio = os.environ.get('CASIO', None)
assert casio is not None, 'CASIO environment variable not set (REQUIRED)'

print(f'  Casio Directory: {casio}')

devname = os.environ.get('DEV', 'cuda:0')
print(f'  Device Name: {devname}')

mode = os.environ.get('MODE', 'bench')
print(f'  Mode: {mode}')

bs = int(os.environ.get('BS', '1'))
print(f'  Batch Size: {bs}')

nw = int(os.environ.get('NW', '30'))
print(f'  Num Warmup Passes: {nw}')

ni = int(os.environ.get('NI', '30'))
print(f'  Num Iters: {ni}')
print(f'==================================================================')
