from .common import *

yd = yaml.safe_load(open('gemms.yaml'))

gemms = set()

for k in yd:
    print(k)

    for b in batch_ranges[k]:
        for v in yd[k]:
            gemm = eval(v, globals(), dict(B=b))
            print(f'{k}-{b}: {gemm}')
            gemms.add(gemm)


print(f'Total number of GEMMS: {len(gemms)}')

for gemm in gemms:
    print(gemm)
    gemm.benchmark()
