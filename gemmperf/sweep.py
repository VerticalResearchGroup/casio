from .common import *

gemms = set()
dev = torch.device('cuda:0')
dtype = torch.float16

mnk_range = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

for m in mnk_range:
    for n in mnk_range:
        for k in mnk_range:
            gemms.add(Matmul(m, n, k, NN))

print(f'Total number of GEMMS: {len(gemms)}')


with open('gemmsweep.yaml', 'w') as f:
    for gemm in gemms:
        print(gemm)
        print(f'{gemm}: {gemm.benchmark(dev, dtype)}', file=f)
