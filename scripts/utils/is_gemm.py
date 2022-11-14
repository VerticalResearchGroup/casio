from .common import *

def strip_types(s : str):
    return s \
        .replace('long long', '') \
        .replace('long', '') \
        .replace('Eigen::half', '') \
        .replace('__half', '')

gemm_kernels = set(map(lambda s: s.strip(), open(f'{CASIO}/scripts/gemm-kernels.txt').readlines()))

addl_gemm_kernels = set()
for kname in gemm_kernels:
    addl_gemm_kernels.add(strip_types(kname))


def is_gemm(kname):
    if kname in gemm_kernels: return True
    if strip_types(kname) in addl_gemm_kernels: return True
    return False
