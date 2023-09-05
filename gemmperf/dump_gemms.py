#!/usr/bin/env python

from .common import *
from .chartutils import *

a100_gemms = yaml.safe_load(open('casio-results/a100/gemmperf.yaml'))
v100_gemms = yaml.safe_load(open('casio-results/v100/gemmperf.yaml'))
p100_gemms = yaml.safe_load(open('casio-results/p100/gemmperf.yaml'))

gemms = set(a100_gemms.keys())

def ami(gemm):
    return eval(gemm).ami


for g in gemms:
    p100_gemms[g]
    v100_gemms[g]
    a100_gemms[g]
    print(f'{str(g).replace(",", "")}, {ami(g)}, {p100_gemms[g]}, {v100_gemms[g]}, {a100_gemms[g]}, {p100_gemms[g] / 21.2e12}, {v100_gemms[g] / 125e12}, {a100_gemms[g] / 312e12}')
