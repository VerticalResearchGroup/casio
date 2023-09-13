#!/usr/bin/env python

from .common import *
from .chartutils import *

a100_gemms = yaml.safe_load(open('casio-results/a100/gemmperf.yaml'))
v100_gemms = yaml.safe_load(open('casio-results/v100/gemmperf.yaml'))
p100_gemms = yaml.safe_load(open('casio-results/p100/gemmperf.yaml'))

gemms = set(a100_gemms.keys())


for g in gemms:
    p100_gemms[g]
    v100_gemms[g]
    a100_gemms[g]

    gg = eval(g)

    if isinstance(gg, Matmul):
        m, n, k = gg.M, gg.N, gg.K
    elif isinstance(gg, BatchMatmul):
        m, n, k = gg.M, gg.N, gg.K
    elif isinstance(gg, Conv2D):
        m, n, k = gg.H * gg.W * gg.R * gg.S, gg.K, gg.C


    print(f'{str(g).replace(",", "")}, {m}, {n}, {k}, {gg.ami}, {p100_gemms[g]}, {v100_gemms[g]}, {a100_gemms[g]}, {p100_gemms[g] / 21.2e12}, {v100_gemms[g] / 125e12}, {a100_gemms[g] / 312e12}')
