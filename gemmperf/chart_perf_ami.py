#!/usr/bin/env python

from .common import *
from .chartutils import *

a100_gemms = yaml.safe_load(open('casio-results/a100/gemmperf.yaml'))
v100_gemms = yaml.safe_load(open('casio-results/v100/gemmperf.yaml'))
p100_gemms = yaml.safe_load(open('casio-results/p100/gemmperf.yaml'))

gemms = set(a100_gemms.keys())

def ami(gemm):
    return eval(gemm).ami


with figure(10, 6, 1, 1, 'gemmperf-ami') as (fig, ax):
    axs = [ax]
    gemms = sorted(list(gemms))
    gemm_names = [str(i) for i, g in enumerate(gemms)]

    p100_data = np.array([[ami(g), p100_gemms[g] / 21.2e12] for g in gemms])
    v100_data = np.array([[ami(g), v100_gemms[g] / 125e12] for g in gemms])
    a100_data = np.array([[ami(g), a100_gemms[g] / 312e12] for g in gemms])


    axs[0].scatter(p100_data[:, 0], p100_data[:, 1], s=8, color=colors[0], label='P100')
    axs[0].scatter(v100_data[:, 0], v100_data[:, 1], s=8, color=colors[1], label='V100')
    axs[0].scatter(a100_data[:, 0], a100_data[:, 1], s=8, color=colors[2], label='A100')

    axs[0].loglog()
    # axs[1].loglog()
    # axs[2].loglog()

    # axs[0].set_ylabel('P100')
    axs[0].set_ylabel('GEMM performance (fraction of peak)')
    # axs[0].set_ylabel('A100')

    fig.legend(loc='lower right')

    plt.xlabel('Arithmetic Intensity (FLOPS/Byte)')
    # plt.ylabel('GEMM performance (fraction of peak)')
    fig.tight_layout()
