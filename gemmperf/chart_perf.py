#!/usr/bin/env python

from .common import *
from .chartutils import *

a100_gemms = yaml.safe_load(open('casio-results/a100/gemmperf.yaml'))
v100_gemms = yaml.safe_load(open('casio-results/v100/gemmperf.yaml'))
p100_gemms = yaml.safe_load(open('casio-results/p100/gemmperf.yaml'))

gemms = set(a100_gemms.keys())


with figure(10, 6, 3, 1, 'gemmperf') as (fig, axs):
    gemms = sorted(list(gemms))
    gemm_names = [str(i) for i, g in enumerate(gemms)]
    data = np.array([
        [p100_gemms[g] / 21.2e12, v100_gemms[g] / 125e12, a100_gemms[g] / 312e12]
        for g in gemms
    ]).transpose().clip(0, 1)

    multibars(axs[0], gemm_names, ['P100'], data[0, :].reshape(1, -1), locator_bins=4)
    multibars(axs[1], gemm_names, ['V100'], data[1, :].reshape(1, -1), locator_bins=4)
    multibars(axs[2], gemm_names, ['A100'], data[2, :].reshape(1, -1), locator_bins=4)

    axs[0].set_ylabel('P100')
    axs[1].set_ylabel('V100')
    axs[2].set_ylabel('A100')

    plt.xlabel('GEMM performance (fraction of peak)')
    fig.tight_layout()
