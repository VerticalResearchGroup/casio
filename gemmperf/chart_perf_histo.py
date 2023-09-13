#!/usr/bin/env python

from .common import *
from .chartutils import *

a100_gemms = yaml.safe_load(open('casio-results/a100/gemmperf.yaml'))
v100_gemms = yaml.safe_load(open('casio-results/v100/gemmperf.yaml'))
p100_gemms = yaml.safe_load(open('casio-results/p100/gemmperf.yaml'))

gemms = set(a100_gemms.keys())


with figure(COL_WIDTH, 1.3, 1, 3, 'gemmperf-hist', sharey=True) as (fig, axs):
    gemms = sorted(list(gemms))
    gemm_names = [str(i) for i, g in enumerate(gemms)]
    data = np.array([
        [p100_gemms[g] / 21.2e12, v100_gemms[g] / 125e12, a100_gemms[g] / 312e12]
        for g in gemms
    ]).transpose().clip(0, 1)

    axs[0].hist(data[0], bins=20, range=(0, 1), density=False, color=colors[0], label='P100', edgecolor='black', linewidth=0.5)
    axs[1].hist(data[1], bins=20, range=(0, 1), density=False, color=colors[1], label='V100', edgecolor='black', linewidth=0.5)
    axs[2].hist(data[2], bins=20, range=(0, 1), density=False, color=colors[2], label='A100', edgecolor='black', linewidth=0.5)

    axs[0].semilogy()
    axs[1].semilogy()
    axs[2].semilogy()

    axs[0].set_xlim(0, 1)
    axs[1].set_xlim(0, 1)
    axs[2].set_xlim(0, 1)


    axs[0].set_xticks([0, 0.5, 1], labels=['0', '50', '100'])
    axs[1].set_xticks([0, 0.5, 1], labels=['0', '50', '100'])
    axs[2].set_xticks([0, 0.5, 1], labels=['0', '50', '100'])

    axs[0].set_ylabel('Count of Kernels', fontsize=6)

    axs[0].set_title('P100', fontsize=8)
    axs[1].set_title('V100', fontsize=8)
    axs[2].set_title('A100', fontsize=8)


    axs[0].tick_params(axis='both', labelsize=6)
    axs[1].tick_params(axis='both', labelsize=6)
    axs[2].tick_params(axis='both', labelsize=6)

    axs[1].set_xlabel('GEMM performance (Percentage of Peak)', fontsize=8)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.2, left=0.13, right=0.97, top=0.86, bottom=0.3)
