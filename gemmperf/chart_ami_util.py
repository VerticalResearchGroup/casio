#!/usr/bin/env python

from .common import *
from .chartutils import *

def m(gg):
    if isinstance(gg, Matmul): return gg.M
    elif isinstance(gg, BatchMatmul): return gg.M
    elif isinstance(gg, Conv2D): return gg.B * gg.H * gg.W * gg.R * gg.S

def n(gg):
    if isinstance(gg, Matmul): return gg.N
    elif isinstance(gg, BatchMatmul): return gg.N
    elif isinstance(gg, Conv2D): return gg.K

def k(gg):
    if isinstance(gg, Matmul): return gg.K
    elif isinstance(gg, BatchMatmul): return gg.K
    elif isinstance(gg, Conv2D): return gg.C

a100_gemms = yaml.safe_load(open('casio-results/a100/gemmsweep.yaml'))
casio_gemms_raw = yaml.safe_load(open('casio-results/a100/gemmperf.yaml'))

gemms = list(set(map(lambda g: eval(g), a100_gemms.keys())))
casio_gemms = list(set(map(lambda g: eval(g), casio_gemms_raw.keys())))


gemms = list(filter(lambda g: m(g) == n(g), gemms))
# casio_gemms = list(filter(lambda g: m(g) == n(g), casio_gemms))


with figure(COL_WIDTH, 1.75, 1, 1) as (fig, ax):
    ami = np.array([g.ami for g in gemms])
    util = np.array([a100_gemms[repr(g)] / 312e12 for g in gemms])

    m_large = np.array([m(g) > 512 for g in gemms])
    n_large = np.array([n(g) > 512 for g in gemms])
    k_large = np.array([k(g) > 512 for g in gemms])


    cat1 = m_large & n_large & k_large
    cat2 = ~cat1 & m_large
    cat3 = ~cat1 & n_large
    cat4 = ~cat1 & k_large
    cat5 = ~cat1 & ~cat2 & ~cat3 & ~cat4


    ax.scatter(ami[cat1], util[cat1], s=8, c=colors[0], label='Large MNK', zorder=10)
    ax.scatter(ami[cat2], util[cat2], s=8, c=colors[1], label='Large M==N')
    ax.scatter(ami[cat4], util[cat4], s=8, c=colors[2], label='Large K')
    ax.scatter(ami[cat5], util[cat5], s=8, c=colors[3], label='Small MNK')

    casio_ami = np.array([g.ami for g in casio_gemms])
    casio_util = np.array([casio_gemms_raw[repr(g)] / 312e12 for g in casio_gemms])


    casio_m_large = np.array([m(g) > 512 for g in casio_gemms])
    casio_n_large = np.array([n(g) > 512 for g in casio_gemms])
    casio_k_large = np.array([k(g) > 512 for g in casio_gemms])
    casio_conv = np.array([isinstance(g, Conv2D) for g in casio_gemms])

    casio_cat1 = casio_m_large & casio_n_large & casio_k_large
    casio_cat2 = ~casio_cat1 & casio_m_large
    casio_cat3 = ~casio_cat1 & casio_n_large
    casio_cat4 = ~casio_cat1 & casio_k_large
    casio_cat5 = ~casio_cat1 & ~casio_cat2 & ~casio_cat3 & ~casio_cat4

    ax.scatter(casio_ami, casio_util, s=1, c='black', label='Ours', marker='*', zorder=1000)
    # ax.scatter(casio_ami[~casio_conv & casio_cat1], casio_util[~casio_conv & casio_cat1], s=6, marker='^', c=colors[0], edgecolors='black', linewidth=0.2, zorder=100)
    # ax.scatter(casio_ami[~casio_conv & casio_cat2], casio_util[~casio_conv & casio_cat2], s=6, marker='^', c=colors[1], edgecolors='black', linewidth=0.2, zorder=97)
    # ax.scatter(casio_ami[~casio_conv & casio_cat4], casio_util[~casio_conv & casio_cat4], s=6, marker='^', c=colors[2], edgecolors='black', linewidth=0.2, zorder=98)
    # ax.scatter(casio_ami[~casio_conv & casio_cat5], casio_util[~casio_conv & casio_cat5], s=6, marker='^', c=colors[3], edgecolors='black', linewidth=0.2, zorder=99)

    # ax.scatter(casio_ami[casio_conv & casio_cat1], casio_util[casio_conv & casio_cat1], s=6, marker='s', c=colors[0], edgecolors='black', linewidth=0.2, zorder=100)
    # ax.scatter(casio_ami[casio_conv & casio_cat2], casio_util[casio_conv & casio_cat2], s=6, marker='s', c=colors[1], edgecolors='black', linewidth=0.2, zorder=97)
    # ax.scatter(casio_ami[casio_conv & casio_cat4], casio_util[casio_conv & casio_cat4], s=6, marker='s', c=colors[2], edgecolors='black', linewidth=0.2, zorder=98)
    # ax.scatter(casio_ami[casio_conv & casio_cat5], casio_util[casio_conv & casio_cat5], s=6, marker='s', c=colors[3], edgecolors='black', linewidth=0.2, zorder=99)

    # : MNK Large (>256)
    # : Large M or N (>1024), small K, AMI > 300
    # : MNK Small (!C1 & !C2)

    plt.xlabel('AMI', fontsize=8)
    plt.ylabel('Utilization', fontsize=8)

    plt.semilogx()

    ax.set_ylim([-0.01, 0.7])
    ax.set_yticks(
        [0.0, 0.2, 0.4, 0.6],
        labels=['0%', '20%', '40%', '60%']
    )

    ax.tick_params(axis='both', labelsize=6)

    plt.legend(
        loc='upper left',
        frameon=False,
        ncol=1,
        # bbox_to_anchor=(0.5, 1.04),
        fontsize=7,
        markerscale=1.5
    )

    fig.tight_layout(pad=0.5)
