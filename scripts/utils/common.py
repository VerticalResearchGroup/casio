import re
import os
import gzip
from dataclasses import dataclass
import numpy as np
import pandas as pd
import glob

CASIO = os.environ.get('CASIO', '.')

COL_WIDTH = (8.5 - 1.5 - 0.25) / 2
TEXT_WIDTH = 8.5 - 1.5

apps = [
    'meshgraphnets-cfd',
    'meshgraphnets-cloth',
    'muzero',
    'nerf',
    # 'pinn-ac',
    # 'pinn-kdv',
    'pinn-navier-stokes',
    'pinn-schrodinger',
    'qdtrack',
    'swin-swinv2_base_patch4_window12_192_22k',
    # 'swin-swinv2_base_patch4_window16_256',
    # 'swin-swinv2_large_patch4_window12_192_22k',
    'swin-swinv2_large_patch4_window12to24_192to384_22kto1k_ft',
    'tabnet',
    'tacotron2',
    'wavenet'
]

app_pretty_names = {
    'meshgraphnets-cfd': 'MGN-CFD',
    'meshgraphnets-cloth': 'MGN-Cloth',
    'muzero': 'MuZero',
    'nerf': 'NeRF',
    'pinn-ac': 'PINN-AC',
    'pinn-kdv': 'PINN-KdV',
    'pinn-navier-stokes': 'PINN-NS',
    'pinn-schrodinger': 'PINN-Schr.',
    'qdtrack': 'QDTrack',
    'swin-swinv2_base_patch4_window12_192_22k': 'SwinV2-B-P',
    'swin-swinv2_base_patch4_window16_256': 'SwinV2-B',
    'swin-swinv2_large_patch4_window12_192_22k': 'SwinV2-L-PT',
    'swin-swinv2_large_patch4_window12to24_192to384_22kto1k_ft': 'SwinV2-L-F',
    'tabnet': 'TabNet',
    'tacotron2': 'Tacotron2',
    'wavenet': 'WaveNet',
    'resnet50': 'Resnet50',
    'bert': 'BERT'
}

plats = ['p100', 'v100', 'a100']

stats_of_interest = [
    'gpc__cycles_elapsed.max',
    'sm__throughput.avg.pct_of_peak_sustained_elapsed',
    'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed',
    'l1tex__throughput.avg.pct_of_peak_sustained_active',
    'lts__throughput.avg.pct_of_peak_sustained_elapsed',
    'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed',
    'sm__issue_active.avg.pct_of_peak_sustained_elapsed',
    'sm__inst_executed.avg.pct_of_peak_sustained_elapsed',
    'sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed',
    'sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed',
    'sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed',
    'sm__inst_executed_pipe_adu.avg.pct_of_peak_sustained_elapsed',
    'sm__mio2rf_writeback_active.avg.pct_of_peak_sustained_elapsed',
    'sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_elapsed',
    'sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed',
    'sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed',
    'sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed',
    'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed'
]

ignore_list = [stats_of_interest[0]]

launch_stats = [
    'Kernel Name',
    'launch__block_dim_x',
    'launch__block_dim_y',
    'launch__block_dim_z',
    'launch__block_size',
    'launch__grid_dim_x',
    'launch__grid_dim_y',
    'launch__grid_dim_z',
    'launch__grid_size',
    'launch__occupancy_limit_blocks',
    'launch__occupancy_limit_registers',
    'launch__occupancy_limit_shared_mem',
    'launch__occupancy_limit_warps',
    'launch__occupancy_per_block_size',
    'launch__occupancy_per_register_count',
    'launch__occupancy_per_shared_mem_size',
    'launch__registers_per_thread',
    'launch__registers_per_thread_allocated',
    'launch__shared_mem_config_size',
    'launch__shared_mem_per_block',
    'launch__shared_mem_per_block_allocated',
    'launch__shared_mem_per_block_driver',
    'launch__shared_mem_per_block_dynamic',
    'launch__shared_mem_per_block_static',
    'launch__thread_count',
    'launch__waves_per_multiprocessor'
]


kern_blacklist = {
    'redzone',
    'CUDA memset',
    'CUDA memcpy'
}

def is_blacklisted(kname):
    for b in kern_blacklist:
        if b in kname:
            return True
    return False

def shorten_string(s, lim=40):
    if len(s) > lim:
        return s[:lim - 3] + '...'
    return s


def get_large_batch_size(plat, query_app):
    batch_sizes = {}

    with open(f'{CASIO}/casio-results/summaries/{plat}-large-batch-list') as f:
        for line in f:
            [plat, app, batchstr] = line.strip().split('/')
            batch = int(batchstr.split('-')[-1])
            batch_sizes[app] =  batch

    return batch_sizes[query_app]

class Reader(object):
    def __init__(self, g):
        self.g = g
    def read(self, n=0):
        try: return next(self.g)
        except StopIteration: return ''




def pick_clusters(dists, tol=0.05):
    rep_list = set()
    ignore_list = set()
    for i in range(len(dists)):
        if i in ignore_list: continue

        for j in range(len(dists)):
            if dists[i, j] > tol:
                if (j in ignore_list): continue
                else: rep_list.add(i)
            else: ignore_list.add(j)

    return rep_list

