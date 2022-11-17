#!/usr/bin/env python
import utils

NUM_BINS = 19
bin_str_to_idx = {
    'L**': 0,
    'LLM': 19,
    'LLH': 20,
    'LML': 21,
    'LMM': 22,
    'LMH': 23,
    'LHL': 24,
    'LHM': 25,
    'LHH': 26,

    'MLL': 1,
    'MLM': 2,
    'MLH': 3,
    'MML': 4,
    'MMM': 5,
    'MMH': 6,
    'MHL': 7,
    'MHM': 8,
    'MHH': 9,

    'HLL': 10,
    'HLM': 11,
    'HLH': 12,
    'HML': 13,
    'HMM': 14,
    'HMH': 15,
    'HHL': 16,
    'HHM': 17,
    'HHH': 18,
}

colors = [
    '#2e8b57',
    '#696969',
    '#228b22',
    '#7f0000',
    '#800080',
    '#ff4500',
    '#ffa500',
    '#00fa9a',
    '#4169e1',
    '#00ffff',
    '#00bfff',
    '#0000ff',
    '#ff00ff',
    '#fa8072',
    '#ffff54',
    '#dda0dd',
    '#ff1493',
    '#ffe4c4'
]

def thread_bin(nthread):
    if nthread < 4000: return 'L'
    elif nthread > 32000: return 'H'
    else: return 'M'

def sm_bin(sm_pct):
    if sm_pct < 10: return 'L'
    elif sm_pct > 70: return 'H'
    else: return 'M'

def mem_bin(mem_pct):
    if mem_pct < 10: return 'L'
    elif mem_pct > 70: return 'H'
    else: return 'M'


def get_bin_str(nthread, sm_pct, mem_pct, collapse_l=True):
    bin_str = thread_bin(nthread) + sm_bin(sm_pct) + mem_bin(mem_pct)
    if collapse_l and bin_str[0] == 'L': return 'L**'
    return bin_str

def get_bin_str_vec(x): return get_bin_str(x[0], x[1], x[2])

def get_bin_idx(nthread, sm_pct, mem_pct):
    return bin_str_to_idx[get_bin_str(nthread, sm_pct, mem_pct)]


