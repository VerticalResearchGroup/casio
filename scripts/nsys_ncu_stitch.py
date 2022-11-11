from dataclasses import dataclass
import utils
import gzip
import numpy as np
import re

# Start (ns),
# Duration (ns),
# CorrId,
# GrdX,
# GrdY,
# GrdZ,
# BlkX,
# BlkY,
# BlkZ,
# Reg/Trd,
# StcSMem (MB),
# DymSMem (MB),
# Bytes (MB),
# Throughput (MBps),
# SrcMemKd,
# DstMemKd,
# Device,
# Ctx,
# Strm,
# Name

nsys_trace_regex = r'(\d*),(\d*),(\d*),(\d*),(\d*),(\d*),(\d*),(\d*),(\d*),(\d*),(\d*\.?\d*),(\d*\.?\d*),[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,\"?([^"]+)\"?'


def stitch_nsys_ncu(nsys_trace_file, ncu_raw_file, cols=None):
    if cols is None: cols = list()

    ncu_knames, ncu_data = utils.read_ncu_raw_file_numpy(
        ncu_raw_file,
        [
            'gpu__time_duration.sum',
            'launch__thread_count',
        ] + cols)

    unique_ncu_knames = set(ncu_knames)
    print(f'Unique kernel names in NCU: {len(unique_ncu_knames)}')
    ncu_kname_to_idx = {
        k: [i for i in range(len(ncu_knames)) if ncu_knames[i] == k]
        for k in unique_ncu_knames
    }

    keys = list(ncu_kname_to_idx.keys())

    for k in keys:
        newk = k.replace('Eigen::half', 'long long')
        ncu_kname_to_idx[newk] = ncu_kname_to_idx[k]

    not_found = set()

    def lookup_kernel(kname, kthreads):
        if kname not in ncu_kname_to_idx:
            # print(f'WARNING: kernel {kname} not found in NCU file')
            nonlocal not_found
            not_found.add(kname)
            # print(ncu_kname_to_idx.keys())
            # assert False
            return np.zeros_like(ncu_data[0]) - 1

        idxs = ncu_kname_to_idx[kname]
        best_idx = None
        best_nthreads = None

        for idx in idxs:
            nthreads = ncu_data[idx, 1]
            if best_idx is None or np.abs(nthreads - kthreads) < np.abs(best_nthreads - kthreads):
                best_idx = idx
                best_nthreads = nthreads

        return ncu_data[best_idx]

    def parse_nsys_line(line):
        m = re.match(nsys_trace_regex, line)
        if m is None:
            assert False, f'Failed to parse line: {line}'

        for i in [4, 5, 6, 7, 8, 9]:
            if m.group(i) == '': return None, None

        num_threads = np.prod([int(m.group(i)) for i in [4, 5, 6, 7, 8, 9]])
        kname = m.group(13)
        return kname, num_threads

    tot_kernels = set()
    with gzip.open(nsys_trace_file,'rt') as f:
        next(f)

        stitched_data = []

        for line in f:
            kname, num_threads = parse_nsys_line(line)
            if kname is None: continue
            tot_kernels.add(kname)
            stitched_data.append(lookup_kernel(kname, num_threads))

    print(f'WARNING: {len(not_found)}/{len(tot_kernels)} kernels not found in NCU file')
    return stitched_data

