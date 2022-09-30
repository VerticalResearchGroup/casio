import torch
import cudaprofile
from torch.profiler import profile, record_function, ProfilerActivity
import time
import platform
import os

import params

def benchmark_wrapper(appname, roi):
    outdir = f'/nobackup/medavies/casio/output/{platform.node()}/{appname}'
    os.makedirs(outdir, exist_ok=True)

    print(f'Warmup with {params.nw} Iters')
    for i in range(params.nw): roi()

    print(f'Running {params.ni} Iters')
    tt0 = time.perf_counter()
    if params.mode in {'ncu', 'nsys'}:
        torch.cuda.synchronize()
        cudaprofile.start()
        for i in range(params.ni): roi()
        torch.cuda.synchronize()
        cudaprofile.stop()

    elif params.mode == 'prof':
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
            torch.cuda.synchronize()
            for i in range(params.ni): roi()
            torch.cuda.synchronize()

        print(prof \
            .key_averages(group_by_input_shape=False, group_by_stack_n=0) \
            .table(sort_by="cuda_time_total", row_limit=-1, top_level_events_only=False))

        print(f'Saving trace to {outdir}/trace-b{params.bs}-n{params.ni}.json')
        prof.export_chrome_trace(f'{outdir}/trace-b{params.bs}-n{params.ni}.json')

    else:
        t0 = time.perf_counter()
        torch.cuda.synchronize()
        for i in range(params.ni): roi()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        print(f'Throughput: {params.ni * params.bs / (t1 - t0)}')

    tt1 = time.perf_counter()

    print(f'Total Time: {tt1 - tt0}')
