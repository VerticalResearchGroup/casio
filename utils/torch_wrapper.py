import torch

from torch.profiler import profile, record_function, ProfilerActivity
import time
import platform
import os

try:
    import utils.cudaprofile as cudaprofile
    import utils.params as params
except ImportError:
    import cudaprofile
    import params

def benchmark_wrapper(appname, roi):
    outdir = f'{params.casio}/output/{params.plat}/{appname}'
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
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)

        t0 = time.perf_counter()
        torch.cuda.synchronize()
        ev_start.record()
        for i in range(params.ni): roi()
        ev_end.record()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        tt = ev_start.elapsed_time(ev_end) / 1000
        print(f'Throughput: {params.ni * params.bs / (tt)}')

    tt1 = time.perf_counter()

    print(f'Total Time: {tt1 - tt0}')

def benchmark_wrapper2(roi): return benchmark_wrapper(params.appname, roi)
