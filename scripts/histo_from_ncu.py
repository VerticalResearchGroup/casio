import pandas as pd
import sys
import tempfile

# class ReadCsvBuffer(ReadBuffer[AnyStr_cov], Protocol[AnyStr_cov]):
#     def __iter__(self) -> Iterator[AnyStr_cov]: ...
#     def fileno(self) -> int: ...
#     def readline(self) -> AnyStr_cov: ...
#     @property
#     def closed(self) -> bool: ...


def read_ncu_file(filename):
    with open(filename) as f:
        line = next(f)
        while not line.startswith('"ID","Process ID","Process Name",'):
            line = next(f)

        yield line
        next(f)

        for line in f:
            yield line


class Reader(object):
    def __init__(self, g):
        self.g = g
    def read(self, n=0):
        try:
            return next(self.g)
        except StopIteration:
            return ''

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

headers=['cumm-fraction-time', 'metric', 'fraction-time'] + launch_stats

def get_histograms(ncu_raw_file):
    df = pd.read_csv(
        Reader(read_ncu_file(ncu_raw_file)), low_memory=False, thousands=r',')

    print(df)

    df2 = df.filter(stats_of_interest, axis=1)
    df3 = df.filter(launch_stats, axis=1)

    averages = {}
    flamegraphs = {}
    nbins = 50
    total_cycles = 0

    for i,y in enumerate(df2['gpc__cycles_elapsed.max']):
        if (i != 0):
            # y = y.replace(",", "")
            total_cycles = total_cycles + int(y)

    for x in stats_of_interest:
        if (x == 'gpc__cycles_elapsed.max'):
            continue
        avg = 0.0
        flamegraphs[x] = []

        running_c = 0
        for i, y in enumerate(df2[x]):
            if i == 0: continue
            bin = int(float(y)/(100/nbins))
            if bin > nbins:
                print("error ", y)
            else:
                c = int(df2['gpc__cycles_elapsed.max'][i])
                f = c/total_cycles
                avg = avg + f * float(y)

                running_c += c
                g = running_c/total_cycles
                z = [g,float(y),f]
                for l in launch_stats:
                    z.append(str(df3[l][i]))
                flamegraphs[x].append(','.join(map(lambda x: f'"{x}"', z)))
        averages[x] = avg

    return averages, flamegraphs

assert len(sys.argv) == 3, 'Usage: histo_from_ncu.py ncu_raw_file.txt outputprefix'

averages, flamegraphs = get_histograms(sys.argv[1])
output_prefix = sys.argv[2]

with open(f'{output_prefix}feature-avg.csv', 'w') as avg_file:
    for x in stats_of_interest:
        if x in ignore_list: continue

        print(x)

        with open(f'{output_prefix}flame.{x}.csv', 'w') as f:
            print(','.join(headers), file=f)
            for j in flamegraphs[x]: print(j, file=f)
            f.close()

        print(f'{x}, {averages[x]}', file=avg_file)
