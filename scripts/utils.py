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
    'wavenet': 'WaveNet'
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

blacklist = {
    'redzone',
    'CUDA memset',
    'CUDA memcpy'
}

gemm_kernels = set(map(lambda s: s.strip(), open(f'{CASIO}/scripts/gemm-kernels.txt').readlines()))


def shorten_string(s, lim=40):
    if len(s) > lim:
        return s[:lim - 3] + '...'
    return s

def is_blacklisted(kname):
    for b in blacklist:
        if b in kname:
            return True
    return False

def is_gemm(kname): return kname in gemm_kernels

def get_bench_file(plat, app, batch):
    pat = f'{CASIO}/casio-results/{plat}/{app}/bench-{app}-train-b{batch}-n*.txt'

    ret_niter = 0
    ret_filename = None

    for filename in glob.glob(pat):
        niter = int(filename.replace('.txt', '').split('-')[-1][1:])
        if ret_filename is None or niter >  ret_niter:
            ret_niter = niter
            ret_filename = filename

    if ret_filename is not None: return ret_filename

    print(pat)
    assert False, f'Could not find bench file for {app} {plat} {batch}'

def throughput(plat, app, batch):
    bench_file = get_bench_file(plat, app, batch)
    with open(bench_file, 'r') as f:
        for line in f:
            if line.startswith('Throughput'): return float(line.split()[1])

    assert False, f'Could not find throughput in {bench_file}'

@dataclass
class FrameworkOp:
    name : str
    accel_time : float

fw_opname_map = {
    # TensorFlow Ops
    'Mul': 'mul',
    'Add': 'add',
    'AddV2': 'add',
    'Sub': 'sub',
    'RealDiv': 'div',
    'MatMul': 'matmul',
    'Relu': 'relu',
    'Tanh': 'tanh',
    'Conv2D': 'conv',
    'Conv2DBackpropInput': 'conv-bwd',
    'Conv2DBackpropFilter': 'conv-bwd',
    'Conv3D': 'conv',
    'Conv3DBackpropInput': 'conv-bwd',
    'Conv3DBackpropFilter': 'conv-bwd',
    'Sum': 'sum',
    'Transpose': 'transpose',
    'DynamicStitch': 'dynamic_stitch',

    # PyTorch Ops
    'mm': 'matmul',
    'bmm': 'matmul',
    'linear': 'matmul',
    'conv2d': 'conv',
    'conv3d': 'conv',
    'conv1d': 'conv',
    'lstm_cell': 'lstm',
    'convolution_backward': 'conv-bwd',
    '_softmax_backward_data': 'softmax-bwd',
    'native_batch_norm_backward': 'batch_norm-bwd',
    'native_layer_norm_backward': 'layer_norm-bwd',
}

def get_large_batch_size(plat, query_app):
    batch_sizes = {}

    with open(f'{CASIO}/casio-results/summaries/{plat}-large-batch-list') as f:
        for line in f:
            [plat, app, batchstr] = line.strip().split('/')
            batch = int(batchstr.split('-')[-1])
            batch_sizes[app] =  batch

    return batch_sizes[query_app]

# Start (ns),Duration (ns),CorrId,GrdX,GrdY,GrdZ,BlkX,BlkY,BlkZ,Reg/Trd,StcSMem (MB),DymSMem (MB),Bytes (MB),Throughput (MBps),SrcMemK       d,DstMemKd,Device,Ctx,Strm,Name

@dataclass
class NsysKernel:
    name : str
    time_ns : float
    num_threads : int

    @property
    def is_gemm(self): return is_gemm(self.name)

    def __repr__(self):
        return f'Kernel(name={shorten_string(self.name)}, {self.num_threads} threads, {self.time_ns}ns )'

nsys_trace_regex = r'(\d*),(\d*),(\d*),(\d*),(\d*),(\d*),(\d*),(\d*),(\d*),(\d*),(\d*\.?\d*),(\d*\.?\d*),[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,\"?([^"]+)\"?'

def get_nsys_gputrace_file(plat : str, app : str, batch : int):
    return f'{CASIO}/casio-results/summaries/{plat}/{app}/batch-{batch}_gputrace.csv.gz'

def parse_nsys_line(line):
    m = re.match(nsys_trace_regex, line.strip())
    if m is None:
        assert False, f'Failed to parse line: {line}'

    for i in [4, 5, 6, 7, 8, 9]:
        if m.group(i) == '': return None

    num_threads = np.prod([int(m.group(i)) for i in [4, 5, 6, 7, 8, 9]])
    kname = m.group(13)
    return NsysKernel(kname.strip(), float(m.group(2)), num_threads)

def read_nsys_trace(nsys_trace_file):
    with gzip.open(nsys_trace_file,'rt') as f:
        next(f)
        return list(
            filter(
                lambda x: x is not None,
                map(
                    parse_nsys_line,
                    filter(
                        lambda line: not is_blacklisted(line),
                        f))))


def parse_nsys_kernsum(line):
    # Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
    regex = r'([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),(.*)'
    m = re.match(regex, line)
    assert m is not None, f'Failed to parse line: "{line}"'
    return m.group(9)

# Mike is in a hurry here. I'm sorry for the duplicate code
def parse_nsys_kernsum2(line):
    # Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
    regex = r'([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),(.*)'
    m = re.match(regex, line)
    assert m is not None, f'Failed to parse line: "{line}"'
    return float(m.group(2)), int(m.group(3)), m.group(9)

def get_nsys_niter(plat, app, batch):
    nsys_file = None
    for filename in glob.glob(f'{CASIO}/casio-results/{plat}/{app}/nsys*b{batch}-*.nsys-rep'):
        nsys_file = filename
        break

    assert nsys_file is not None, f'Failed to find nsys file for {plat}/{app} batch {batch}'

    return int(nsys_file.replace('.nsys-rep', '').split('-')[-1][1:])


def normalize_fw_opname(opname):
    if opname.endswith('_'): opname = opname[:-1]

    if opname.startswith('aten::'):
        opname = opname[6:]

    if opname in fw_opname_map:
        return fw_opname_map[opname]

    if opname.endswith('Grad'):
        fwd_opname = opname[:-4]
        if fwd_opname in fw_opname_map:
            return fw_opname_map[fwd_opname] + '-bwd'

    return opname

@dataclass
class SassInst:
    pc : str
    opcode : str
    inst_exec : int
    thread_inst_exec : int


@dataclass
class Kernel:
    name : str
    # ncalls : int
    trace : list[SassInst]

    def to_feature_vector(self, opcode_map : dict[str, int]):
        features = np.zeros(len(opcode_map))
        for inst in self.trace:
            features[opcode_map[inst.opcode]] += inst.inst_exec

        return features

def get_ncu_raw_file(plat : str, app : str, batch : int, samp : str = '10th'):
    return f'{CASIO}/casio-results/{plat}/{app}/ncu-{samp}-{app}-train-b{batch}-raw.txt'

def read_ncu_raw_file(filename):
    with open(filename) as f:
        line = next(f)
        while not line.startswith('"ID","Process ID","Process Name",'):
            line = next(f)

        if not is_blacklisted(line): yield line
        next(f)

        for line in f:
            if not is_blacklisted(line): yield line

class Reader(object):
    def __init__(self, g):
        self.g = g
    def read(self, n=0):
        try:
            return next(self.g)
        except StopIteration:
            return ''

def read_ncu_raw_file_numpy(filename, cols):
    df = pd.read_csv(
        Reader(read_ncu_raw_file(filename)),
        low_memory=False,
        thousands=r',')

    names = list(df['Kernel Name'].values)
    data = df.filter(cols, axis=1)
    return names, data.to_numpy()

def parse_sass_opcode(raw_opcode):
    opcode = raw_opcode[5:].split(' ')[0].strip()
    if len(opcode) <= 1: opcode = raw_opcode[6:].split(' ')[0].strip()
    assert len(opcode) > 1, f'Failed to parse opcode from {raw_opcode}'
    return opcode

def kernels_are_equal(k1, k2):
    for i, (i1, i2) in enumerate(zip(k1.trace, k2.trace)):
        if i1 != i2:
            print(f'Kernel {k1.name} has different traces at index {i}!')
            print(f'  {i1} != {i2}')
            return False

    return True

def get_ncu_sass_file(plat : str, app : str, batch : int, samp : str = '10th'):
    return f'{CASIO}/casio-results/{plat}/{app}/ncu-{samp}-{app}-train-b{batch}-sass.txt'

def parse_ncu_sass(filename):
    with open(filename) as file:
        kernels = []
        kname = None
        trace = None
        capture = False

        for line in file:
            if line.startswith('"Kernel Name"'):
                if capture:
                    kern = Kernel(kname, trace)
                    if not is_blacklisted(kname): kernels.append(kern)
                    capture = False

                m = re.match(r'"Kernel Name",\s*"(.+)"', line)
                assert m, f'Failed to parse kernel name from {line}'
                kname = m.group(1)

                ignore = False
                for b in blacklist:
                    if b in kname:
                        ignore = True

                if not ignore:
                    capture = True
                    trace = []

            elif capture and not line.startswith('"Address","Source"'):
                m = re.match(r'^\"(\w+)\",\"([^\"]+)\",\"(\d+)\",\"(\d+)\",\"(\d+)\",\"(\d+)\"', line)
                assert m is not None, line
                trace.append(SassInst(m.group(1), parse_sass_opcode(m.group(2)), int(m.group(5)), int(m.group(6))))

    return kernels


def ncu_sass_opcodes(kernels : list[Kernel]):
    opcodes = set()
    for k in kernels:
        for inst in k.trace:
            opcodes.add(inst.opcode)

    return opcodes

def ncu_sass_stats(kernels : list[Kernel]):
    k : Kernel
    addr_map : dict[str, int] = {}
    opcode_map : dict[str, int] = {}
    total_dyn_inst = 0

    for k in kernels:
        inst : SassInst
        for inst in k.trace:
            if inst.pc not in addr_map: addr_map[inst.pc] = 0
            addr_map[inst.pc] += inst.thread_inst_exec

            if inst.opcode not in opcode_map: opcode_map[inst.opcode] = 0
            opcode_map[inst.opcode] += inst.thread_inst_exec

            total_dyn_inst += inst.thread_inst_exec

    return addr_map, opcode_map, total_dyn_inst


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

