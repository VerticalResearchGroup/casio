import re
import os
from dataclasses import dataclass

CASIO = os.environ.get('CASIO', '.')

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

def is_blacklisted(kname):
    for b in blacklist:
        if b in kname:
            return True
    return False

@dataclass
class FrameworkOp:
    name : str
    accel_time : float

fw_opname_map = {
    # TensorFlow Ops
    'Mul': 'mul',
    'Add': 'add',
    'Sub': 'sub',
    'RealDiv': 'div',
    'MatMul': 'matmul',
    'Relu': 'relu',
    'ReluGrad': 'relu-bwd',
    'Conv2D': 'conv2d',
    'Conv3D': 'conv3d',
    'Sum': 'sum',

    # PyTorch Ops
    'aten::mul': 'mul',
    'aten::add': 'add',
    'aten::sub': 'sub',
    'aten::div': 'div',
    'aten::mul_': 'mul',
    'aten::add_': 'add',
    'aten::sub_': 'sub',
    'aten::div_': 'div',
    'aten::mm': 'matmul',
    'aten::bmm': 'matmul',
    'aten::relu': 'relu',
    'aten::relu_': 'relu',
    'aten::conv2d': 'conv2d',
    'aten::conv3d': 'conv3d',
}

def normalize_fw_opname(opname):
    if opname in fw_opname_map:
        return fw_opname_map[opname]
    return opname

@dataclass
class SassInst:
    pc : str
    opcode : str
    thread_inst_exec : int

@dataclass
class Kernel:
    name : str
    # ncalls : int
    trace : list[SassInst]

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
                    kernels.append(kern)
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
                trace.append(SassInst(m.group(1), parse_sass_opcode(m.group(2)), int(m.group(6))))

    return kernels

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
