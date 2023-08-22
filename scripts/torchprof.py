from dataclasses import dataclass
import re
import utils
import json

@dataclass
class AtenOp:
    name: str
    ti_us : float
    tf_us : float

@dataclass
class CudaLaunchKernel:
    corr : int
    ti_us : float
    tf_us : float
    accel_time_us : float = 0

def parse_torch_prof(filename):
    # print(f'Loading {filename}')
    jd = json.load(open(filename))
    aten_ops = []
    cuda_launch_kernels = []
    kernel_durations = {}

    # print(f'Parsing Events for {filename}...')
    for ev in jd['traceEvents']:
        if ev['name'].startswith('aten::'):
            aten_ops.append(AtenOp(ev['name'], ev['ts'], ev['ts'] + ev['dur']))

        elif ev['name'] == 'cudaLaunchKernel':
            cuda_launch_kernels.append(
                CudaLaunchKernel(
                    ev['args']['correlation'], ev['ts'], ev['ts'] + ev['dur']))

        elif 'cat' in ev and (ev['cat'] == 'Kernel' or ev['cat'] == 'kernel'):
            kernel_durations[ev['args']['correlation']] = ev['dur']

    cuda_launch_kernels = [
        CudaLaunchKernel(cl.corr, cl.ti_us, cl.tf_us, kernel_durations[cl.corr])
        for cl in cuda_launch_kernels
        if cl.corr in kernel_durations and kernel_durations[cl.corr] > 0
    ]

    aten_ops = sorted(aten_ops, key=lambda op: op.ti_us)
    cuda_launch_kernels = sorted(cuda_launch_kernels, key=lambda x: x.ti_us)

    deduped_aten_ops = []
    i = 0
    while i < len(aten_ops):
        deduped_aten_ops.append(aten_ops[i])
        tf_us = aten_ops[i].tf_us

        j = i + 1
        while j < len(aten_ops) and aten_ops[j].tf_us <= tf_us:
            j += 1

        i = j

    trace = []
    ki = 0
    for atop in deduped_aten_ops:
        accel_time_us = 0

        while ki < len(cuda_launch_kernels) and \
            cuda_launch_kernels[ki].ti_us < atop.ti_us:
            ki += 1

        while ki < len(cuda_launch_kernels) and \
            cuda_launch_kernels[ki].tf_us <= atop.tf_us:
            accel_time_us += cuda_launch_kernels[ki].accel_time_us
            ki += 1

        if accel_time_us > 0:
            trace.append(utils.FrameworkOp(atop.name, accel_time_us / 1e6))

    return trace
