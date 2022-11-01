import re
from dataclasses import dataclass

blacklist = {
    'redzone',
    'CUDA memset',
    'CUDA memcpy'
}

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
