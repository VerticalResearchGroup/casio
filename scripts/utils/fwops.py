from .common import *

@dataclass
class FrameworkOp:
    name : str
    accel_time : float

fwop_blacklist = {
    '_arg_Placeholder',

}

def is_blacklisted_fwop(opname):
    for b in fwop_blacklist:
        if b in opname:
            return True
    return False

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
    'GatherV2_1': 'GatherV2',
    'GatherV2_2': 'GatherV2',

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

def normalize_fw_opname(opname):
    if '/' in opname: opname = opname.split('/')[-2]
    if opname.endswith('_'): opname = opname[:-1]
    if opname.startswith('aten::'): opname = opname[6:]
    if opname in fw_opname_map: return fw_opname_map[opname]

    if opname.endswith('Grad'):
        fwd_opname = opname[:-4]
        if fwd_opname in fw_opname_map:
            return fw_opname_map[fwd_opname] + '-bwd'

    return opname

def get_optrace_file_lb(plat, app):
    return f'{CASIO}/casio-results/postproc/{plat}/{app}/op-trace-large-batch.csv'

def read_optrace(optrace_file):
    trace = []
    with open(optrace_file) as f:
        for line in f:
            opname, accel_time_str = line.strip().split(',')
            accel_time = float(accel_time_str)
            if not is_blacklisted_fwop(opname):
                trace.append(FrameworkOp(
                    normalize_fw_opname(opname),
                    accel_time
                ))

    return trace
