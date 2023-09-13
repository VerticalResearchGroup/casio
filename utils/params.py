import os

print(f'==================================================================')
print(f'Benchmark Parameters:')
casio = os.environ.get('CASIO', None)
assert casio is not None, 'CASIO environment variable not set (REQUIRED)'

plat = os.environ.get('PLAT', None)
assert plat is not None, 'PLAT environment variable not set (REQUIRED)'

print(f'  Casio Directory: {casio}')

appname = os.environ.get('APP', None)
print(f'  Application Name: {appname}')

devname = os.environ.get('DEV', 'cuda:0')
print(f'  Device Name: {devname}')

dev_id = None
devname_parts = devname.split(':')
if len(devname_parts) > 1:
    dev_id = int(devname_parts[1])

mode = os.environ.get('MODE', 'bench')
print(f'  Mode: {mode}')

bs = int(os.environ.get('BS', '1'))
print(f'  Batch Size: {bs}')

dtype_str = os.environ.get('DT', 'FP16')
print(f'  Data Type: {dtype_str}')

nw = int(os.environ.get('NW', '30'))
print(f'  Num Warmup Passes: {nw}')

ni = int(os.environ.get('NI', '30'))
print(f'  Num Iters: {ni}')
print(f'==================================================================')

def __getattr__(name):
    if name == 'dtype_torch':
        import torch
        if dtype_str == 'FP16': return torch.float16
        elif dtype_str == 'FP32': return torch.float32
        elif dtype_str == 'I8': return torch.int8
        else: raise ValueError(f'Unknown data type: {dtype_str}')

    elif name == 'dtype_tf':
        import tensorflow as tf
        if dtype_str == 'FP16': return tf.float16
        elif dtype_str == 'FP32': return tf.float32
        elif dtype_str == 'I8': return tf.int8
        else: raise ValueError(f'Unknown data type: {dtype_str}')

    else: raise AttributeError(f'Unknown attribute: {name}')
