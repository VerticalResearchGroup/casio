import torch
import torchvision

from utils import params
from utils.torch_wrapper import benchmark_wrapper2

from . import ssdrn34

device = torch.device(params.devname)
net = ssdrn34.SsdRn34_1200().to(params.dtype_torch).to(device)
x = torch.randn((params.bs, 3, 1200, 1200), device=device, dtype=params.dtype_torch)

def roi():
    [locs, confs] = net(x)

benchmark_wrapper2(roi)
