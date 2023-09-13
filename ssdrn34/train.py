import torch
import torchvision

from utils import params
from utils.torch_wrapper import benchmark_wrapper2

from . import ssdrn34

device = torch.device(params.devname)
net = ssdrn34.SsdRn34_300().to(params.dtype_torch).to(device)
x = torch.randn((params.bs, 3, 300, 300), device=device, dtype=params.dtype_torch)
opt = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def roi():
    opt.zero_grad()
    [locs, confs] = net(x)
    loss = torch.sum(locs) + torch.sum(confs)
    loss.backward()
    opt.step()

benchmark_wrapper2(roi)
