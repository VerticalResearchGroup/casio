import torch

from utils import params
from utils.torch_wrapper import benchmark_wrapper2
from .model import unet3d as U

device = torch.device(params.devname)
net = U.Unet3D(1, 3, normalization='instancenorm', activation='relu').to(params.dtype_torch).to(device)
x = torch.randn((params.bs, 1, 128, 128, 128), device=device, dtype=params.dtype_torch)
opt = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def roi():
    opt.zero_grad()
    yp = net(x)
    loss = torch.sum(yp)
    loss.backward()
    opt.step()

benchmark_wrapper2(roi)