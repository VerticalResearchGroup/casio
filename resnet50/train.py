import torch
import torchvision

from utils import params
from utils.torch_wrapper import benchmark_wrapper2

device = torch.device(params.devname)
net = torchvision.models.resnet50(pretrained=False).to(params.dtype_torch).to(device)
x = torch.randn((params.bs, 3, 224, 224), device=device, dtype=params.dtype_torch)
opt = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def roi():
    opt.zero_grad()
    yp = net(x)
    loss = torch.sum(yp)
    loss.backward()
    opt.step()

benchmark_wrapper2(roi)
