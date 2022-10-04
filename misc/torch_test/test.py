import os
import sys
CASIO=os.environ.get('CASIO')
sys.path.append(f'{CASIO}/utils')
import params
import cudaprofile
from torch_wrapper import benchmark_wrapper

import torch


net = torch.nn.Linear(1024, 1024)
x = torch.randn(params.bs, 1024)
opt = torch.optim.SGD(net.parameters(), lr=0.1)

def roi():
    opt.zero_grad()
    y = net(x)
    y.sum().backward()
    opt.step()

benchmark_wrapper('torch_test', roi)
