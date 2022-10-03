import torch
import torchvision
from torch.profiler import profile, record_function, ProfilerActivity
import time
import sys
import os
import argparse
import numpy as np

import sys
import os

CASIO=os.environ.get('CASIO')
sys.path.append(f'{CASIO}/utils')
import cudaprofile
from torch_wrapper import benchmark_wrapper

devname = os.environ.get('DEV', 'cuda:0')
print(f'Device: {devname}')
mode = sys.argv[1]
B = int(sys.argv[2])
W = 20
N = int(sys.argv[3])

print(f'Batch Size: {B}, Warmup: {W} iters, Benchmark: {N} iters')
dev = torch.device(devname)

net = torchvision.models.resnet50(pretrained=False).half().eval().to(dev)
x = torch.randn((B, 3, 224, 224), device=dev, dtype=torch.float16)
opt = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def roi():
    opt.zero_grad()
    yp = net(x)
    loss = torch.sum(yp)
    loss.backward()
    opt.step()
    torch.cuda.synchronize()

benchmark_wrapper('resnet50', roi, mode, W, N, B)
