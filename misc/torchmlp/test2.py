
import sys
import os

CASIO=os.environ.get('CASIO')
sys.path.append(f'{CASIO}/utils')
# import cudaprofile
# from torch_wrapper import benchmark_wrapper
# import params
import torch

import torchvision
from torch.profiler import profile, record_function, ProfilerActivity

class Mlp(torch.nn.Module):
    def __init__(self, widths : list[int]):
        super(Mlp, self).__init__()
        self.layers = torch.nn.Sequential(*[
            torch.nn.Linear(widths[i], widths[i + 1])
            for i in range(len(widths) - 1)
        ])

    def forward(self, x):
        return self.layers(x)

def make_mlp(widths, layer_norm : bool):
    mlp = Mlp(widths)
    if layer_norm:
        mlp = torch.nn.Sequential(mlp, torch.nn.LayerNorm(384))
    return mlp

#device = torch.device("cuda:0")
device = torch.device("cpu")


class StackedMlp(torch.nn.Module):
    def __init__(self, widthss : list[list[int]], norms : list[bool]):
        super(StackedMlp, self).__init__()
        self.layers = torch.nn.Sequential(*[
            make_mlp(widthss[i], norms[i])
            for i in range(len(widthss) - 1)
        ])

    def forward(self, x):
        return self.layers(x)

net = StackedMlp(
    [
        [384,128,128,384],
    ] * 15,
    [
        True
    ] * 15)

net.to(device)

opt = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


x = torch.randn(950, 384).to(device)

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    for _ in range(100):
        with record_function("forward"):
            yp = net(x)

        with record_function("backward"):
            loss = torch.sum(yp)
            loss.backward()
            opt.step()

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))


